"""
three latent factor models for asset pricing: PCA (linear baseline), a plain
autoencoder (nonlinear, cross-sectional), and CAE (conditional autoencoder with
residual decoder). all models expose .fit(), .get_factors(), and .reconstruct().
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA as SklearnPCA
from typing import Optional, Tuple


# convert ndarray to float32 tensor on device
def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def orthonormality_penalty(factors: torch.Tensor) -> torch.Tensor:
    """penalise deviation of F'F/T from identity. resolves rotational indeterminacy.
    params: {factors: (T, K) tensor}. returns scalar tensor."""
    T = factors.shape[0]
    gram     = (factors.T @ factors) / T                           # (K, K)
    identity = torch.eye(factors.shape[1], device=factors.device)
    return torch.norm(gram - identity, p="fro") ** 2


class PCAModel:
    """linear factor model via PCA. params: {n_factors: int}."""

    def __init__(self, n_factors: int = 5):
        self.n_factors = n_factors
        self._pca: Optional[SklearnPCA] = None

    def fit(self, returns: np.ndarray, **kwargs) -> "PCAModel":
        """fits PCA on training returns. params: {returns: (T, N) ndarray with NaN for missing}. returns PCAModel."""
        # NaNs filled with column means; all-NaN columns fall back to 0.0
        col_means = np.nanmean(returns, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        filled = np.where(np.isnan(returns), col_means[None, :], returns)

        self._pca = SklearnPCA(n_components=self.n_factors, random_state=42)
        self._pca.fit(filled)
        return self

    def get_factors(self, returns: np.ndarray, **kwargs) -> np.ndarray:
        """projects returns onto fitted principal components. returns (T, K) ndarray."""
        col_means = np.nanmean(returns, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        filled = np.where(np.isnan(returns), col_means[None, :], returns)
        return self._pca.transform(filled)   # (T, K)

    def reconstruct(self, returns: np.ndarray, **kwargs) -> np.ndarray:
        """reconstruct returns from K principal components. uses realized returns — not OOS-safe. returns (T, N) ndarray."""
        factors = self.get_factors(returns)          # (T, K)
        return self._pca.inverse_transform(factors)  # (T, N)

    def predict(self, returns: np.ndarray,
                train_returns: np.ndarray, **kwargs) -> np.ndarray:
        """OOS-safe forecast: loadings × mean training factor. uses raw (uncentered)
        projections because sklearn PCA centers data, making get_factors().mean() always 0.
        returns (T, N) ndarray broadcast constant across all months."""
        col_means = np.nanmean(train_returns, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        train_filled = np.where(np.isnan(train_returns), col_means[None, :], train_returns)

        components = self._pca.components_                  # (K, N)
        # raw projections — no centering subtracted
        raw_factors  = train_filled @ components.T          # (T_train, K)
        mean_raw_fac = raw_factors.mean(axis=0)             # (K,)
        # K-dimensional approximation of mean cross-sectional return
        r_hat_const  = mean_raw_fac @ components            # (N,)

        return np.repeat(r_hat_const[None, :], returns.shape[0], axis=0)  # (T, N)


class IPCAModel:
    """IPCA estimated via alternating least squares. loadings are linear in characteristics:
    β_{i,t} = Γ z_{i,t}. Γ is used to warm-start the CAE decoder's W_skip.
    params: {n_factors: int, max_iter: int, tol: float}."""

    def __init__(self, n_factors: int, max_iter: int = 100, tol: float = 1e-4):
        self.n_factors = n_factors
        self.max_iter  = max_iter
        self.tol       = tol
        self.gamma: Optional[np.ndarray] = None   # (K, P) after fitting

    def fit(self, returns: np.ndarray, chars: np.ndarray) -> "IPCAModel":
        """fits IPCA via ALS. alternates between estimating f_t (OLS per t) and
        updating Γ (pooled OLS). inner stock sums are vectorised; only non-NaN
        pairs are used. params: {returns: (T, N), chars: (N, T, P)}. returns IPCAModel."""
        T, N = returns.shape
        P    = chars.shape[2]
        K    = self.n_factors

        # initialise Γ: orthonormal rows via QR when K ≤ P, unit-norm random rows otherwise
        rng   = np.random.default_rng(42)
        Gamma = rng.standard_normal((K, P))
        if K <= P:
            Q, _ = np.linalg.qr(Gamma.T)   # Q: (P, K) — orthonormal columns
            Gamma = Q.T                     # (K, P)
        else:
            norms = np.linalg.norm(Gamma, axis=1, keepdims=True)
            Gamma = Gamma / np.maximum(norms, 1e-10)  # (K, P), unit-norm rows

        for iteration in range(self.max_iter):
            # step 1: estimate factors given Γ
            factors = np.zeros((T, K))
            for t in range(T):
                r_t = returns[t, :]         # (N,)
                z_t = chars[:, t, :]        # (N, P)
                valid = ~np.isnan(r_t) & ~np.any(np.isnan(z_t), axis=1)
                if valid.sum() < K:
                    continue
                Z_v = z_t[valid]            # (N_t, P)
                r_v = r_t[valid]            # (N_t,)
                B_t = Z_v @ Gamma.T         # (N_t, K)  loadings matrix
                # f_t = (B_t'B_t)^{-1} B_t' r_t (OLS)
                try:
                    factors[t] = np.linalg.solve(B_t.T @ B_t, B_t.T @ r_v)
                except np.linalg.LinAlgError:
                    factors[t] = np.linalg.lstsq(B_t, r_v, rcond=None)[0]

            # step 2: update Γ given {f_t} — accumulate normal equations over t
            KP = K * P
            XX = np.zeros((KP, KP))
            Xy = np.zeros(KP)

            for t in range(T):
                r_t = returns[t, :]
                z_t = chars[:, t, :]
                valid = ~np.isnan(r_t) & ~np.any(np.isnan(z_t), axis=1)
                if valid.sum() == 0:
                    continue
                Z_v = z_t[valid]
                r_v = r_t[valid]
                f_t = factors[t]

                XX += np.kron(np.outer(f_t, f_t), Z_v.T @ Z_v)
                Xy += np.kron(f_t, Z_v.T @ r_v)

            try:
                gamma_vec = np.linalg.solve(XX, Xy)
            except np.linalg.LinAlgError:
                gamma_vec = np.linalg.lstsq(XX, Xy, rcond=None)[0]

            Gamma_new = gamma_vec.reshape(K, P)
            delta     = np.linalg.norm(Gamma_new - Gamma, "fro")
            Gamma     = Gamma_new

            if delta < self.tol:
                print(f"    IPCA converged at iteration {iteration + 1}  "
                      f"delta={delta:.2e}")
                break

        self.gamma          = Gamma
        self.factors_train_ = factors.copy()       # (T, K)
        self.factor_mean_   = factors.mean(axis=0) # (K,)
        return self

    def predict(self,
                returns: np.ndarray,
                chars: np.ndarray,
                train_returns: np.ndarray = None,
                train_chars: np.ndarray = None,
                **kwargs) -> np.ndarray:
        """OOS-safe forecast: r̂_{i,t} = z_{i,t}' Γ' f̄_train. characteristics are
        lagged so no lookahead. returns (T, N) ndarray with NaN where chars are missing."""
        assert self.gamma is not None, "Call fit() first."

        import warnings
        factor_norm = np.linalg.norm(self.factor_mean_)
        if factor_norm < 1e-4:
            warnings.warn(
                f"IPCA K={self.n_factors}: mean factor norm is {factor_norm:.2e} — "
                f"very close to zero. Predictions will be near-zero; "
                f"Pred R² will be close to zero (or negative). "
                f"This is expected when factors are mean-zero over the training period.",
                UserWarning, stacklevel=2,
            )

        T, N = returns.shape
        Z = chars.transpose(1, 0, 2)   # (T, N, P)
        r_hat = np.full((T, N), np.nan)
        for t in range(T):
            Z_t   = Z[t]
            valid = ~np.any(np.isnan(Z_t), axis=1)
            if valid.sum() == 0:
                continue
            B_t = Z_t[valid] @ self.gamma.T       # (N_v, K)
            r_hat[t, valid] = B_t @ self.factor_mean_
        return r_hat

    def reconstruct(self,
                    returns: np.ndarray,
                    chars: np.ndarray = None,
                    **kwargs) -> np.ndarray:
        """in-sample reconstruction via realized f_t. not OOS-safe; use for total R² only.
        returns (T, N) ndarray with NaN where data is unavailable."""
        assert self.gamma is not None, "Call fit() first."
        T, N = returns.shape
        K    = self.n_factors
        Z    = chars.transpose(1, 0, 2)   # (T, N, P)
        r_hat = np.full((T, N), np.nan)
        for t in range(T):
            r_t   = returns[t]
            Z_t   = Z[t]
            valid = np.isfinite(r_t) & ~np.any(np.isnan(Z_t), axis=1)
            if valid.sum() < K + 1:
                continue
            B_t = Z_t[valid] @ self.gamma.T       # (N_v, K)
            try:
                f_t = np.linalg.lstsq(B_t, r_t[valid], rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            valid_z = ~np.any(np.isnan(Z_t), axis=1)
            r_hat[t, valid_z] = (Z_t[valid_z] @ self.gamma.T) @ f_t
        return r_hat


class _AENet(nn.Module):
    """feedforward autoencoder on the full cross-sectional return vector.
    encoder: N→128→64→K (ReLU); decoder: K→64→128→N (linear output).
    batch norm omitted — behaves inconsistently with unbalanced panels."""

    def __init__(self, n_assets: int, n_factors: int):
        super().__init__()
        N, K = n_assets, n_factors

        self.encoder = nn.Sequential(
            nn.Linear(N, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, K),
        )
        self.decoder = nn.Sequential(
            nn.Linear(K, 64),  nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, N),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """params: {x: (batch, N) with NaN replaced by 0}. returns (factors (batch, K), r_hat (batch, N))."""
        factors = self.encoder(x)
        r_hat   = self.decoder(factors)
        return factors, r_hat


class AutoencoderModel:
    """wrapper around _AENet. params: {n_factors: int, device: str}."""

    def __init__(self, n_factors: int = 5, device: str = "cpu"):
        self.n_factors = n_factors
        self.device = torch.device(device)
        self.net: Optional[_AENet] = None

    def _make_input(self, returns: np.ndarray) -> torch.Tensor:
        """replace NaNs with 0 and convert to tensor. (T, N) → tensor."""
        filled = np.nan_to_num(returns, nan=0.0)
        return _to_tensor(filled, self.device)

    def fit(self, returns: np.ndarray, **kwargs) -> "AutoencoderModel":
        """initialises the network from asset dimension. training handled by src/train.py."""
        N = returns.shape[1]
        self.net = _AENet(N, self.n_factors).to(self.device)
        return self

    def get_factors(self, returns: np.ndarray, **kwargs) -> np.ndarray:
        """runs the encoder on returns. returns (T, K) ndarray."""
        assert self.net is not None, "Call fit() first."
        self.net.eval()
        with torch.no_grad():
            x = self._make_input(returns)
            factors, _ = self.net(x)
        return factors.cpu().numpy()

    def reconstruct(self, returns: np.ndarray, **kwargs) -> np.ndarray:
        """encode then decode returns. uses current-period returns — not OOS-safe.
        use for total R² only. returns (T, N) ndarray."""
        assert self.net is not None, "Call fit() first."
        self.net.eval()
        with torch.no_grad():
            x = self._make_input(returns)
            _, r_hat = self.net(x)
        return r_hat.cpu().numpy()

    def predict(self, returns: np.ndarray,
                train_returns: np.ndarray, **kwargs) -> np.ndarray:
        """OOS-safe forecast: decoder(mean training factor). mean factor substitutes
        for unknown current-period factor. returns (T_test, N) ndarray."""
        assert self.net is not None, "Call fit() first."
        mean_factor = self.get_factors(train_returns).mean(axis=0)  # (K,)
        factor_tensor = _to_tensor(
            np.tile(mean_factor, (returns.shape[0], 1)), self.device  # (T, K)
        )
        self.net.eval()
        with torch.no_grad():
            r_hat = self.net.decoder(factor_tensor)
        return r_hat.cpu().numpy()


class _CAEEncoder(nn.Module):
    """maps managed portfolio returns r̃_t = Z_t' r_t to K latent factors.
    architecture: P→32→K (single hidden layer). batch norm omitted to avoid
    instability with unbalanced panels. params: {n_chars: int, n_factors: int}."""

    def __init__(self, n_chars: int, n_factors: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_chars, 32), nn.ReLU(),
            nn.Linear(32, n_factors),
        )

    def forward(self, managed: torch.Tensor) -> torch.Tensor:
        """params: {managed: (batch, P)}. returns factors (batch, K)."""
        return self.net(managed)


class _CAEDecoder(nn.Module):
    """maps stock characteristics to beta loadings via residual connection:
    β_{i,t} = W_skip·z_{i,t} + g(z_{i,t}). W_skip warm-started from IPCA so
    g learns only the residual. weights are shared across stocks.
    params: {n_chars: int, n_factors: int, use_linear: bool}."""

    def __init__(self, n_chars: int, n_factors: int, use_linear: bool = True):
        super().__init__()
        self.use_linear = use_linear

        # omitted for NL-only ablation
        if use_linear:
            self.W_skip = nn.Linear(n_chars, n_factors, bias=False)

        # nonlinear correction network
        self.g = nn.Sequential(
            nn.Linear(n_chars, 32), nn.ReLU(),
            nn.Linear(32, 32),      nn.ReLU(),
            nn.Linear(32, n_factors),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """params: {z: (batch_stocks, P)}. returns (beta (batch_stocks, K), beta_nonlin (batch_stocks, K))."""
        beta_nonlin = self.g(z)           # (batch_stocks, K)
        if self.use_linear:
            beta = self.W_skip(z) + beta_nonlin
        else:
            beta = beta_nonlin
        return beta, beta_nonlin


class _CAENet(nn.Module):
    """full CAE: managed-portfolio encoder + residual decoder."""

    def __init__(self, n_chars: int, n_factors: int, use_linear: bool = True):
        super().__init__()
        self.encoder = _CAEEncoder(n_chars, n_factors)
        self.decoder = _CAEDecoder(n_chars, n_factors, use_linear=use_linear)

    def forward(self,
                managed: torch.Tensor,
                z_stocks: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """params: {managed: (T_batch, P), z_stocks: (T_batch, N_max, P)}.
        returns (r_hat (T_batch, N_max), beta (T, N, K), beta_nonlin (T, N, K), factors (T_batch, K))."""
        T_batch, N_max, P = z_stocks.shape

        factors = self.encoder(managed)                          # (T_batch, K)

        # flatten stock dimension for a single decoder forward pass
        z_flat = z_stocks.reshape(T_batch * N_max, P)           # (T*N, P)
        beta_flat, beta_nonlin_flat = self.decoder(z_flat)      # (T*N, K)

        beta        = beta_flat.reshape(T_batch, N_max, -1)     # (T, N, K)
        beta_nonlin = beta_nonlin_flat.reshape(T_batch, N_max, -1)

        # r̂_{i,t} = β_{i,t}' f_t; broadcast factors (T,K) → (T,1,K)
        r_hat = (beta * factors.unsqueeze(1)).sum(dim=-1)       # (T, N)

        return r_hat, beta, beta_nonlin, factors


class CAEModel:
    """CAE with residual decoder. combines a characteristics-based encoder with a nonlinear
    decoder that nests IPCA as a special case. when freeze_linear=True (ResCAE-Fixed),
    W_skip is frozen at the IPCA solution and only encoder + g are updated.
    params: {n_chars: int, n_factors: int, lam_lin: float, lam_nonlin: float,
    use_linear: bool, freeze_linear: bool, device: str}."""

    def __init__(self,
                 n_chars: int = 7,
                 n_factors: int = 5,
                 lam_lin: float = 0.0,
                 lam_nonlin: float = 0.0,
                 use_linear: bool = True,
                 freeze_linear: bool = False,
                 device: str = "cpu"):
        self.n_chars       = n_chars
        self.n_factors     = n_factors
        self.lam_lin       = lam_lin
        self.lam_nonlin    = lam_nonlin
        self.use_linear    = use_linear
        self.freeze_linear = freeze_linear
        self.device        = torch.device(device)
        self.net: Optional[_CAENet] = None

    @property
    def model_name(self) -> str:
        if self.freeze_linear and self.use_linear:
            return "ResCAE-Fixed"
        elif self.use_linear:
            return "ResCAE"
        else:
            return "CAE-NL"

    def fit(self, **kwargs) -> "CAEModel":
        """initialises the network. training handled by src/train.py."""
        self.net = _CAENet(self.n_chars, self.n_factors,
                           use_linear=self.use_linear).to(self.device)
        self.gamma_init: Optional[np.ndarray] = None

        # init g with small weights so predictions start near zero — default Kaiming
        # gives |g(z)| ~ 0.3, ~10× larger than typical monthly returns, which
        # destabilises the encoder gradient and can collapse factors to near-zero mean
        for module in self.net.decoder.g.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

        return self

    def initialize_from_ipca(self,
                              returns: np.ndarray,
                              chars: np.ndarray) -> None:
        """warm-starts W_skip from an IPCA fit; reinitialises g with std=0.01 so
        the nonlinear correction starts negligible. params: {returns: (T, N), chars: (N, T, P)}."""
        assert self.net is not None, "Call fit() before initialize_from_ipca()."
        if not self.use_linear:
            return  # NL-only variant has no W_skip to warm-start
        print(f"    Fitting IPCA for warm-start (K={self.n_factors}) …")
        ipca = IPCAModel(n_factors=self.n_factors)
        ipca.fit(returns, chars)
        ipca_gamma = ipca.gamma   # (K, P)

        gamma_tensor = torch.tensor(ipca_gamma, dtype=torch.float32,
                                    device=self.device)
        with torch.no_grad():
            self.net.decoder.W_skip.weight.copy_(gamma_tensor)

        assert torch.allclose(
            self.net.decoder.W_skip.weight, gamma_tensor, atol=1e-5
        ), "W_skip weight copy failed."

        for module in self.net.decoder.g.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

        self.gamma_init = ipca_gamma.copy()
        norm = np.linalg.norm(ipca_gamma, "fro")
        print(f"    IPCA init done: ‖Γ‖_F = {norm:.4f}")

        if self.freeze_linear:
            for param in self.net.decoder.W_skip.parameters():
                param.requires_grad = False
            print(f"    [ResCAE-Fixed] W_skip frozen at IPCA solution. "
                  f"Only encoder and g will be updated.")

    def compute_ipca_drift(self) -> Optional[float]:
        """relative Frobenius drift of W_skip from IPCA init. returns 0.0 for
        ResCAE-Fixed (frozen by construction), None if no linear branch or no warm-start."""
        if not self.use_linear:
            return None
        if self.freeze_linear:
            return 0.0
        if not hasattr(self, "gamma_init") or self.gamma_init is None:
            return None
        W     = self.net.decoder.W_skip.weight.detach().cpu().numpy()
        gamma = self.gamma_init
        denom = np.linalg.norm(gamma, "fro")
        if denom < 1e-10:
            return None
        return float(np.linalg.norm(W - gamma, "fro") / denom)

    def l1_penalty(self) -> torch.Tensor:
        """L1 regularisation on decoder weights with separate lambdas for W_skip and g.
        separate lambdas prevent g being over-penalised relative to W_skip due to
        parameter count differences. returns scalar tensor."""
        if self.lam_lin > 0 and not self.use_linear:
            import warnings
            warnings.warn(
                "lam_lin > 0 but use_linear=False: the linear branch W_skip does "
                "not exist in this model. lam_lin is being ignored. Set lam_lin=0.0 "
                "for NL-only configurations to suppress this warning.",
                UserWarning, stacklevel=2,
            )
        penalty = torch.tensor(0.0, device=self.device)
        if self.use_linear and self.lam_lin > 0:
            penalty = penalty + self.lam_lin * sum(
                p.abs().sum() for p in self.net.decoder.W_skip.parameters()
            )
        if self.lam_nonlin > 0:
            penalty = penalty + self.lam_nonlin * sum(
                p.abs().sum() for p in self.net.decoder.g.parameters()
            )
        return penalty

    def _build_batch(self,
                     returns: np.ndarray,
                     chars: np.ndarray,
                     t_indices: np.ndarray,
                     ) -> Tuple[torch.Tensor, torch.Tensor,
                                torch.Tensor, torch.Tensor]:
        """builds (managed, z_stocks, r_true, mask) tensors for a mini-batch of time indices.
        NaNs zeroed out; mask tracks valid stocks per t. returns tensors on self.device."""
        T_b  = len(t_indices)
        T, N = returns.shape       # full time × stocks
        P    = chars.shape[2]      # characteristics per stock

        managed_np  = np.zeros((T_b, P), dtype=np.float32)
        z_np        = np.zeros((T_b, N, P), dtype=np.float32)
        r_np        = np.zeros((T_b, N), dtype=np.float32)
        mask_np     = np.zeros((T_b, N), dtype=np.float32)

        for b, t in enumerate(t_indices):
            r_t = returns[t, :]      # (N,)  — NaN where missing
            z_t = chars[:, t, :]     # (N, P) — NaN where missing

            valid = ~np.isnan(r_t) & ~np.any(np.isnan(z_t), axis=1)
            mask_np[b, valid] = 1.0

            r_clean = np.nan_to_num(r_t, nan=0.0)
            z_clean = np.nan_to_num(z_t, nan=0.0)

            # divide by N_t to normalise for varying cross-section size across time
            n_valid = valid.sum()
            managed_np[b] = (z_clean[valid].T @ r_t[valid] / n_valid
                             if n_valid > 0 else np.zeros(P))

            z_np[b]   = z_clean
            r_np[b]   = r_clean

        managed  = _to_tensor(managed_np,  self.device)
        z_stocks = _to_tensor(z_np,        self.device)
        r_true   = _to_tensor(r_np,        self.device)
        mask     = _to_tensor(mask_np,     self.device)

        return managed, z_stocks, r_true, mask

    def get_factors(self,
                    returns: np.ndarray,
                    chars: np.ndarray) -> np.ndarray:
        """encodes managed portfolio returns. returns (T, K) factor matrix."""
        assert self.net is not None
        T, N = returns.shape
        P = chars.shape[2]
        t_indices = np.arange(T)
        managed, z_stocks, r_true, mask = self._build_batch(
            returns, chars, t_indices)

        self.net.eval()
        with torch.no_grad():
            factors = self.net.encoder(managed)
        return factors.cpu().numpy()

    def reconstruct(self,
                    returns: np.ndarray,
                    chars: np.ndarray) -> np.ndarray:
        """reconstructs returns using realized managed portfolio. not OOS-safe;
        use for total R² only. returns (T, N) ndarray with NaN where input was NaN."""
        assert self.net is not None
        T, N = returns.shape
        t_indices = np.arange(T)
        managed, z_stocks, r_true, mask = self._build_batch(
            returns, chars, t_indices)

        self.net.eval()
        with torch.no_grad():
            r_hat, _, _, _ = self.net(managed, z_stocks)

        r_hat_np = r_hat.cpu().numpy()  # (T, N)
        nan_mask = np.isnan(returns) | np.any(np.isnan(chars.transpose(1, 0, 2)), axis=2)
        r_hat_np[nan_mask] = np.nan
        return r_hat_np

    def predict(self,
                returns: np.ndarray,
                chars: np.ndarray,
                train_returns: np.ndarray,
                train_chars: np.ndarray) -> np.ndarray:
        """OOS-safe forecast: r̂_{i,t} = β_{i,t}' f̄_train. mean training factor
        substitutes for unobservable current-period factor.
        returns (T_test, N) ndarray with NaN where input was NaN."""
        assert self.net is not None
        mean_factor_np = self.get_factors(train_returns, train_chars).mean(axis=0)

        # near-zero f̄_train is the primary cause of flat predictions
        f_bar_norm = float(np.linalg.norm(mean_factor_np))
        print(f"  [{self.model_name} K={self.n_factors}] "
              f"‖f̄_train‖ = {f_bar_norm:.4f}  "
              f"(< 1e-3 → predictions will be near-zero for all stocks)")

        mean_factor = _to_tensor(mean_factor_np, self.device)  # (K,)

        T, N = returns.shape
        t_indices = np.arange(T)
        # only z_stocks needed here; other outputs unused
        _, z_stocks, _, _ = self._build_batch(returns, chars, t_indices)

        self.net.eval()
        with torch.no_grad():
            T_b, N_max, P = z_stocks.shape
            z_flat = z_stocks.reshape(T_b * N_max, P)
            beta_flat, _ = self.net.decoder(z_flat)
            beta = beta_flat.reshape(T_b, N_max, -1)             # (T, N, K)
            # r̂_{i,t} = β_{i,t}' f̄ — broadcast mean factor across all (t, i)
            r_hat = (beta * mean_factor.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (T, N)

        r_hat_np = r_hat.cpu().numpy()
        nan_mask = np.isnan(returns) | np.any(np.isnan(chars.transpose(1, 0, 2)), axis=2)
        r_hat_np[nan_mask] = np.nan

        # flag flat predictions — indicates collapsed g(z) or near-zero f̄_train
        cs_std = float(np.nanstd(r_hat_np))
        if cs_std < 1e-6:
            beta_np = beta_flat.cpu().numpy()
            beta_cs_std = float(np.std(beta_np))
            print(f"  [{self.model_name} K={self.n_factors}] "
                  f"WARNING: predictions are flat (cs_std={cs_std:.2e}).  "
                  f"beta cs_std={beta_cs_std:.4f}, ‖f̄‖={f_bar_norm:.4e}.  "
                  f"{'g(z) is near-constant' if beta_cs_std < 1e-4 else 'f̄_train is near-zero'}")

        return r_hat_np

    def get_nonlinear_contribution(self,
                                   returns: np.ndarray,
                                   chars: np.ndarray,
                                   ) -> Tuple[float, float]:
        """decomposes loading variance into linear vs nonlinear parts.
        returns (var_lin, var_nonlin) across all valid (i, t) pairs."""
        assert self.net is not None
        T, N = returns.shape
        t_indices = np.arange(T)
        managed, z_stocks, r_true, mask = self._build_batch(
            returns, chars, t_indices)

        self.net.eval()
        with torch.no_grad():
            T_b, N_max, P = z_stocks.shape
            z_flat = z_stocks.reshape(T_b * N_max, P)
            beta_nonlin = self.net.decoder.g(z_flat)
            if self.use_linear:
                # when frozen, W_skip = Gamma_IPCA throughout, so var_lin = IPCA's loading variance
                beta_lin = self.net.decoder.W_skip(z_flat)
            else:
                # NL-only: no linear branch, so var_lin = 0 and frac_nonlin = 1.0 always
                beta_lin = torch.zeros_like(beta_nonlin)

        mask_flat = mask.reshape(-1).cpu().numpy().astype(bool)
        bl = beta_lin.cpu().numpy().reshape(-1, self.n_factors)[mask_flat]
        bn = beta_nonlin.cpu().numpy().reshape(-1, self.n_factors)[mask_flat]

        var_lin    = float(np.var(bl))
        var_nonlin = float(np.var(bn))

        # collapsed g(z) makes phi=1.0 meaningless for NL-only; return (0,0) so caller reports N/A
        if var_nonlin < 1e-10 and not self.use_linear:
            return 0.0, 0.0

        return var_lin, var_nonlin
