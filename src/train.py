"""
training routines for the AE and CAE models.

mini-batches slice the time dimension; at each step all available stocks contribute to the loss.
early stopping monitors validation loss and restores best weights.
"""

import os
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, List, Tuple, Optional

from src.models import PCAModel, AutoencoderModel, CAEModel, IPCAModel, orthonormality_penalty


LR            = 1e-3
BATCH_SIZE    = 32       # time steps per mini-batch
MAX_EPOCHS    = 400      # upper bound; early stopping fires before this in practice
PATIENCE      = 25       # early stopping patience in epochs
K_GRID             = [2,3,5,8,10]
LAMBDA_LIN_GRID    = [1e-03, 5e-03, 1e-02, 5e-02, 1e-01]   # L1 on W_skip (linear branch)
LAMBDA_NONLIN_GRID = [1e-06, 1e-05, 5e-05, 1e-04, 5e-04]   # L1 on g (nonlinear branch)
LAMBDA_ORTH   = 1e-3     # weight on F'F/T = I_K penalty; resolves rotational indeterminacy
SEEDS         = [10, 30, 45, 99, 2048]   # S=5 seeds for multi-run experiments


# fit PCA on training returns
def train_pca(splits: dict, k: int) -> PCAModel:
    """fit PCA and return the model. params: {splits: dict, k: int}. returns PCAModel."""
    train_ret = splits["train"]["returns"].values  # (T_train, N)
    model = PCAModel(n_factors=k)
    model.fit(train_ret)
    print(f"  PCA K={k}: fitted on {train_ret.shape[0]} months, "
          f"{train_ret.shape[1]} stocks.")
    return model


# fit IPCA on training returns and characteristics
def train_ipca(splits: dict, k: int) -> IPCAModel:
    """fit IPCA and return the model. params: {splits: dict, k: int}. returns IPCAModel."""
    train_ret   = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)
    model = IPCAModel(n_factors=k)
    model.fit(train_ret, train_chars)
    print(f"  IPCA K={k}: fitted on {train_ret.shape[0]} months, "
          f"{train_ret.shape[1]} stocks.")
    return model


# MSE over valid (non-NaN) entries; NaN positions are zeroed in the tensor and excluded via mask
def _ae_loss(r_true: torch.Tensor,
             r_hat: torch.Tensor,
             nan_mask: torch.Tensor) -> torch.Tensor:
    sq_err = (r_true - r_hat) ** 2          # (batch, N)
    sq_err = sq_err * nan_mask              # zero out missing entries
    n_valid = nan_mask.sum().clamp(min=1)
    return sq_err.sum() / n_valid


def train_autoencoder(splits: dict,
                      k: int,
                      device: str = "cpu",
                      ) -> AutoencoderModel:
    """train AE with early stopping; returns best-checkpoint model. params: {splits: dict, k: int, device: str}. returns AutoencoderModel."""
    train_ret = splits["train"]["returns"].values.astype(np.float32)  # (T_tr, N)
    val_ret   = splits["val"]["returns"].values.astype(np.float32)    # (T_val, N)
    N = train_ret.shape[1]

    model = AutoencoderModel(n_factors=k, device=device)
    model.fit(train_ret)
    net = model.net
    optimizer = Adam(net.parameters(), lr=LR)

    # pre-build NaN masks once to avoid recomputing each epoch
    nan_mask_train = torch.tensor(~np.isnan(train_ret), dtype=torch.float32,
                                  device=model.device)
    nan_mask_val   = torch.tensor(~np.isnan(val_ret),   dtype=torch.float32,
                                  device=model.device)

    r_train_tensor = torch.tensor(np.nan_to_num(train_ret, nan=0.0),
                                  dtype=torch.float32, device=model.device)
    r_val_tensor   = torch.tensor(np.nan_to_num(val_ret,   nan=0.0),
                                  dtype=torch.float32, device=model.device)

    T_train = train_ret.shape[0]
    best_val_loss   = float("inf")
    best_state      = None
    patience_count  = 0
    train_losses: List[float] = []
    val_losses:   List[float] = []

    for epoch in range(MAX_EPOCHS):
        net.train()
        # shuffle time indices each epoch
        idx = np.random.permutation(T_train)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, T_train, BATCH_SIZE):
            batch_idx = idx[start : start + BATCH_SIZE]
            r_b = r_train_tensor[batch_idx]
            m_b = nan_mask_train[batch_idx]

            optimizer.zero_grad()
            factors, r_hat = net(r_b)
            loss = _ae_loss(r_b, r_hat, m_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        # full-sample orth penalty after mini-batch pass; separate step avoids
        # interference between MSE and orthonormality gradients
        net.train()
        optimizer.zero_grad()
        all_factors, _ = net(r_train_tensor)
        orth_loss = LAMBDA_ORTH * orthonormality_penalty(all_factors)
        orth_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        # validation loss
        net.eval()
        with torch.no_grad():
            _, r_hat_val = net(r_val_tensor)
            vl = _ae_loss(r_val_tensor, r_hat_val, nan_mask_val).item()
        val_losses.append(vl)

        if vl < best_val_loss - 1e-7:
            best_val_loss  = vl
            best_state     = copy.deepcopy(net.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if (epoch + 1) % 20 == 0:
            print(f"    AE K={k}  epoch {epoch+1:3d}  "
                  f"train={train_losses[-1]:.6f}  val={vl:.6f}")

        if patience_count >= PATIENCE:
            print(f"    Early stop at epoch {epoch+1}  best_val={best_val_loss:.6f}")
            break

    # restore best weights before returning
    net.load_state_dict(best_state)

    return model


# MSE over valid stock-month pairs for the CAE
def _cae_loss(r_hat: torch.Tensor,
              r_true: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    sq_err  = (r_hat - r_true) ** 2 * mask
    n_valid = mask.sum().clamp(min=1)
    return sq_err.sum() / n_valid


def _run_cae_epoch(model: CAEModel,
                   returns: np.ndarray,
                   chars: np.ndarray,
                   t_indices: np.ndarray,
                   optimizer: Optional[Adam],
                   train: bool) -> float:
    """run one epoch (or eval pass) of the CAE, batching over the time dimension.
    params: {model: CAEModel, returns: np.ndarray, chars: np.ndarray,
    t_indices: np.ndarray, optimizer: Adam|None, train: bool}. returns float loss."""
    net = model.net
    if train:
        net.train()
    else:
        net.eval()

    np.random.shuffle(t_indices) if train else None
    total_loss = 0.0
    n_batches  = 0

    for start in range(0, len(t_indices), BATCH_SIZE):
        batch_t = t_indices[start : start + BATCH_SIZE]
        managed, z_stocks, r_true, mask = model._build_batch(
            returns, chars, batch_t)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            r_hat, _, _, _ = model.net(managed, z_stocks)
            loss = _cae_loss(r_hat, r_true, mask)
            if train and (model.lam_lin > 0 or model.lam_nonlin > 0):
                loss = loss + model.l1_penalty()

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def train_cae_single(splits: dict,
                     k: int,
                     lam_lin: float,
                     lam_nonlin: float,
                     use_linear: bool = True,
                     freeze_linear: bool = False,
                     device: str = "cpu",
                     ) -> CAEModel:
    """train one CAE config (K, λ_lin, λ_nonlin) with early stopping; returns best-checkpoint CAEModel.
    use_linear=False omits W_skip (NL-only ablation); freeze_linear=True freezes W_skip after IPCA init."""
    train_ret   = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)   # (N, T_tr, P)
    val_ret     = splits["val"]["returns"].values.astype(np.float32)
    val_chars   = splits["val"]["chars"].astype(np.float32)

    # chars is (N, T, P); returns is (T, N) — _build_batch expects this layout
    P = train_chars.shape[2]

    model = CAEModel(n_chars=P, n_factors=k,
                     lam_lin=lam_lin, lam_nonlin=lam_nonlin,
                     use_linear=use_linear, freeze_linear=freeze_linear,
                     device=device)
    model.fit()
    if use_linear:
        model.initialize_from_ipca(train_ret, train_chars)
    optimizer = Adam(model.net.parameters(), lr=LR)

    T_train = train_ret.shape[0]
    T_val   = val_ret.shape[0]
    t_train = np.arange(T_train)
    t_val   = np.arange(T_val)

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    train_losses: List[float] = []
    val_losses:   List[float] = []

    for epoch in range(MAX_EPOCHS):
        tl = _run_cae_epoch(model, train_ret, train_chars, t_train.copy(),
                            optimizer, train=True)

        # full-sample orth penalty on all training managed portfolios
        model.net.train()
        optimizer.zero_grad()
        managed_all, _, _, _ = model._build_batch(train_ret, train_chars, t_train)
        all_factors = model.net.encoder(managed_all)
        orth_loss = LAMBDA_ORTH * orthonormality_penalty(all_factors)
        orth_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), max_norm=1.0)
        optimizer.step()

        vl = _run_cae_epoch(model, val_ret, val_chars, t_val.copy(),
                            optimizer=None, train=False)

        train_losses.append(tl)
        val_losses.append(vl)

        if vl < best_val_loss - 1e-7:
            best_val_loss  = vl
            best_state     = copy.deepcopy(model.net.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if (epoch + 1) % 20 == 0:
            tag = "CAE-NL" if not use_linear else "CAE"
            print(f"    {tag} K={k} λ_lin={lam_lin:.0e} λ_nonlin={lam_nonlin:.0e}  "
                  f"epoch {epoch+1:3d}  train={tl:.6f}  val={vl:.6f}")

        if patience_count >= PATIENCE:
            print(f"    Early stop at epoch {epoch+1}  best_val={best_val_loss:.6f}")
            break

    model.net.load_state_dict(best_state)

    # diagnostic: near-zero mean factor signals the model will predict ~0 for all stocks
    if not use_linear:
        mf = model.get_factors(train_ret, train_chars).mean(axis=0)
        mf_norm = float(np.linalg.norm(mf))
        print(f"    [CAE-NL K={k}] post-training ‖f̄_train‖ = {mf_norm:.4f}"
              + (" ← NEAR ZERO: predict() will return ~0 for all stocks"
                 if mf_norm < 1e-3 else " ← OK"))

    return model


def train_all_cae(splits: dict,
                  device: str = "cpu",
                  run_dir: str = "results",
                  ) -> Dict[Tuple[int, float, float], CAEModel]:
    """grid search over (K, λ_lin, λ_nonlin) for the CAE; returns dict mapping (k, lam_lin, lam_nonlin) → CAEModel."""
    results:      Dict[Tuple[int, float, float], CAEModel] = {}
    val_losses:   Dict[Tuple[int, float, float], float]    = {}
    pred_val_mse: Dict[Tuple[int, float, float], float]    = {}

    train_ret   = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)

    total = len(K_GRID) * len(LAMBDA_LIN_GRID) * len(LAMBDA_NONLIN_GRID)
    config_num = 0

    for k in K_GRID:
        for lam_lin in LAMBDA_LIN_GRID:
            for lam_nonlin in LAMBDA_NONLIN_GRID:
                config_num += 1
                print(f"\n  Training CAE K={k} λ_lin={lam_lin:.0e} "
                      f"λ_nonlin={lam_nonlin:.0e} ({config_num}/{total}) …")
                model = train_cae_single(splits, k, lam_lin, lam_nonlin, device)

                # reconstruction validation loss
                val_ret   = splits["val"]["returns"].values.astype(np.float32)
                val_chars = splits["val"]["chars"].astype(np.float32)
                T_val = val_ret.shape[0]
                t_val = np.arange(T_val)
                vl = _run_cae_epoch(model, val_ret, val_chars, t_val,
                                    optimizer=None, train=False)

                # OOS-safe predictive MSE: factor mean estimated from training period only
                r_hat_val = model.predict(val_ret, val_chars,
                                          train_returns=train_ret, train_chars=train_chars)
                pmask    = np.isfinite(val_ret) & np.isfinite(r_hat_val)
                pred_mse = (float(np.mean((val_ret[pmask] - r_hat_val[pmask]) ** 2))
                            if pmask.any() else float("inf"))

                results[(k, lam_lin, lam_nonlin)]      = model
                val_losses[(k, lam_lin, lam_nonlin)]   = vl
                pred_val_mse[(k, lam_lin, lam_nonlin)] = pred_mse

    best_config = min(pred_val_mse, key=pred_val_mse.get)
    print(f"\n  Best CAE config: K={best_config[0]}  "
          f"λ_lin={best_config[1]:.0e}  λ_nonlin={best_config[2]:.0e}  "
          f"pred_val_mse={pred_val_mse[best_config]:.6f}")

    # save hyperparameter search results
    summary_path = os.path.join(run_dir, "cae_hparam_search.pkl")
    os.makedirs(run_dir, exist_ok=True)
    with open(summary_path, "wb") as f:
        pickle.dump({"val_losses": val_losses,
                     "pred_val_mse": pred_val_mse,
                     "best": best_config}, f)

    return results


def train_all_cae_nonlinear(splits: dict,
                            device: str = "cpu",
                            run_dir: str = "results",
                            ) -> Dict[Tuple[int, float], CAEModel]:
    """grid search over (K, λ_nonlin) for the NL-only CAE ablation (no W_skip); returns dict mapping (k, lam_nonlin) → CAEModel."""
    results:      Dict[Tuple[int, float], CAEModel] = {}
    val_losses:   Dict[Tuple[int, float], float]    = {}
    pred_val_mse: Dict[Tuple[int, float], float]    = {}

    train_ret   = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)

    total = len(K_GRID) * len(LAMBDA_NONLIN_GRID)
    config_num = 0

    for k in K_GRID:
        for lam_nonlin in LAMBDA_NONLIN_GRID:
            config_num += 1
            print(f"\n  Training CAE-NL K={k} λ_nonlin={lam_nonlin:.0e} "
                  f"({config_num}/{total}) …")
            model = train_cae_single(
                splits, k,
                lam_lin=0.0, lam_nonlin=lam_nonlin,
                use_linear=False, device=device,
            )

            val_ret   = splits["val"]["returns"].values.astype(np.float32)
            val_chars = splits["val"]["chars"].astype(np.float32)
            T_val = val_ret.shape[0]
            t_val = np.arange(T_val)
            vl = _run_cae_epoch(model, val_ret, val_chars, t_val,
                                optimizer=None, train=False)

            # OOS-safe predictive MSE
            r_hat_val = model.predict(val_ret, val_chars,
                                      train_returns=train_ret, train_chars=train_chars)
            pmask    = np.isfinite(val_ret) & np.isfinite(r_hat_val)
            pred_mse = (float(np.mean((val_ret[pmask] - r_hat_val[pmask]) ** 2))
                        if pmask.any() else float("inf"))

            results[(k, lam_nonlin)]      = model
            val_losses[(k, lam_nonlin)]   = vl
            pred_val_mse[(k, lam_nonlin)] = pred_mse

    best_config = min(pred_val_mse, key=pred_val_mse.get)
    print(f"\n  Best CAE-NL config: K={best_config[0]}  "
          f"λ_nonlin={best_config[1]:.0e}  "
          f"pred_val_mse={pred_val_mse[best_config]:.6f}")

    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "cae_nl_hparam_search.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump({"val_losses": val_losses,
                     "pred_val_mse": pred_val_mse,
                     "best": best_config}, f)

    return results


# seed all RNGs then delegate to train_cae_single
def train_cae_single_seeded(
    splits: dict,
    k: int,
    lam_lin: float,
    lam_nonlin: float,
    seed: int,
    use_linear: bool = True,
    freeze_linear: bool = False,
    device: str = "cpu",
) -> CAEModel:
    """train a single CAE with a fixed random seed. params: {splits: dict, k: int, lam_lin: float, lam_nonlin: float, seed: int, use_linear: bool, freeze_linear: bool, device: str}. returns CAEModel."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return train_cae_single(splits, k, lam_lin, lam_nonlin,
                            use_linear=use_linear, freeze_linear=freeze_linear,
                            device=device)


def train_all_cae_multiseed(
    splits: dict,
    seeds: List[int] = SEEDS,
    device: str = "cpu",
    run_dir: str = "results",
) -> Dict[Tuple[int, float, float], List[CAEModel]]:
    """run the full CAE grid search once per seed; selects best config by mean predictive val MSE.
    saves per-seed details and a standard hparam pkl for downstream eval.
    returns dict mapping (k, lam_lin, lam_nonlin) → List[CAEModel] of length S."""
    all_models:   Dict[Tuple, List[CAEModel]] = {}
    all_pred_mse: Dict[Tuple, List[float]]    = {}

    train_ret   = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)
    val_ret     = splits["val"]["returns"].values.astype(np.float32)
    val_chars   = splits["val"]["chars"].astype(np.float32)

    total_configs = len(K_GRID) * len(LAMBDA_LIN_GRID) * len(LAMBDA_NONLIN_GRID)
    config_num = 0

    for k in K_GRID:
        for lam_lin in LAMBDA_LIN_GRID:
            for lam_nonlin in LAMBDA_NONLIN_GRID:
                config_num += 1
                key = (k, lam_lin, lam_nonlin)
                all_models[key]   = []
                all_pred_mse[key] = []

                print(f"\n  Config {config_num}/{total_configs}: "
                      f"K={k} λ_lin={lam_lin:.0e} λ_nonlin={lam_nonlin:.0e}")

                for s_idx, seed in enumerate(seeds):
                    print(f"    Seed {seed} ({s_idx+1}/{len(seeds)}) ...")
                    model = train_cae_single_seeded(
                        splits, k, lam_lin, lam_nonlin,
                        seed=seed, use_linear=True, device=device,
                    )
                    r_hat_val = model.predict(val_ret, val_chars,
                                              train_returns=train_ret,
                                              train_chars=train_chars)
                    pmask    = np.isfinite(val_ret) & np.isfinite(r_hat_val)
                    pred_mse = (float(np.mean((val_ret[pmask] - r_hat_val[pmask]) ** 2))
                                if pmask.any() else float("inf"))
                    all_models[key].append(model)
                    all_pred_mse[key].append(pred_mse)

                mean_mse = float(np.mean(all_pred_mse[key]))
                std_mse  = float(np.std(all_pred_mse[key]))
                print(f"    Mean pred val MSE: {mean_mse:.6f} ± {std_mse:.6f}")

    mean_mse_per_config = {k: float(np.mean(v)) for k, v in all_pred_mse.items()}
    best_config = min(mean_mse_per_config, key=mean_mse_per_config.get)
    print(f"\n  Best CAE config (by mean pred val MSE across {len(seeds)} seeds):")
    print(f"  K={best_config[0]}  λ_lin={best_config[1]:.0e}  "
          f"λ_nonlin={best_config[2]:.0e}  "
          f"mean_pred_val_mse={mean_mse_per_config[best_config]:.6f}")

    os.makedirs(run_dir, exist_ok=True)
    # full per-seed details
    with open(os.path.join(run_dir, "cae_multiseed_hparam_search.pkl"), "wb") as f:
        pickle.dump({"all_pred_mse": all_pred_mse,
                     "mean_mse_per_config": mean_mse_per_config,
                     "best": best_config, "seeds": seeds}, f)
    # standard pkl with mean scores so evaluate.py can read it transparently
    with open(os.path.join(run_dir, "cae_hparam_search.pkl"), "wb") as f:
        pickle.dump({"val_losses":   mean_mse_per_config,
                     "pred_val_mse": mean_mse_per_config,
                     "best": best_config}, f)

    return all_models


def train_all_cae_nl_multiseed(
    splits: dict,
    seeds: List[int] = SEEDS,
    device: str = "cpu",
    run_dir: str = "results",
) -> Dict[Tuple[int, float], List[CAEModel]]:
    """run the CAE-NL grid search once per seed; saves per-seed and standard hparam pkls.
    returns dict mapping (k, lam_nonlin) → List[CAEModel] of length S."""
    all_models:   Dict[Tuple, List[CAEModel]] = {}
    all_pred_mse: Dict[Tuple, List[float]]    = {}

    train_ret   = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)
    val_ret     = splits["val"]["returns"].values.astype(np.float32)
    val_chars   = splits["val"]["chars"].astype(np.float32)

    total_configs = len(K_GRID) * len(LAMBDA_NONLIN_GRID)
    config_num = 0

    for k in K_GRID:
        for lam_nonlin in LAMBDA_NONLIN_GRID:
            config_num += 1
            key = (k, lam_nonlin)
            all_models[key]   = []
            all_pred_mse[key] = []

            print(f"\n  Config {config_num}/{total_configs}: "
                  f"K={k} λ_nonlin={lam_nonlin:.0e}")

            for s_idx, seed in enumerate(seeds):
                print(f"    Seed {seed} ({s_idx+1}/{len(seeds)}) ...")
                model = train_cae_single_seeded(
                    splits, k, 0.0, lam_nonlin,
                    seed=seed, use_linear=False, device=device,
                )
                r_hat_val = model.predict(val_ret, val_chars,
                                          train_returns=train_ret,
                                          train_chars=train_chars)
                pmask    = np.isfinite(val_ret) & np.isfinite(r_hat_val)
                pred_mse = (float(np.mean((val_ret[pmask] - r_hat_val[pmask]) ** 2))
                            if pmask.any() else float("inf"))
                all_models[key].append(model)
                all_pred_mse[key].append(pred_mse)

            mean_mse = float(np.mean(all_pred_mse[key]))
            std_mse  = float(np.std(all_pred_mse[key]))
            print(f"    Mean pred val MSE: {mean_mse:.6f} ± {std_mse:.6f}")

    mean_mse_per_config = {k: float(np.mean(v)) for k, v in all_pred_mse.items()}
    best_config = min(mean_mse_per_config, key=mean_mse_per_config.get)
    print(f"\n  Best CAE-NL config (by mean pred val MSE across {len(seeds)} seeds):")
    print(f"  K={best_config[0]}  λ_nonlin={best_config[1]:.0e}  "
          f"mean_pred_val_mse={mean_mse_per_config[best_config]:.6f}")

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "cae_nl_multiseed_hparam_search.pkl"), "wb") as f:
        pickle.dump({"all_pred_mse": all_pred_mse,
                     "mean_mse_per_config": mean_mse_per_config,
                     "best": best_config, "seeds": seeds}, f)
    with open(os.path.join(run_dir, "cae_nl_hparam_search.pkl"), "wb") as f:
        pickle.dump({"val_losses":   mean_mse_per_config,
                     "pred_val_mse": mean_mse_per_config,
                     "best": best_config}, f)

    return all_models


# train ResCAE-Fixed: W_skip is frozen at its IPCA init; only g and encoder are updated
def train_cae_fixed_single(
    splits: dict,
    k: int,
    lam_nonlin: float,
    device: str = "cpu",
) -> CAEModel:
    """train ResCAE-Fixed (frozen W_skip). params: {splits: dict, k: int, lam_nonlin: float, device: str}. returns CAEModel."""
    return train_cae_single(
        splits, k,
        lam_lin=0.0,
        lam_nonlin=lam_nonlin,
        use_linear=True,
        freeze_linear=True,
        device=device,
    )


def train_all_cae_fixed(
    splits: dict,
    device: str = "cpu",
    run_dir: str = "results",
) -> Dict[Tuple[int, float], CAEModel]:
    """grid search over (K, λ_nonlin) for ResCAE-Fixed; selects best by predictive val MSE.
    saves to results/cae_fixed_hparam_search.pkl. returns dict mapping (k, lam_nonlin) → CAEModel."""
    results:      Dict[Tuple[int, float], CAEModel] = {}
    val_losses:   Dict[Tuple[int, float], float]    = {}
    pred_val_mse: Dict[Tuple[int, float], float]    = {}

    train_ret   = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)

    total      = len(K_GRID) * len(LAMBDA_NONLIN_GRID)
    config_num = 0

    for k in K_GRID:
        for lam_nonlin in LAMBDA_NONLIN_GRID:
            config_num += 1
            print(f"\n  Training ResCAE-Fixed K={k} λ_nonlin={lam_nonlin:.0e} "
                  f"({config_num}/{total}) …")
            model = train_cae_fixed_single(splits, k, lam_nonlin, device)

            val_ret   = splits["val"]["returns"].values.astype(np.float32)
            val_chars = splits["val"]["chars"].astype(np.float32)
            T_val     = val_ret.shape[0]
            t_val     = np.arange(T_val)
            vl = _run_cae_epoch(model, val_ret, val_chars, t_val,
                                optimizer=None, train=False)

            r_hat_val = model.predict(val_ret, val_chars,
                                      train_returns=train_ret, train_chars=train_chars)
            pmask    = np.isfinite(val_ret) & np.isfinite(r_hat_val)
            pred_mse = (float(np.mean((val_ret[pmask] - r_hat_val[pmask]) ** 2))
                        if pmask.any() else float("inf"))

            results[(k, lam_nonlin)]      = model
            val_losses[(k, lam_nonlin)]   = vl
            pred_val_mse[(k, lam_nonlin)] = pred_mse

    best_config = min(pred_val_mse, key=pred_val_mse.get)
    print(f"\n  Best ResCAE-Fixed config: K={best_config[0]}  "
          f"λ_nonlin={best_config[1]:.0e}  "
          f"pred_val_mse={pred_val_mse[best_config]:.6f}")

    # verify W_skip hasn't drifted from its IPCA init (should be zero drift)
    for key, m in results.items():
        if m.gamma_init is not None:
            w_norm = np.linalg.norm(
                m.net.decoder.W_skip.weight.detach().cpu().numpy(), "fro")
            g_norm = np.linalg.norm(m.gamma_init, "fro")
            if abs(w_norm - g_norm) > 1e-5:
                print(f"  WARNING: ResCAE-Fixed {key}: "
                      f"‖W_skip‖={w_norm:.6f} ≠ ‖Γ‖={g_norm:.6f} "
                      f"(diff={abs(w_norm-g_norm):.2e}) — W_skip may have moved.")
            assert m.compute_ipca_drift() == 0.0, \
                f"Drift should be zero for ResCAE-Fixed but got {m.compute_ipca_drift()}"

    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "cae_fixed_hparam_search.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump({"val_losses":   val_losses,
                     "pred_val_mse": pred_val_mse,
                     "best":         best_config}, f)

    return results


def train_all_models(splits: dict,
                     k_list: List[int] = K_GRID,
                     device: str = "cpu",
                     multi_seed: bool = False,
                     seeds: List[int] = SEEDS,
                     run_dir: str = "results",
                     ) -> dict:
    """train all model families (PCA, IPCA, AE, CAE, CAE-NL, ResCAE-Fixed) for each K in k_list.
    if multi_seed=True, CAE and CAE-NL are each trained S times and results are lists.
    returns a dict keyed by model name with per-K or per-config sub-dicts."""
    models: dict = {"pca": {}, "ipca": {}, "ae": {}, "cae": {}, "cae_nl": {},
                    "cae_fixed": {}, "multi_seed": multi_seed}

    print("\n=== Training PCA ===")
    for k in k_list:
        models["pca"][k] = train_pca(splits, k)

    print("\n=== Training IPCA ===")
    for k in k_list:
        models["ipca"][k] = train_ipca(splits, k)

    print("\n=== Training Autoencoder ===")
    for k in k_list:
        print(f"\n  Training AE K={k} …")
        models["ae"][k] = train_autoencoder(splits, k, device)

    if multi_seed:
        models["seeds"] = seeds

        print(f"\n=== Training CAE ({len(seeds)} seeds × grid search) ===")
        models["cae"] = train_all_cae_multiseed(splits, seeds, device, run_dir=run_dir)

        print(f"\n=== Training CAE-NL ablation ({len(seeds)} seeds) ===")
        models["cae_nl"] = train_all_cae_nl_multiseed(splits, seeds, device, run_dir=run_dir)
    else:
        print("\n=== Training CAE (grid search) ===")
        models["cae"] = train_all_cae(splits, device, run_dir=run_dir)

        print("\n=== Training CAE-NL ablation (no linear term) ===")
        models["cae_nl"] = train_all_cae_nonlinear(splits, device, run_dir=run_dir)

    # ResCAE-Fixed is always single-seed; W_skip is deterministically frozen at IPCA solution
    print("\n=== Training ResCAE-Fixed (frozen W_skip) ===")
    models["cae_fixed"] = train_all_cae_fixed(splits, device, run_dir=run_dir)

    return models
