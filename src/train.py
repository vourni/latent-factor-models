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
LAMBDA_ORTH   = 1e-3     # scale/orthogonality penalty; rotations remain unidentified
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
    model.train_losses = train_losses
    model.val_losses = val_losses

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
    n_observations = 0

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

        count = int(mask.sum().item())
        total_loss += loss.item() * count
        n_observations += count

    return total_loss / n_observations if n_observations else float("inf")


def train_cae_single(splits: dict,
                     k: int,
                     lam_lin: float,
                     lam_nonlin: float,
                     use_linear: bool = True,
                     freeze_linear: bool = False,
                     device: str = "cpu",
                     ipca: Optional[IPCAModel] = None,
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
        model.initialize_from_ipca(train_ret, train_chars, ipca=ipca)
    optimizer = Adam(model.net.parameters(), lr=LR)

    T_train = train_ret.shape[0]
    T_val   = val_ret.shape[0]
    t_train = np.flatnonzero((np.isfinite(train_ret) & np.isfinite(train_chars).all(axis=2).T).any(axis=1))
    t_val = np.flatnonzero((np.isfinite(val_ret) & np.isfinite(val_chars).all(axis=2).T).any(axis=1))
    if len(t_train) == 0 or len(t_val) == 0:
        raise ValueError("CAE training and validation each need characteristic-complete observations.")

    # Inputs do not change between epochs. Cache tensors only while this model
    # trains, then release them so a full multi-seed grid stays memory bounded.
    model._training_batches = {
        (id(train_ret), id(train_chars)): model._build_batch(train_ret, train_chars, np.arange(T_train)),
        (id(val_ret), id(val_chars)): model._build_batch(val_ret, val_chars, np.arange(T_val)),
    }
    managed_all = model._build_batch(train_ret, train_chars, t_train)[0]

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

    if best_state is None:
        raise RuntimeError("Training produced no finite validation checkpoint.")
    model.net.load_state_dict(best_state)
    model.train_losses = train_losses
    model.val_losses = val_losses
    del model._training_batches

    # diagnostic: near-zero mean factor signals the model will predict ~0 for all stocks
    if not use_linear:
        mf = np.nanmean(model.get_factors(train_ret, train_chars), axis=0)
        mf_norm = float(np.linalg.norm(mf))
        print(f"    [CAE-NL K={k}] post-training ‖f̄_train‖ = {mf_norm:.4f}"
              + (" ← NEAR ZERO: predict() will return ~0 for all stocks"
                 if mf_norm < 1e-3 else " ← OK"))

    return model


def train_cae_single_seeded(splits, k, lam_lin, lam_nonlin, seed,
                            use_linear=True, freeze_linear=False, device="cpu", ipca=None):
    """Train one configuration with an explicit random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = train_cae_single(
        splits, k, lam_lin, lam_nonlin, use_linear=use_linear,
        freeze_linear=freeze_linear, device=device, ipca=ipca)
    model.seed = seed
    return model


def _train_grid(splits, family, k_list, device, run_dir, ipca_models=None, seeds=None):
    """Shared search/selection protocol for all three conditional model families.

    Multi-seed selection uses mean individual-seed validation predictive MSE;
    evaluation averages predictions. Test data never enter model selection.
    """
    models, scores, reconstruction_scores = {}, {}, {}
    train_ret = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)
    val_ret = splits["val"]["returns"].values.astype(np.float32)
    val_chars = splits["val"]["chars"].astype(np.float32)
    linear_grid = LAMBDA_LIN_GRID if family == "cae" else [0.0]
    run_seeds = list(seeds) if seeds is not None else [None]
    if not run_seeds or len(run_seeds) != len(set(run_seeds)):
        raise ValueError("Seeds must be nonempty and distinct.")

    for k in k_list:
        for lam_lin in linear_grid:
            for lam_nonlin in LAMBDA_NONLIN_GRID:
                key = (k, lam_lin, lam_nonlin) if family == "cae" else (k, lam_nonlin)
                fitted, pred_losses, rec_losses = [], [], []
                for seed in run_seeds:
                    print(f"\n  Training {family}: {key}, seed={seed}")
                    kwargs = dict(use_linear=family != "cae_nl",
                                  freeze_linear=family == "cae_fixed", device=device,
                                  ipca=(ipca_models or {}).get(k))
                    if seed is None:
                        model = train_cae_single(splits, k, lam_lin, lam_nonlin, **kwargs)
                    else:
                        model = train_cae_single_seeded(
                            splits, k, lam_lin, lam_nonlin, seed=seed, **kwargs)
                    pred = model.predict(val_ret, val_chars, train_ret, train_chars)
                    mask = np.isfinite(val_ret) & np.isfinite(pred)
                    mse = float(np.mean((val_ret[mask] - pred[mask]) ** 2)) if mask.any() else np.inf
                    if not np.isfinite(mse):
                        raise RuntimeError(f"Nonfinite validation MSE for {family} {key}.")
                    fitted.append(model)
                    pred_losses.append(mse)
                    rec_losses.append(min(model.val_losses))
                    if family == "cae_fixed":
                        np.testing.assert_array_equal(
                            model.net.decoder.W_skip.weight.detach().cpu().numpy(), model.gamma_init)
                models[key] = fitted if seeds is not None else fitted[0]
                scores[key] = pred_losses
                reconstruction_scores[key] = float(np.mean(rec_losses))

    mean_scores = {key: float(np.mean(value)) for key, value in scores.items()}
    best = min(mean_scores, key=mean_scores.get)
    print(f"\n  Best {family} by validation predictive MSE: {best} ({mean_scores[best]:.6f})")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, f"{family}_hparam_search.pkl"), "wb") as f:
        pickle.dump({"val_losses": reconstruction_scores, "pred_val_mse": mean_scores,
                     "best": best}, f)
    if seeds is not None:
        with open(os.path.join(run_dir, f"{family}_multiseed_hparam_search.pkl"), "wb") as f:
            pickle.dump({"all_pred_mse": scores, "mean_mse_per_config": mean_scores,
                         "best": best, "seeds": list(seeds)}, f)
    return models


def train_all_cae(splits, device="cpu", run_dir="results", k_list=None, ipca_models=None):
    return _train_grid(splits, "cae", K_GRID if k_list is None else k_list,
                       device, run_dir, ipca_models)


def train_all_cae_nonlinear(splits, device="cpu", run_dir="results", k_list=None):
    return _train_grid(splits, "cae_nl", K_GRID if k_list is None else k_list, device, run_dir)


def train_all_cae_multiseed(splits, seeds=SEEDS, device="cpu", run_dir="results",
                           k_list=None, ipca_models=None):
    return _train_grid(splits, "cae", K_GRID if k_list is None else k_list,
                       device, run_dir, ipca_models, seeds)


def train_all_cae_nl_multiseed(splits, seeds=SEEDS, device="cpu", run_dir="results", k_list=None):
    return _train_grid(splits, "cae_nl", K_GRID if k_list is None else k_list,
                       device, run_dir, seeds=seeds)


def train_cae_fixed_single(splits, k, lam_nonlin, device="cpu", ipca=None):
    return train_cae_single(splits, k, 0.0, lam_nonlin, freeze_linear=True, device=device, ipca=ipca)


def train_all_cae_fixed(splits, device="cpu", run_dir="results", k_list=None,
                        ipca_models=None, seeds=None):
    return _train_grid(splits, "cae_fixed", K_GRID if k_list is None else k_list,
                       device, run_dir, ipca_models, seeds)


def train_all_models(splits, k_list=None, device="cpu", multi_seed=False,
                      seeds=SEEDS, run_dir="results"):
    """Train all six families on the requested K grid; ensemble all CAE variants."""
    k_list = K_GRID if k_list is None else list(k_list)
    if not k_list or any(k <= 0 for k in k_list):
        raise ValueError("Provide at least one positive factor dimension.")
    models = {"pca": {}, "ipca": {}, "ae": {}, "multi_seed": multi_seed}
    for k in k_list:
        models["pca"][k] = train_pca(splits, k)
        models["ipca"][k] = train_ipca(splits, k)
        models["ae"][k] = train_autoencoder(splits, k, device)
    if multi_seed:
        models["seeds"] = list(seeds)
    for family in ("cae", "cae_nl", "cae_fixed"):
        models[family] = _train_grid(splits, family, k_list, device, run_dir,
                                     ipca_models=models["ipca"],
                                     seeds=seeds if multi_seed else None)
    return models
