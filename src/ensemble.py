"""multi-seed ensemble utilities for the residual CAE study."""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

from src.models import CAEModel


class EnsembledModel:
    """wraps S CAEModels and returns averaged predictions."""

    def __init__(self, models: List[CAEModel]) -> None:
        assert len(models) > 0, "Need at least one model"
        self.models     = models
        self.n_seeds    = len(models)
        # surface attributes evaluate.py may inspect on a CAEModel
        self.gamma_init = getattr(models[0], "gamma_init", None)
        self.lam_lin    = getattr(models[0], "lam_lin",    0.0)
        self.lam_nonlin = getattr(models[0], "lam_nonlin", 0.0)
        self.n_factors  = getattr(models[0], "n_factors",  None)

    def predict(self,
                returns: np.ndarray,
                chars: np.ndarray,
                train_returns: np.ndarray = None,
                train_chars: np.ndarray = None,
                **kwargs) -> np.ndarray:
        """nanmean of predicted returns across all S models. params: {returns: np.ndarray, chars: np.ndarray, train_returns: np.ndarray, train_chars: np.ndarray}. returns np.ndarray."""
        preds = [m.predict(returns, chars,
                           train_returns=train_returns,
                           train_chars=train_chars, **kwargs)
                 for m in self.models]
        return np.nanmean(np.stack(preds, axis=0), axis=0)

    def reconstruct(self,
                    returns: np.ndarray,
                    chars: np.ndarray = None,
                    **kwargs) -> np.ndarray:
        """nanmean of reconstructed returns across all S models. params: {returns: np.ndarray, chars: np.ndarray}. returns np.ndarray."""
        recs = [m.reconstruct(returns, chars, **kwargs) for m in self.models]
        return np.nanmean(np.stack(recs, axis=0), axis=0)

    def get_nonlinear_contribution(self,
                                   returns: np.ndarray,
                                   chars: np.ndarray) -> Tuple[float, float]:
        """mean (var_lin, var_nonlin) across models. params: {returns: np.ndarray, chars: np.ndarray}. returns Tuple[float, float]."""
        results = [m.get_nonlinear_contribution(returns, chars)
                   for m in self.models]
        return (float(np.mean([r[0] for r in results])),
                float(np.mean([r[1] for r in results])))

    def compute_ipca_drift(self) -> Optional[float]:
        """mean W_skip drift from IPCA initialisation across seeds. returns Optional[float]."""
        drifts = []
        for m in self.models:
            if not hasattr(m, "gamma_init") or m.gamma_init is None:
                continue
            w     = m.net.decoder.W_skip.weight.detach().cpu().numpy()
            gamma = m.gamma_init
            denom = np.linalg.norm(gamma, "fro")
            if denom < 1e-10:
                continue
            drifts.append(float(np.linalg.norm(w - gamma, "fro") / denom))
        return float(np.mean(drifts)) if drifts else None

    # std of drift for diff seeds
    def drift_std(self) -> Optional[float]:
        drifts = []
        for m in self.models:
            if not hasattr(m, "gamma_init") or m.gamma_init is None:
                continue
            w     = m.net.decoder.W_skip.weight.detach().cpu().numpy()
            gamma = m.gamma_init
            denom = np.linalg.norm(gamma, "fro")
            if denom < 1e-10:
                continue
            drifts.append(float(np.linalg.norm(w - gamma, "fro") / denom))
        return float(np.std(drifts)) if len(drifts) > 1 else None


def build_ensembles(models: dict) -> dict:
    """converts a multi-seed models dict into an ensembled models dict. PCA, IPCA, and AE pass through unchanged. params: {models: dict}. returns dict."""
    ensembled = {k: v for k, v in models.items()
                 if k not in ("cae", "cae_nl", "multi_seed", "seeds")}

    ensembled["cae"] = {}
    for key, model_list in models["cae"].items():
        if isinstance(model_list, list):
            ensembled["cae"][key] = EnsembledModel(model_list)
        else:
            ensembled["cae"][key] = model_list  # already a single model

    ensembled["cae_nl"] = {}
    for key, model_list in models.get("cae_nl", {}).items():
        if isinstance(model_list, list):
            ensembled["cae_nl"][key] = EnsembledModel(model_list)
        else:
            ensembled["cae_nl"][key] = model_list

    ensembled["multi_seed"] = False
    return ensembled


def compute_seed_stability(splits: dict, models: dict,
                           results_dir: str = "results") -> pd.DataFrame:
    """computes per-seed pred R², Sharpe, φ, and drift for each CAE config. params: {splits: dict, models: dict, results_dir: str}. returns pd.DataFrame."""
    from src.evaluate import (predictive_r2, factor_sharpe,
                               nonlinear_contribution)

    test_ret    = splits["test"]["returns"].values.astype(np.float32)
    test_chars  = splits["test"]["chars"].astype(np.float32)
    train_ret   = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)

    seeds = models.get("seeds", [])
    rows  = []

    for key, model_list in models.get("cae", {}).items():
        if not isinstance(model_list, list):
            continue   # single-seed run — no stability to report
        k, lam_lin, lam_nonlin = key

        for s_idx, model in enumerate(model_list):
            seed = seeds[s_idx] if s_idx < len(seeds) else s_idx

            r_hat = model.predict(test_ret, test_chars,
                                  train_returns=train_ret,
                                  train_chars=train_chars).astype(np.float32)

            # drift from IPCA init
            drift = None
            if hasattr(model, "gamma_init") and model.gamma_init is not None:
                w     = model.net.decoder.W_skip.weight.detach().cpu().numpy()
                gamma = model.gamma_init
                denom = np.linalg.norm(gamma, "fro")
                if denom >= 1e-10:
                    drift = float(np.linalg.norm(w - gamma, "fro") / denom)

            _, _, phi = nonlinear_contribution(model, test_ret, test_chars)

            rows.append({
                "config_key": str(key),
                "K":          k,
                "lam_lin":    lam_lin,
                "lam_nonlin": lam_nonlin,
                "seed":       seed,
                "pred_r2":    predictive_r2(test_ret, r_hat),
                "sharpe":     factor_sharpe(test_ret, r_hat),
                "phi":        phi,
                "drift":      drift,
            })

    # ResCAE-Fixed is single-seed; drift=0 by construction
    for key, model in models.get("cae_fixed", {}).items():
        if isinstance(model, list):
            continue  # skip unexpected multi-seed fixed models
        k, lam_nonlin = key
        r_hat = model.predict(test_ret, test_chars,
                              train_returns=train_ret,
                              train_chars=train_chars).astype(np.float32)
        _, _, phi = nonlinear_contribution(model, test_ret, test_chars)
        rows.append({
            "config_key": f"fixed_{key}",
            "K":          k,
            "lam_lin":    0.0,
            "lam_nonlin": lam_nonlin,
            "seed":       "fixed",
            "pred_r2":    predictive_r2(test_ret, r_hat),
            "sharpe":     factor_sharpe(test_ret, r_hat),
            "phi":        phi,
            "drift":      0.0,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("  compute_seed_stability: no multi-seed CAE models found.")
        return df

    print("\n--- Seed Stability Summary (mean ± std across seeds) ---")
    for k in sorted(df["K"].unique()):
        print(f"\n  K={k}")
        for metric in ["pred_r2", "sharpe", "phi", "drift"]:
            sub = df[df["K"] == k][metric].dropna()
            if not sub.empty:
                print(f"    {metric:8s}: {sub.mean():.4f} ± {sub.std():.4f}"
                      f"  (n={len(sub)})")

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "seed_stability.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Seed stability → {csv_path}")

    return df


def validate_ensemble_improvement(
    splits: dict,
    models: dict,
    ensemble_models: dict,
    results_dir: str = "results",
) -> None:
    """warns if ensemble pred R² < best single-seed pred R² for the best config. params: {splits: dict, models: dict, ensemble_models: dict, results_dir: str}. returns None."""
    import pickle
    from src.evaluate import predictive_r2

    pkl_path = os.path.join(results_dir, "cae_multiseed_hparam_search.pkl")
    if not os.path.exists(pkl_path):
        return

    with open(pkl_path, "rb") as f:
        hd = pickle.load(f)
    best_key = hd.get("best")
    if best_key is None or best_key not in models["cae"]:
        return

    test_ret    = splits["test"]["returns"].values.astype(np.float32)
    test_chars  = splits["test"]["chars"].astype(np.float32)
    train_ret   = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)

    def _r2(m):
        r_hat = m.predict(test_ret, test_chars,
                          train_returns=train_ret,
                          train_chars=train_chars).astype(np.float32)
        return predictive_r2(test_ret, r_hat)

    ensemble_r2   = _r2(ensemble_models["cae"][best_key])
    single_seed_r2 = max(_r2(m) for m in models["cae"][best_key])

    if ensemble_r2 < single_seed_r2:
        print(f"WARNING: Ensemble did not improve over single seed for "
              f"config {best_key}.")
        print(f"  Ensemble Pred R²:    {ensemble_r2:.4f}")
        print(f"  Single-seed best R²: {single_seed_r2:.4f}")
        print("  This may indicate instability in training. "
              "Check seed_stability.csv.")
    else:
        print(f"  Ensemble check passed: R²={ensemble_r2:.4f} ≥ "
              f"single-seed best {single_seed_r2:.4f}")
