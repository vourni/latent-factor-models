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
        self.use_linear = models[0].use_linear
        self.freeze_linear = models[0].freeze_linear
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
    ensembled = dict(models)
    for family in ("cae", "cae_nl", "cae_fixed"):
        ensembled[family] = {
            key: EnsembledModel(value) if isinstance(value, list) else value
            for key, value in models.get(family, {}).items()
        }
    ensembled["multi_seed"] = False
    return ensembled


def compute_seed_stability(splits, models, results_dir="results"):
    """Per-seed test diagnostics with validation-selected configurations marked."""
    from src.evaluate import (predictive_r2, factor_sharpe, nonlinear_contribution,
                              compute_ipca_drift, _best_key_per_k, evaluation_splits,
                              predictions_collapsed)
    splits = evaluation_splits(splits)
    test_ret = splits["test"]["returns"].values.astype(np.float32)
    test_chars = splits["test"]["chars"].astype(np.float32)
    train_ret = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)
    rows = []
    for family in ("cae", "cae_fixed", "cae_nl"):
        configs = models.get(family, {})
        best = _best_key_per_k(configs, os.path.join(results_dir, f"{family}_hparam_search.pkl"))
        for key, fitted in configs.items():
            if not isinstance(fitted, list):
                continue
            for model in fitted:
                pred = model.predict(test_ret, test_chars, train_ret, train_chars)
                _, _, phi = nonlinear_contribution(model, test_ret, test_chars)
                if predictions_collapsed(test_ret, pred):
                    phi = np.nan
                rows.append({"Model": model.model_name, "config_key": f"{family}:{key}",
                             "K": key[0], "lam_lin": key[1] if family == "cae" else 0.0,
                             "lam_nonlin": key[-1], "seed": model.seed,
                             "selected_by_validation": key == best.get(key[0]),
                             "pred_r2": predictive_r2(test_ret, pred),
                             "sharpe": factor_sharpe(test_ret, pred), "phi": phi,
                             "drift": compute_ipca_drift(model)})
    df = pd.DataFrame(rows)
    if not df.empty:
        os.makedirs(results_dir, exist_ok=True)
        df.to_csv(os.path.join(results_dir, "seed_stability.csv"), index=False)
    return df


def validate_ensemble_improvement(splits, models, ensemble_models, results_dir="results"):
    """Descriptive diagnostic: averaging is not guaranteed to beat the best seed."""
    from src.evaluate import predictive_r2, evaluation_splits, _best_key_per_k
    splits = evaluation_splits(splits)
    test_ret = splits["test"]["returns"].values.astype(np.float32)
    test_chars = splits["test"]["chars"].astype(np.float32)
    train_ret = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)
    best = _best_key_per_k(models["cae"], os.path.join(results_dir, "cae_hparam_search.pkl"))
    def score(model):
        return predictive_r2(test_ret, model.predict(test_ret, test_chars, train_ret, train_chars))
    for k, key in sorted(best.items()):
        values = [score(model) for model in models["cae"][key]]
        print(f"  K={k}: ensemble R²={score(ensemble_models['cae'][key]):.5f}; "
              f"individual-seed mean={np.mean(values):.5f}, best={max(values):.5f}")
    print("  Best-seed test scores are descriptive only; seeds are not selected on the test set.")
