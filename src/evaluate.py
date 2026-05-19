"""
evaluation metrics for the ResCAE study. tests whether the nonlinear residual g(z)
adds predictability beyond IPCA across six metrics: total R², predictive R², factor
Sharpe, nonlinear contribution φ, IPCA drift, and OOS pricing error (CSPE).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple

from src.models import PCAModel, AutoencoderModel, CAEModel

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass


def _safe_mask(*arrays: np.ndarray) -> np.ndarray:
    """boolean mask: True where ALL arrays are finite."""
    mask = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def total_r2(r_true: np.ndarray, r_hat: np.ndarray) -> float:
    """proportion of return variance explained. params: {r_true: (T,N), r_hat: (T,N)}. returns float.

    denominator is Σ r² not Σ(r-r̄)², following Gu et al. 2020 convention.
    """
    mask = _safe_mask(r_true, r_hat)
    ss_res = np.sum((r_true[mask] - r_hat[mask]) ** 2)
    ss_tot = np.sum(r_true[mask] ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def predictive_r2(r_true_test: np.ndarray,
                  r_hat_test: np.ndarray) -> float:
    """OOS predictive R². params: {r_true_test: (T,N), r_hat_test: (T,N)}. returns float.

    caller must ensure r_hat_test uses only t-1 information; this function just applies the R² formula.
    """
    return total_r2(r_true_test, r_hat_test)


def factor_sharpe(r_true_test: np.ndarray,
                  r_hat_test: np.ndarray,
                  annualize: int = 12) -> float:
    """annualized Sharpe of a long-short decile portfolio. params: {r_true_test: (T,N), r_hat_test: (T,N), annualize: int}. returns float.

    each month: long top-decile predicted, short bottom-decile predicted, equal-weight within each leg.
    """
    T, N = r_true_test.shape

    # return NaN instead of 0 if predictions are globally flat (collapsed model)
    pred_cs_std  = np.nanstd(r_hat_test, axis=1)
    mean_cs_std  = float(np.nanmean(pred_cs_std))
    if mean_cs_std < 1e-6:
        return np.nan

    port_returns = np.full(T, np.nan)

    for t in range(T):
        r_t     = r_true_test[t]
        r_hat_t = r_hat_test[t]

        valid = _safe_mask(r_t, r_hat_t)
        if valid.sum() < 20:
            continue

        pred_valid = r_hat_t[valid]
        ret_valid  = r_t[valid]

        # skip months where all predictions are essentially equal
        if pred_valid.std() < 1e-6:
            continue

        q10 = np.percentile(pred_valid, 10)
        q90 = np.percentile(pred_valid, 90)

        long_mask  = pred_valid >= q90
        short_mask = pred_valid <= q10

        if long_mask.sum() == 0 or short_mask.sum() == 0:
            continue

        long_ret  = ret_valid[long_mask].mean()
        short_ret = ret_valid[short_mask].mean()
        port_returns[t] = long_ret - short_ret

    valid_returns = port_returns[~np.isnan(port_returns)]
    if len(valid_returns) < 2:
        return np.nan

    sr = valid_returns.mean() / (valid_returns.std(ddof=1) + 1e-10)
    return float(sr * np.sqrt(annualize))


def nonlinear_contribution(model: CAEModel,
                           returns: np.ndarray,
                           chars: np.ndarray) -> Tuple[float, float, float]:
    """decompose ResCAE loading variance into linear and nonlinear parts. params: {model: CAEModel, returns: (T,N), chars: (N,T,P)}. returns (var_lin, var_nonlin, frac_nonlin).

    φ = var_nonlin / (var_lin + var_nonlin); high φ means g(z) captures variation the linear term misses.
    """
    var_lin, var_nonlin = model.get_nonlinear_contribution(returns, chars)

    # both vars == 0 signals a collapsed model; φ is undefined in that case
    if var_lin < 1e-10 and var_nonlin < 1e-10:
        return 0.0, 0.0, np.nan

    total = var_lin + var_nonlin + 1e-10
    frac_nonlin = var_nonlin / total
    return var_lin, var_nonlin, float(frac_nonlin)


def compute_ipca_drift(model) -> Optional[float]:
    """relative Frobenius drift of W_skip from IPCA warm-start: ‖W_skip − Γ_init‖_F / ‖Γ_init‖_F. returns None if model was not IPCA-warm-started."""
    # delegate to instance method when available; handles freeze_linear and ensemble averaging
    fn = getattr(model, "compute_ipca_drift", None)
    if callable(fn):
        drift = fn()
    elif not hasattr(model, "gamma_init") or model.gamma_init is None:
        return None
    else:
        w     = model.net.decoder.W_skip.weight.detach().cpu().numpy()
        gamma = model.gamma_init
        denom = np.linalg.norm(gamma, "fro")
        if denom < 1e-10:
            return None
        drift = float(np.linalg.norm(w - gamma, "fro") / denom)

    if drift is not None and drift > 1.0:
        import warnings
        warnings.warn(
            f"W_skip drift = {drift:.4f} > 1.0. The IPCA initialization was "
            f"effectively discarded during training. The phi decomposition is "
            f"not interpretable for this configuration.",
            UserWarning, stacklevel=2,
        )
    return drift


def oos_pricing_error(r_true: np.ndarray, r_hat: np.ndarray) -> float:
    """cross-sectional RMSE of per-stock OOS alphas (CSPE). params: {r_true: (T,N), r_hat: (T,N)}. returns float.

    alpha_i = mean(r_i) - mean(r_hat_i); CSPE = sqrt(mean(alpha_i^2)). lower is better.
    """
    N = r_true.shape[1]
    alphas = np.full(N, np.nan)

    for i in range(N):
        valid = np.isfinite(r_true[:, i]) & np.isfinite(r_hat[:, i])
        if valid.sum() < 2:
            continue
        alphas[i] = r_true[valid, i].mean() - r_hat[valid, i].mean()

    finite = alphas[np.isfinite(alphas)]
    if len(finite) == 0:
        return np.nan
    return float(np.sqrt((finite ** 2).mean()))


def _best_key_per_k(model_dict: dict, pkl_path: str) -> dict:
    """return {k: best_key} selecting by lowest pred_val_mse, falling back to val_losses if absent."""
    import os, pickle
    scores: dict = {}
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as _f:
            data = pickle.load(_f)
        scores = data.get("pred_val_mse") or data.get("val_losses", {})

    best: dict = {}
    for key in model_dict:
        k = key[0]
        score = scores.get(key, float("inf"))
        if k not in best or score < scores.get(best[k], float("inf")):
            best[k] = key
    return best


def build_summary_table(splits: dict,
                        models: dict,
                        results_dir: str = "results",
                        ) -> pd.DataFrame:
    """compute all metrics for the best-λ config per (model family, K) on the test set. params: {splits: dict, models: dict, results_dir: str}. returns pd.DataFrame.

    model order: PCA → IPCA → ResCAE → CAE-NL. IPCA rows are NaN placeholders if models["ipca"] is absent.
    """
    import os
    test_ret    = splits["test"]["returns"].values.astype(np.float32)
    test_chars  = splits["test"]["chars"].astype(np.float32)
    train_ret   = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)

    rows = []

    # PCA
    for k, model in sorted(models["pca"].items()):
        r_hat_in  = model.reconstruct(test_ret).astype(np.float32)
        r_hat_oos = model.predict(test_ret, train_returns=train_ret).astype(np.float32)
        rows.append({"Model": "PCA", "K": k, "λ_lin": "-", "λ_nonlin": "-",
                     "Total_R2": total_r2(test_ret, r_hat_in),
                     "Pred_R2":  predictive_r2(test_ret, r_hat_oos),
                     "Sharpe":   factor_sharpe(test_ret, r_hat_oos),
                     "CSPE":     oos_pricing_error(test_ret, r_hat_oos),
                     "NL_frac":  "-", "drift_rel": "-"})

    # IPCA
    if "ipca" in models:
        for k, model in sorted(models["ipca"].items()):
            r_hat_in  = model.reconstruct(test_ret, test_chars).astype(np.float32)
            r_hat_oos = model.predict(
                test_ret, test_chars,
                train_returns=train_ret, train_chars=train_chars,
            ).astype(np.float32)
            rows.append({"Model": "IPCA", "K": k, "λ_lin": "-", "λ_nonlin": "-",
                         "Total_R2": total_r2(test_ret, r_hat_in),
                         "Pred_R2":  predictive_r2(test_ret, r_hat_oos),
                         "Sharpe":   factor_sharpe(test_ret, r_hat_oos),
                         "CSPE":     oos_pricing_error(test_ret, r_hat_oos),
                         "NL_frac":  "-", "drift_rel": "-"})
    else:
        for k in sorted(models["pca"].keys()):
            rows.append({"Model": "IPCA", "K": k, "λ_lin": "-", "λ_nonlin": "-",
                         "Total_R2": np.nan, "Pred_R2": np.nan,
                         "Sharpe":   np.nan, "CSPE":    np.nan,
                         "NL_frac":  "-",    "drift_rel": "-"})

    # ResCAE-Fixed — best λ_nonlin per K
    if "cae_fixed" in models and models["cae_fixed"]:
        fixed_best = _best_key_per_k(
            models["cae_fixed"],
            os.path.join(results_dir, "cae_fixed_hparam_search.pkl"),
        )
        for k, key in sorted(fixed_best.items()):
            _, lam_nonlin = key
            model = models["cae_fixed"][key]
            r_hat_in  = model.reconstruct(test_ret, test_chars).astype(np.float32)
            r_hat_oos = model.predict(
                test_ret, test_chars,
                train_returns=train_ret, train_chars=train_chars,
            ).astype(np.float32)
            _, _, nl_frac = nonlinear_contribution(model, test_ret, test_chars)
            rows.append({"Model": "ResCAE-Fixed", "K": k,
                         "λ_lin": "frozen", "λ_nonlin": f"{lam_nonlin:.0e}",
                         "Total_R2": total_r2(test_ret, r_hat_in),
                         "Pred_R2":  predictive_r2(test_ret, r_hat_oos),
                         "Sharpe":   factor_sharpe(test_ret, r_hat_oos),
                         "CSPE":     oos_pricing_error(test_ret, r_hat_oos),
                         "NL_frac":  f"{nl_frac:.3f}",
                         "drift_rel": "0.0000"})

    # ResCAE — best λ per K
    cae_best = _best_key_per_k(
        models["cae"],
        os.path.join(results_dir, "cae_hparam_search.pkl"),
    )
    for k, key in sorted(cae_best.items()):
        _, lam_lin, lam_nonlin = key
        model = models["cae"][key]
        r_hat_in  = model.reconstruct(test_ret, test_chars).astype(np.float32)
        r_hat_oos = model.predict(
            test_ret, test_chars,
            train_returns=train_ret, train_chars=train_chars,
        ).astype(np.float32)
        _, _, nl_frac = nonlinear_contribution(model, test_ret, test_chars)
        drift = compute_ipca_drift(model)
        drift_capped = (drift is not None and drift > 1.0)
        drift_display = (f"{min(drift, 1.0):.4f}{'*' if drift_capped else ''}"
                         if drift is not None else None)
        rows.append({"Model": "ResCAE", "K": k,
                     "λ_lin": f"{lam_lin:.0e}", "λ_nonlin": f"{lam_nonlin:.0e}",
                     "Total_R2": total_r2(test_ret, r_hat_in),
                     "Pred_R2":  predictive_r2(test_ret, r_hat_oos),
                     "Sharpe":   factor_sharpe(test_ret, r_hat_oos),
                     "CSPE":     oos_pricing_error(test_ret, r_hat_oos),
                     "NL_frac":  f"{nl_frac:.3f}" if np.isfinite(nl_frac) else "N/A",
                     "drift_rel":   drift_display,
                     "drift_capped": drift_capped})

    # CAE-NL — best λ_nonlin per K (robustness)
    cae_nl_best = _best_key_per_k(
        models.get("cae_nl", {}),
        os.path.join(results_dir, "cae_nl_hparam_search.pkl"),
    )
    for k, key in sorted(cae_nl_best.items()):
        _, lam_nonlin = key
        model = models["cae_nl"][key]
        r_hat_in  = model.reconstruct(test_ret, test_chars).astype(np.float32)
        r_hat_oos = model.predict(
            test_ret, test_chars,
            train_returns=train_ret, train_chars=train_chars,
        ).astype(np.float32)
        _, _, nl_frac = nonlinear_contribution(model, test_ret, test_chars)
        rows.append({"Model": "CAE-NL (robustness)", "K": k,
                     "λ_lin": "-", "λ_nonlin": f"{lam_nonlin:.0e}",
                     "Total_R2": total_r2(test_ret, r_hat_in),
                     "Pred_R2":  predictive_r2(test_ret, r_hat_oos),
                     "Sharpe":   factor_sharpe(test_ret, r_hat_oos),
                     "CSPE":     oos_pricing_error(test_ret, r_hat_oos),
                     "NL_frac":  f"{nl_frac:.3f}",
                     "drift_rel": "-"})

    df = pd.DataFrame(rows)

    # collapsed models have NaN Sharpe from flat predictions
    sharpe_numeric = pd.to_numeric(df["Sharpe"], errors="coerce")
    df["Collapsed"] = sharpe_numeric.isna() & df["Model"].isin(
        ["CAE-NL (robustness)", "ResCAE", "ResCAE-Fixed"])

    # collapsed Sharpe → "COLLAPSED", undefined phi → "N/A"
    for col in ["Total_R2", "Pred_R2", "CSPE"]:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    df["Sharpe"] = df.apply(
        lambda row: "COLLAPSED" if row["Collapsed"] and pd.isna(row["Sharpe"])
                    else (f"{row['Sharpe']:.4f}" if isinstance(row["Sharpe"], float) else row["Sharpe"]),
        axis=1)
    df["NL_frac"] = df["NL_frac"].apply(
        lambda x: "N/A" if (isinstance(x, float) and np.isnan(x)) else x)

    model_order = {"PCA": 0, "IPCA": 1, "ResCAE-Fixed": 2,
                   "ResCAE": 3, "CAE-NL (robustness)": 4}
    df["_order"] = df["Model"].map(model_order)
    df = df.sort_values(["K", "_order"]).drop(columns="_order")
    return df


def _save_fig(fig: plt.Figure, base_path: str, dpi: int = 200) -> None:
    """save figure as PNG and PDF."""
    import os
    png_path = base_path if base_path.endswith(".png") else base_path + ".png"
    pdf_path = png_path.replace(".png", ".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {png_path}")


def plot_loss_curves(results_dir: str = "results",
                     figures_dir: str = "paper/figures") -> None:
    """plot train/val loss curves from saved checkpoints."""
    import os, glob, torch

    os.makedirs(figures_dir, exist_ok=True)
    ckpt_files = glob.glob(os.path.join(results_dir, "*.pt"))
    if not ckpt_files:
        print("  No checkpoint files found; skipping loss curve plots.")
        return

    fig, axes = plt.subplots(len(ckpt_files), 1,
                              figsize=(8, 3 * len(ckpt_files)),
                              squeeze=False)

    for ax, ckpt_path in zip(axes[:, 0], sorted(ckpt_files)):
        ckpt   = torch.load(ckpt_path, map_location="cpu")
        label  = os.path.basename(ckpt_path).replace(".pt", "")
        ax.plot(ckpt.get("train_losses", []), label="Train")
        ax.plot(ckpt.get("val_losses",   []), label="Val")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("MSE Loss", fontsize=9)
        ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(figures_dir, "loss_curves.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Loss curves → {out}")


def plot_summary_heatmap(summary_df: pd.DataFrame,
                          figures_dir: str = "paper/figures") -> None:
    """heatmap of Total R² and Sharpe for all model × K combinations."""
    import os
    os.makedirs(figures_dir, exist_ok=True)

    df = summary_df.copy()
    # track collapsed cells before coercing to numeric
    collapsed_mask = df.get("Collapsed", pd.Series(False, index=df.index))
    for col in ["Total_R2", "Pred_R2", "Sharpe"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    pivot_r2 = df.pivot_table(index="Model", columns="K",
                               values="Total_R2", aggfunc="mean")
    pivot_sr = df.pivot_table(index="Model", columns="K",
                               values="Sharpe", aggfunc="mean")
    # boolean pivot for collapsed cells (all-NaN Sharpe per group)
    if collapsed_mask.any():
        pivot_col = df.pivot_table(index="Model", columns="K",
                                   values="Sharpe", aggfunc=lambda x: x.isna().all())
    else:
        pivot_col = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(pivot_r2, ax=axes[0], annot=True, fmt=".4f",
                cmap="RdYlGn", linewidths=0.5)
    axes[0].set_title("Total R²  (test set)", fontsize=13)
    sns.heatmap(pivot_sr, ax=axes[1], annot=True, fmt=".3f",
                cmap="RdYlGn", linewidths=0.5)
    axes[1].set_title("Annualized Sharpe  (test set)", fontsize=13)

    # hatch collapsed cells in the Sharpe heatmap
    if pivot_col is not None:
        models_idx = list(pivot_sr.index)
        k_cols     = list(pivot_sr.columns)
        for r_i, model in enumerate(models_idx):
            for c_i, k in enumerate(k_cols):
                if model in pivot_col.index and k in pivot_col.columns:
                    if pivot_col.loc[model, k]:
                        axes[1].add_patch(
                            plt.Rectangle((c_i, r_i), 1, 1,
                                          fill=True, facecolor="lightgray",
                                          hatch="//", edgecolor="dimgray", lw=0))
        from matplotlib.patches import Patch as _Patch
        axes[1].legend(
            handles=[_Patch(facecolor="lightgray", hatch="//",
                            edgecolor="dimgray", label="Model collapsed (flat predictions)")],
            fontsize=8, loc="upper right")

    fig.tight_layout()
    _save_fig(fig, os.path.join(figures_dir, "summary_heatmap.png"))


def plot_factor_portfolios(splits: dict,
                            models: dict,
                            figures_dir: str = "paper/figures",
                            results_dir: str = "results") -> None:
    """cumulative long-short portfolio returns for the test period (figure 1). one line per model family; CAE-NL dashed."""
    import os, pickle
    os.makedirs(figures_dir, exist_ok=True)

    test_ret    = splits["test"]["returns"].values.astype(np.float32)
    test_chars  = splits["test"]["chars"].astype(np.float32)
    train_ret   = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)
    test_dates  = splits["test"]["returns"].index

    hparam_path = os.path.join(results_dir, "cae_hparam_search.pkl")
    if os.path.exists(hparam_path):
        with open(hparam_path, "rb") as f:
            best_cae_key = pickle.load(f)["best"]
    elif models["cae"]:
        best_cae_key = next(iter(models["cae"]))
    else:
        best_cae_key = None

    nl_hparam_path = os.path.join(results_dir, "cae_nl_hparam_search.pkl")
    if os.path.exists(nl_hparam_path):
        with open(nl_hparam_path, "rb") as f:
            best_nl_key = pickle.load(f)["best"]
    elif models.get("cae_nl"):
        best_nl_key = next(iter(models["cae_nl"]))
    else:
        best_nl_key = None

    fixed_hparam_path = os.path.join(results_dir, "cae_fixed_hparam_search.pkl")
    if os.path.exists(fixed_hparam_path):
        with open(fixed_hparam_path, "rb") as f:
            best_fixed_key = pickle.load(f).get("best")
    elif models.get("cae_fixed"):
        best_fixed_key = next(iter(models["cae_fixed"]))
    else:
        best_fixed_key = None

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {
        "PCA":          "steelblue",
        "IPCA":         "darkorange",
        "ResCAE-Fixed": "mediumpurple",
        "ResCAE":       "seagreen",
        "CAE-NL":       "dimgray",
    }

    def _port_ts(r_true, r_hat, label, color, lw=1.8, ls="-"):
        T = r_true.shape[0]
        port = []
        for t in range(T):
            r_t  = r_true[t]
            rh   = r_hat[t]
            valid = _safe_mask(r_t, rh)
            if valid.sum() < 20:
                port.append(np.nan)
                continue
            pv = rh[valid]; rv = r_t[valid]
            q10 = np.percentile(pv, 10); q90 = np.percentile(pv, 90)
            lng = rv[pv >= q90].mean() if (pv >= q90).any() else np.nan
            sht = rv[pv <= q10].mean() if (pv <= q10).any() else np.nan
            port.append(lng - sht)
        cum = np.nancumsum(np.array(port, dtype=float))
        ax.plot(test_dates[:len(cum)], cum, label=label,
                color=color, lw=lw, linestyle=ls)

    # use middle K as a representative
    k_vals = sorted(models["pca"].keys())
    k_rep  = k_vals[len(k_vals) // 2]

    if k_rep in models["pca"]:
        _port_ts(test_ret,
                 models["pca"][k_rep].predict(test_ret, train_returns=train_ret),
                 f"PCA K={k_rep}", colors["PCA"])

    if "ipca" in models and k_rep in models["ipca"]:
        _port_ts(test_ret,
                 models["ipca"][k_rep].predict(
                     test_ret, test_chars,
                     train_returns=train_ret, train_chars=train_chars),
                 f"IPCA K={k_rep}", colors["IPCA"])

    if best_fixed_key is not None and best_fixed_key in models.get("cae_fixed", {}):
        k_f, lam_nl_f = best_fixed_key
        _port_ts(test_ret,
                 models["cae_fixed"][best_fixed_key].predict(
                     test_ret, test_chars,
                     train_returns=train_ret, train_chars=train_chars),
                 f"ResCAE-Fixed K={k_f} λNL={lam_nl_f:.0e}",
                 colors["ResCAE-Fixed"], lw=1.5)

    if best_cae_key is not None and best_cae_key in models["cae"]:
        k, lam_lin, lam_nonlin = best_cae_key
        _port_ts(test_ret,
                 models["cae"][best_cae_key].predict(
                     test_ret, test_chars,
                     train_returns=train_ret, train_chars=train_chars),
                 f"ResCAE K={k} λL={lam_lin:.0e} λNL={lam_nonlin:.0e}",
                 colors["ResCAE"])

    if best_nl_key is not None and best_nl_key in models.get("cae_nl", {}):
        k_nl, lam_nl = best_nl_key
        _port_ts(test_ret,
                 models["cae_nl"][best_nl_key].predict(
                     test_ret, test_chars,
                     train_returns=train_ret, train_chars=train_chars),
                 f"CAE-NL K={k_nl} λNL={lam_nl:.0e}  (robustness)",
                 colors["CAE-NL"], lw=1.2, ls="--")

    bnh = np.nanmean(test_ret, axis=1)
    ax.plot(test_dates, np.nancumsum(bnh),
            label="Buy & Hold (EW)", color="black", lw=1.0, linestyle=":")
    ax.axhline(0, color="black", lw=0.5)

    ax.set_title("Cumulative Long-Short Portfolio Returns — Test Period 2020–2024",
                 fontsize=13)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save_fig(fig, os.path.join(figures_dir, "factor_portfolio_cumret.png"))


def plot_phi_r2_scatter(splits: dict,
                         models: dict,
                         sig_results: dict,
                         figures_dir: str = "paper/figures") -> None:
    """scatter of φ vs predictive R² for all ResCAE configs (figure 4). color = DM significance vs IPCA."""
    import os
    from matplotlib.patches import Patch
    os.makedirs(figures_dir, exist_ok=True)

    test_ret    = splits["test"]["returns"].values.astype(np.float32)
    test_chars  = splits["test"]["chars"].astype(np.float32)
    train_ret   = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)

    points = []
    for cae_key, model in models.get("cae", {}).items():
        k, lam_lin, lam_nonlin = cae_key
        r_hat_oos = model.predict(test_ret, test_chars,
                                  train_returns=train_ret, train_chars=train_chars)
        pr2 = predictive_r2(test_ret, r_hat_oos.astype(np.float32))
        _, _, phi = nonlinear_contribution(model, test_ret, test_chars)
        p_ipca = (sig_results.get("per_config", {})
                              .get(cae_key, {})
                              .get("dm_vs_ipca", {})
                              .get("p_value", np.nan))
        drift = compute_ipca_drift(model)
        points.append({"phi": phi, "pred_r2": pr2, "k": k,
                       "lam_nonlin": lam_nonlin, "p_vs_ipca": p_ipca,
                       "high_drift": (drift is not None and drift > 1.0)})

    if not points:
        print("  No ResCAE configs for φ–R² scatter; skipping.")
        return

    # ResCAE-Fixed points: φ identified by construction since W_skip is frozen
    fixed_points = []
    for fixed_key, model in models.get("cae_fixed", {}).items():
        k_f, lam_nl_f = fixed_key
        rh_f = model.predict(test_ret, test_chars,
                             train_returns=train_ret, train_chars=train_chars)
        pr2_f = predictive_r2(test_ret, rh_f.astype(np.float32))
        _, _, phi_f = nonlinear_contribution(model, test_ret, test_chars)
        fixed_points.append({"phi": phi_f, "pred_r2": pr2_f,
                              "k": k_f, "lam_nonlin": lam_nl_f})

    def _col(p):
        if not np.isfinite(p): return "lightgray"
        if p < 0.05:           return "seagreen"
        if p < 0.10:           return "gold"
        return "firebrick"

    fig, ax = plt.subplots(figsize=(7, 5))
    for pt in points:
        marker = "X" if pt["high_drift"] else "o"
        ax.scatter(pt["pred_r2"], pt["phi"],
                   color=_col(pt["p_vs_ipca"]),
                   marker=marker, s=80, edgecolors="black", lw=0.5, zorder=3)
        ax.annotate(f"K={pt['k']}, λNL={pt['lam_nonlin']:.0e}",
                    (pt["pred_r2"], pt["phi"]),
                    fontsize=7, textcoords="offset points", xytext=(5, 3))

    for pt in fixed_points:
        ax.scatter(pt["pred_r2"], pt["phi"],
                   color="mediumpurple", marker="D",
                   s=80, edgecolors="black", lw=0.5, zorder=3)
        ax.annotate(f"F K={pt['k']}, λNL={pt['lam_nonlin']:.0e}",
                    (pt["pred_r2"], pt["phi"]),
                    fontsize=7, textcoords="offset points", xytext=(5, -10))

    best = max(points, key=lambda x: x["pred_r2"] if np.isfinite(x["pred_r2"]) else -np.inf)
    ax.annotate(f"Best: K={best['k']}, λNL={best['lam_nonlin']:.0e}",
                (best["pred_r2"], best["phi"]),
                fontsize=8, fontweight="bold",
                textcoords="offset points", xytext=(10, -12),
                arrowprops=dict(arrowstyle="->", lw=0.8))

    from matplotlib.lines import Line2D as _Line2D
    legend_els = [
        Patch(facecolor="seagreen",     edgecolor="black", label="ResCAE p < 0.05  (vs IPCA)"),
        Patch(facecolor="gold",         edgecolor="black", label="ResCAE p < 0.10"),
        Patch(facecolor="firebrick",    edgecolor="black", label="ResCAE p ≥ 0.10"),
        Patch(facecolor="lightgray",    edgecolor="black", label="IPCA not available"),
        _Line2D([0], [0], marker="D", color="w", markerfacecolor="mediumpurple",
                markeredgecolor="black", markersize=8,
                label="ResCAE-Fixed (φ by construction)"),
        _Line2D([0], [0], marker="X", color="w", markerfacecolor="gray",
                markeredgecolor="black", markersize=8,
                label="× = IPCA init discarded (drift > 1.0)"),
    ]
    ax.legend(handles=legend_els, fontsize=9)
    ax.set_xlabel("Predictive R²", fontsize=11)
    ax.set_ylabel("φ  (Nonlinear Fraction)", fontsize=11)
    ax.set_title("Nonlinear Contribution (φ) vs. Predictive R²", fontsize=13)
    ax.tick_params(labelsize=9)

    fig.text(0.5, -0.06,
             "Each point is one (K, λ_nonlin) configuration.  φ measures the fraction of "
             "loading variance attributable to the nonlinear residual g(z).\n"
             "Color indicates statistical significance of ResCAE vs IPCA (DM test).",
             ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    _save_fig(fig, os.path.join(figures_dir, "phi_decomposition.png"))


def plot_ipca_drift(summary_df: pd.DataFrame,
                    figures_dir: str = "paper/figures") -> None:
    """bar chart of relative W_skip drift from IPCA init per ResCAE config (figure 5). small drift validates φ interpretability."""
    import os
    os.makedirs(figures_dir, exist_ok=True)

    cae_df = summary_df[summary_df["Model"] == "ResCAE"].copy()
    if cae_df.empty or "drift_rel" not in cae_df.columns:
        print("  No ResCAE drift data; skipping IPCA drift plot.")
        return

    cae_df["drift_rel"] = pd.to_numeric(cae_df["drift_rel"], errors="coerce")
    cae_df = cae_df.dropna(subset=["drift_rel"])
    if cae_df.empty:
        print("  All ResCAE drift values are None; skipping IPCA drift plot.")
        return

    labels = [f"K={row.K} λL={row['λ_lin']} λNL={row['λ_nonlin']}"
              for _, row in cae_df.iterrows()]
    drifts = cae_df["drift_rel"].tolist()

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9 + 1), 4))
    ax.bar(labels, drifts, color="seagreen", alpha=0.8, edgecolor="black", lw=0.6)
    ax.axhline(1.0, color="dimgray", lw=1.0, linestyle="--",
               label="100% drift  (‖Δ‖_F = ‖Γ_init‖_F)")
    ax.axhline(0.10, color="darkorange", lw=1.2, linestyle=":",
               label="Low drift threshold  (0.10)")
    # shade the "IPCA initialization discarded" zone
    y_hi = max(max(drifts, default=1.2), 1.2) * 1.05
    ax.axhspan(0.9, y_hi, color="firebrick", alpha=0.08,
               label="IPCA initialization discarded  (drift > 0.9)")

    ax.set_ylabel("‖W_skip − Γ_init‖_F / ‖Γ_init‖_F", fontsize=11)
    ax.set_title(
        "W_skip Drift from IPCA Initialization —\n"
        "How Much Does Joint Optimization Move the Linear Component?",
        fontsize=13)
    ax.set_xlabel("ResCAE configuration", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)

    fig.text(0.5, -0.04,
             "Small drift confirms the linear branch remains close to the IPCA solution,\n"
             "validating the interpretability of the φ decomposition.",
             ha="center", fontsize=8, style="italic")

    plt.xticks(rotation=30, ha="right", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, os.path.join(figures_dir, "ipca_drift.png"))


def get_portfolio_returns(r_true: np.ndarray, r_hat: np.ndarray) -> np.ndarray:
    """monthly long-short decile portfolio return time series. params: {r_true: (T,N), r_hat: (T,N)}. returns (T,) array with NaN for months with fewer than 20 valid stocks."""
    T = r_true.shape[0]
    port = np.full(T, np.nan)
    for t in range(T):
        r_t = r_true[t]; rh = r_hat[t]
        valid = _safe_mask(r_t, rh)
        if valid.sum() < 20:
            continue
        pv = rh[valid]; rv = r_t[valid]
        q10 = np.percentile(pv, 10); q90 = np.percentile(pv, 90)
        if not (pv >= q90).any() or not (pv <= q10).any():
            continue
        port[t] = rv[pv >= q90].mean() - rv[pv <= q10].mean()
    return port


def diebold_mariano_test(errors_1: np.ndarray,
                          errors_2: np.ndarray,
                          max_lag: Optional[int] = None,
                          ) -> dict:
    """Diebold-Mariano test for equal predictive accuracy (Diebold & Mariano 1995). params: {errors_1: (T,N), errors_2: (T,N), max_lag: int|None}. returns dict with dm_stat, p_value, mean_loss_diff, hac_variance, n_lags.

    positive DM means model 1 has larger errors; default lag uses ⌊4(T/100)^{2/9}⌋.
    """
    from scipy.stats import norm as _norm

    sq1 = errors_1.astype(np.float64) ** 2
    sq2 = errors_2.astype(np.float64) ** 2
    both_finite = np.isfinite(sq1) & np.isfinite(sq2)

    T  = errors_1.shape[0]
    d  = np.full(T, np.nan)
    for t in range(T):
        vm = both_finite[t]
        if vm.sum() == 0:
            continue
        d[t] = (sq1[t, vm] - sq2[t, vm]).mean()

    valid_t = np.isfinite(d)
    dropped = int((~valid_t).sum())
    if dropped > 0:
        print(f"    DM: dropped {dropped} months with no valid stocks.")
    d_clean = d[valid_t]
    T_eff   = len(d_clean)

    if T_eff < 30:
        print(f"    WARNING: DM test has only {T_eff} valid months — "
              f"results are unreliable (need ≥ 30).")
    if T_eff == 0:
        return {"dm_stat": np.nan, "p_value": np.nan,
                "mean_loss_diff": np.nan, "hac_variance": np.nan, "n_lags": 0}

    d_bar = d_clean.mean()

    if max_lag is None:
        max_lag = int(np.floor(4 * (T_eff / 100) ** (2 / 9)))
    max_lag = max(max_lag, 0)

    d_dev   = d_clean - d_bar
    gamma0  = float((d_dev ** 2).mean())
    hac_var = gamma0
    for h in range(1, max_lag + 1):
        gamma_h = float((d_dev[h:] * d_dev[:-h]).mean())
        hac_var += 2.0 * (1.0 - h / (max_lag + 1)) * gamma_h

    if hac_var <= 0:
        print(f"    WARNING: Newey-West variance non-positive ({hac_var:.2e}); "
              f"falling back to sample variance.")
        hac_var = gamma0

    dm_stat = d_bar / np.sqrt(hac_var / T_eff)
    p_value = 2.0 * (1.0 - float(_norm.cdf(abs(dm_stat))))

    return {
        "dm_stat":        float(dm_stat),
        "p_value":        float(p_value),
        "mean_loss_diff": float(d_bar),
        "hac_variance":   float(hac_var),
        "n_lags":         max_lag,
    }


def bootstrap_r2_ci(r_true: np.ndarray,
                     r_hat: np.ndarray,
                     n_bootstrap: int = 1000,
                     seed: int = 42,
                     ) -> dict:
    """bootstrap 95% CI for predictive R² by resampling the time dimension. params: {r_true: (T,N), r_hat: (T,N), n_bootstrap: int, seed: int}. returns dict with r2_observed, ci_lower, ci_upper, bootstrap_distribution."""
    T      = r_true.shape[0]
    r2_obs = float(predictive_r2(r_true, r_hat))
    rng    = np.random.default_rng(seed)
    boot_r2 = np.full(n_bootstrap, np.nan)

    for b in range(n_bootstrap):
        idx        = rng.integers(0, T, size=T)
        boot_r2[b] = predictive_r2(r_true[idx], r_hat[idx])

    valid = boot_r2[np.isfinite(boot_r2)]
    if len(valid) == 0:
        return {"r2_observed": r2_obs, "ci_lower": np.nan, "ci_upper": np.nan,
                "bootstrap_distribution": boot_r2}
    return {
        "r2_observed":            r2_obs,
        "ci_lower":               float(np.percentile(valid, 2.5)),
        "ci_upper":               float(np.percentile(valid, 97.5)),
        "bootstrap_distribution": boot_r2,
    }


def bootstrap_sharpe_test(port_returns_1: np.ndarray,
                           port_returns_2: np.ndarray,
                           n_bootstrap: int = 1000,
                           seed: int = 42,
                           ) -> dict:
    """bootstrap test for equality of Sharpe ratios. params: {port_returns_1: (T,), port_returns_2: (T,), n_bootstrap: int, seed: int}. returns dict with sharpe_1, sharpe_2, sharpe_diff, p_value, ci_lower, ci_upper, bootstrap_distribution.

    pairs are resampled jointly to preserve contemporaneous correlation.
    """
    valid   = np.isfinite(port_returns_1) & np.isfinite(port_returns_2)
    dropped = int((~valid).sum())
    if dropped > 0:
        print(f"    Bootstrap Sharpe: dropped {dropped} NaN months.")
    r1 = port_returns_1[valid]
    r2 = port_returns_2[valid]
    T  = len(r1)

    if T < 30:
        print(f"    WARNING: bootstrap Sharpe has only {T} valid months — "
              f"results are unreliable.")

    def _sr(r: np.ndarray) -> float:
        s = r.std(ddof=1)
        return float(r.mean() / s * np.sqrt(12)) if s > 1e-10 else np.nan

    nan_result = {"sharpe_1": np.nan, "sharpe_2": np.nan, "sharpe_diff": np.nan,
                  "p_value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                  "bootstrap_distribution": np.full(n_bootstrap, np.nan)}
    if T == 0:
        return nan_result

    sr1       = _sr(r1)
    sr2       = _sr(r2)
    delta_obs = sr1 - sr2 if (np.isfinite(sr1) and np.isfinite(sr2)) else np.nan

    rng        = np.random.default_rng(seed)
    boot_delta = np.full(n_bootstrap, np.nan)
    for b in range(n_bootstrap):
        idx           = rng.integers(0, T, size=T)
        boot_delta[b] = _sr(r1[idx]) - _sr(r2[idx])

    bv = boot_delta[np.isfinite(boot_delta)]
    if len(bv) == 0 or not np.isfinite(delta_obs):
        return {**nan_result, "sharpe_1": sr1, "sharpe_2": sr2,
                "sharpe_diff": delta_obs, "bootstrap_distribution": boot_delta}

    centered = bv - delta_obs
    p_value  = float((np.abs(centered) >= np.abs(delta_obs)).mean())

    return {
        "sharpe_1":               sr1,
        "sharpe_2":               sr2,
        "sharpe_diff":            float(delta_obs),
        "p_value":                p_value,
        "ci_lower":               float(np.percentile(bv, 2.5)),
        "ci_upper":               float(np.percentile(bv, 97.5)),
        "bootstrap_distribution": boot_delta,
    }


def plot_lambda_interaction(summary_df: pd.DataFrame,
                            figures_dir: str = "paper/figures") -> None:
    """heatmaps showing how λ_lin and λ_nonlin jointly affect φ and Pred_R². one subplot per K; two figures produced."""
    import os
    os.makedirs(figures_dir, exist_ok=True)

    cae_df = summary_df[summary_df["Model"] == "ResCAE"].copy()
    if cae_df.empty:
        print("  No ResCAE data; skipping lambda interaction plots.")
        return

    for col in ["λ_lin", "λ_nonlin", "NL_frac", "Pred_R2"]:
        cae_df[col] = pd.to_numeric(cae_df[col], errors="coerce")

    k_values = sorted(cae_df["K"].unique())
    if not k_values:
        return

    for metric, metric_label, fname in [
        ("NL_frac", "Nonlinear Fraction φ", "lambda_interaction_phi.png"),
        ("Pred_R2", "Predictive R²",         "lambda_interaction_r2.png"),
    ]:
        fig, axes = plt.subplots(1, len(k_values),
                                  figsize=(5 * len(k_values) + 1, 4),
                                  squeeze=False)
        for ax, k in zip(axes[0], k_values):
            sub   = cae_df[cae_df["K"] == k]
            pivot = sub.pivot_table(index="λ_lin", columns="λ_nonlin",
                                    values=metric, aggfunc="mean")
            sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f",
                        cmap="RdYlGn", linewidths=0.5)
            ax.set_title(f"K={k}", fontsize=11)
            ax.set_xlabel("λ_nonlin", fontsize=9)
            ax.set_ylabel("λ_lin", fontsize=9)

        fig.suptitle(f"Lambda Interaction: {metric_label}", fontsize=13, y=1.02)
        fig.tight_layout()
        _save_fig(fig, os.path.join(figures_dir, fname))


def phi_decomposition_analysis(summary_df: pd.DataFrame,
                                figures_dir: str = "paper/figures") -> None:
    """isolate parameter-count asymmetry from regularisation asymmetry in φ. diagonal configs (λ_lin==λ_nonlin) isolate parameter-count effects; skipped if fewer than 3 diagonal configs."""
    cae_df = summary_df[summary_df["Model"] == "ResCAE"].copy()
    if cae_df.empty:
        return

    for col in ["λ_lin", "λ_nonlin", "NL_frac"]:
        cae_df[col] = pd.to_numeric(cae_df[col], errors="coerce")
    cae_df = cae_df.dropna(subset=["NL_frac"])

    diag_mask  = cae_df["λ_lin"] == cae_df["λ_nonlin"]
    diag_df    = cae_df[diag_mask]
    offdiag_df = cae_df[~diag_mask]

    if len(diag_df) < 3:
        print("  phi_decomposition_analysis: fewer than 3 diagonal configurations "
              "available; skipping.")
        return

    print("\n--- φ Decomposition: Diagonal configs (λ_lin == λ_nonlin) ---")
    d_cols = ["K", "λ_lin", "λ_nonlin", "NL_frac"]
    print(diag_df[d_cols].sort_values(["K", "λ_lin"]).to_string(index=False))

    if not offdiag_df.empty:
        print("\n--- φ Decomposition: Off-diagonal configs (λ_lin ≠ λ_nonlin) ---")
        print(offdiag_df[d_cols].sort_values(["K", "λ_lin", "λ_nonlin"])
              .to_string(index=False))

    print("\n--- Mean φ by K (diagonal vs off-diagonal) ---")
    for k in sorted(cae_df["K"].unique()):
        k_diag = diag_df[diag_df["K"] == k]["NL_frac"].mean()
        k_off  = (offdiag_df[offdiag_df["K"] == k]["NL_frac"].mean()
                  if not offdiag_df.empty else float("nan"))
        print(f"  K={k}:  diagonal mean φ = {k_diag:.4f},  "
              f"off-diagonal mean φ = {k_off:.4f}")


def run_significance_tests(splits: dict,
                            models: dict,
                            figures_dir: str = "paper/figures",
                            n_bootstrap: int = 1000,
                            results_dir: str = "results",
                            ) -> dict:
    """run pairwise DM and bootstrap Sharpe tests on the test set. params: {splits, models, figures_dir, n_bootstrap, results_dir}. returns dict saved to results/significance_tests.pkl.

    primary: ResCAE vs IPCA. supporting: vs PCA, vs CAE-NL. IPCA comparisons skipped if not in models["ipca"].
    """
    import os, pickle
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    test_ret    = splits["test"]["returns"].values.astype(np.float32)
    test_chars  = splits["test"]["chars"].astype(np.float32)
    train_ret   = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)

    have_ipca  = "ipca" in models and len(models["ipca"]) > 0
    have_fixed = "cae_fixed" in models and len(models.get("cae_fixed", {})) > 0

    # resolve per-K best ResCAE from saved hparam search
    hparam_path = os.path.join(results_dir, "cae_hparam_search.pkl")
    scores_by_config: dict = {}
    if os.path.exists(hparam_path):
        with open(hparam_path, "rb") as _f:
            _hd = pickle.load(_f)
        scores_by_config = _hd.get("pred_val_mse") or _hd.get("val_losses", {})

    def _best_rescae_for_k(k: int):
        k_cfgs = {key: sc for key, sc in scores_by_config.items()
                  if key[0] == k and key in models["cae"]}
        if k_cfgs:
            return min(k_cfgs, key=k_cfgs.get)
        for key in models["cae"]:
            if key[0] == k:
                return key
        return None

    # resolve per-K best ResCAE-Fixed
    fixed_hparam_path = os.path.join(results_dir, "cae_fixed_hparam_search.pkl")
    fixed_scores: dict = {}
    if os.path.exists(fixed_hparam_path):
        with open(fixed_hparam_path, "rb") as _f:
            _fd = pickle.load(_f)
        fixed_scores = _fd.get("pred_val_mse") or _fd.get("val_losses", {})

    def _best_fixed_for_k(k: int):
        k_cfgs = {key: sc for key, sc in fixed_scores.items()
                  if key[0] == k and key in models.get("cae_fixed", {})}
        if k_cfgs:
            return min(k_cfgs, key=k_cfgs.get)
        for key in models.get("cae_fixed", {}):
            if key[0] == k:
                return key
        return None

    def _pred(mtype: str, k: int, cae_key=None):
        if mtype == "pca":
            rh = models["pca"][k].predict(test_ret, train_returns=train_ret)
        elif mtype == "ipca":
            rh = models["ipca"][k].predict(
                test_ret, test_chars,
                train_returns=train_ret, train_chars=train_chars)
        elif mtype == "cae_nl":
            rh = models["cae_nl"][cae_key].predict(
                test_ret, test_chars,
                train_returns=train_ret, train_chars=train_chars)
        elif mtype == "cae_fixed":
            rh = models["cae_fixed"][cae_key].predict(
                test_ret, test_chars,
                train_returns=train_ret, train_chars=train_chars)
        else:  # "rescae"
            rh = models["cae"][cae_key].predict(
                test_ret, test_chars,
                train_returns=train_ret, train_chars=train_chars)
        err = test_ret.astype(np.float64) - rh.astype(np.float64)
        return rh.astype(np.float64), err

    def _sig(p: float) -> str:
        if not np.isfinite(p): return ""
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    k_values = sorted(models["pca"].keys())
    sig_results: dict = {}

    # per-K comparisons
    for k in k_values:
        rescae_key = _best_rescae_for_k(k)
        if rescae_key is None:
            continue

        rh_pca,    err_pca    = _pred("pca",    k)
        rh_rescae, err_rescae = _pred("rescae", k, rescae_key)

        entry: dict = {"rescae_key": rescae_key}
        entry["dm_rescae_vs_pca"] = diebold_mariano_test(err_pca, err_rescae)
        entry["boot_sharpe_rescae_vs_pca"] = bootstrap_sharpe_test(
            get_portfolio_returns(test_ret, rh_rescae),
            get_portfolio_returns(test_ret, rh_pca),
            n_bootstrap)
        entry["boot_r2_pca"]    = bootstrap_r2_ci(test_ret, rh_pca,    n_bootstrap)
        entry["boot_r2_rescae"] = bootstrap_r2_ci(test_ret, rh_rescae, n_bootstrap)

        rh_ipca = err_ipca = None
        if have_ipca and k in models["ipca"]:
            rh_ipca, err_ipca = _pred("ipca", k)
            entry["dm_rescae_vs_ipca"] = diebold_mariano_test(err_ipca, err_rescae)
            entry["dm_ipca_vs_pca"]    = diebold_mariano_test(err_pca,  err_ipca)
            entry["boot_sharpe_rescae_vs_ipca"] = bootstrap_sharpe_test(
                get_portfolio_returns(test_ret, rh_rescae),
                get_portfolio_returns(test_ret, rh_ipca),
                n_bootstrap)
            entry["boot_r2_ipca"] = bootstrap_r2_ci(test_ret, rh_ipca, n_bootstrap)

        if have_fixed:
            fixed_key = _best_fixed_for_k(k)
            if fixed_key is not None and fixed_key in models.get("cae_fixed", {}):
                rh_fixed, err_fixed = _pred("cae_fixed", k, fixed_key)
                entry["fixed_key"]            = fixed_key
                entry["boot_r2_rescae_fixed"] = bootstrap_r2_ci(
                    test_ret, rh_fixed, n_bootstrap)
                entry["dm_rescae_fixed_vs_pca"] = diebold_mariano_test(
                    err_pca, err_fixed)
                if rh_ipca is not None:
                    entry["dm_rescae_fixed_vs_ipca"] = diebold_mariano_test(
                        err_ipca, err_fixed)
                    entry["boot_sharpe_rescae_fixed_vs_ipca"] = bootstrap_sharpe_test(
                        get_portfolio_returns(test_ret, rh_fixed),
                        get_portfolio_returns(test_ret, rh_ipca),
                        n_bootstrap)
                entry["dm_rescae_vs_rescae_fixed"] = diebold_mariano_test(
                    err_fixed, err_rescae)
                entry["boot_sharpe_rescae_vs_rescae_fixed"] = bootstrap_sharpe_test(
                    get_portfolio_returns(test_ret, rh_rescae),
                    get_portfolio_returns(test_ret, rh_fixed),
                    n_bootstrap)

        sig_results[k] = entry

    # overall best ResCAE across all K
    avail = {k: v for k, v in scores_by_config.items() if k in models["cae"]}
    overall_best = (min(avail, key=avail.get) if avail
                    else next(iter(models["cae"]), None))
    if overall_best is not None:
        k_ob = overall_best[0]
        rh_rescae_ob, err_rescae_ob = _pred("rescae", k_ob, overall_best)
        port_rescae_ob = get_portfolio_returns(test_ret, rh_rescae_ob)
        ob_entry: dict = {"rescae_key": overall_best}
        if k_ob in models["pca"]:
            rh_pca_ob, err_pca_ob = _pred("pca", k_ob)
            ob_entry["dm_rescae_vs_pca"] = diebold_mariano_test(err_pca_ob, err_rescae_ob)
            ob_entry["boot_sharpe_rescae_vs_pca"] = bootstrap_sharpe_test(
                port_rescae_ob,
                get_portfolio_returns(test_ret, rh_pca_ob),
                n_bootstrap)
        if have_ipca and k_ob in models["ipca"]:
            rh_ipca_ob, err_ipca_ob = _pred("ipca", k_ob)
            ob_entry["dm_rescae_vs_ipca"] = diebold_mariano_test(err_ipca_ob, err_rescae_ob)
            ob_entry["boot_sharpe_rescae_vs_ipca"] = bootstrap_sharpe_test(
                port_rescae_ob,
                get_portfolio_returns(test_ret, rh_ipca_ob),
                n_bootstrap)
        sig_results["overall"] = ob_entry

    # per-config DM tests for all ResCAE configs
    pca_pred_cache:  dict = {}
    ipca_pred_cache: dict = {}
    per_config:      dict = {}

    for cae_key in models["cae"]:
        k_c = cae_key[0]
        if k_c not in models["pca"]:
            continue
        if k_c not in pca_pred_cache:
            pca_pred_cache[k_c] = _pred("pca", k_c)
        _, err_rescae = _pred("rescae", k_c, cae_key)
        _, err_pca    = pca_pred_cache[k_c]
        entry_pc: dict = {"dm_vs_pca": diebold_mariano_test(err_pca, err_rescae)}
        if have_ipca and k_c in models["ipca"]:
            if k_c not in ipca_pred_cache:
                ipca_pred_cache[k_c] = _pred("ipca", k_c)
            _, err_ipca = ipca_pred_cache[k_c]
            entry_pc["dm_vs_ipca"] = diebold_mariano_test(err_ipca, err_rescae)
        per_config[cae_key] = entry_pc
    sig_results["per_config"] = per_config

    # per-config DM tests for all CAE-NL configs
    per_config_nl: dict = {}
    for nl_key in models.get("cae_nl", {}):
        k_c = nl_key[0]
        if k_c not in models["pca"]:
            continue
        if k_c not in pca_pred_cache:
            pca_pred_cache[k_c] = _pred("pca", k_c)
        _, err_nl  = _pred("cae_nl", k_c, nl_key)
        _, err_pca = pca_pred_cache[k_c]
        entry_nl: dict = {"dm_vs_pca": diebold_mariano_test(err_pca, err_nl)}
        if have_ipca and k_c in models["ipca"]:
            if k_c not in ipca_pred_cache:
                ipca_pred_cache[k_c] = _pred("ipca", k_c)
            _, err_ipca = ipca_pred_cache[k_c]
            entry_nl["dm_vs_ipca"] = diebold_mariano_test(err_ipca, err_nl)
        per_config_nl[nl_key] = entry_nl
    sig_results["per_config_nl"] = per_config_nl

    # per-config DM tests for all ResCAE-Fixed configs
    per_config_fixed: dict = {}
    for fixed_key in models.get("cae_fixed", {}):
        k_c = fixed_key[0]
        if k_c not in models["pca"]:
            continue
        if k_c not in pca_pred_cache:
            pca_pred_cache[k_c] = _pred("pca", k_c)
        _, err_fixed_c = _pred("cae_fixed", k_c, fixed_key)
        _, err_pca_c   = pca_pred_cache[k_c]
        entry_fc: dict = {"dm_vs_pca": diebold_mariano_test(err_pca_c, err_fixed_c)}
        if have_ipca and k_c in models["ipca"]:
            if k_c not in ipca_pred_cache:
                ipca_pred_cache[k_c] = _pred("ipca", k_c)
            _, err_ipca_c = ipca_pred_cache[k_c]
            entry_fc["dm_vs_ipca"] = diebold_mariano_test(err_ipca_c, err_fixed_c)
        per_config_fixed[fixed_key] = entry_fc
    sig_results["per_config_fixed"] = per_config_fixed

    # DM tests vs best CAE-NL per K
    nl_hparam_path = os.path.join(results_dir, "cae_nl_hparam_search.pkl")
    nl_scores: dict = {}
    if os.path.exists(nl_hparam_path):
        with open(nl_hparam_path, "rb") as _f:
            _nl_data = pickle.load(_f)
        nl_scores = _nl_data.get("pred_val_mse") or _nl_data.get("val_losses", {})

    vs_cae_nl: dict = {}
    for k in k_values:
        k_nl_cfgs = {key: nl_scores.get(key, float("inf"))
                     for key in models.get("cae_nl", {}) if key[0] == k}
        if not k_nl_cfgs:
            continue
        best_nl_key = min(k_nl_cfgs, key=k_nl_cfgs.get)
        rh_nl, err_nl = _pred("cae_nl", k, best_nl_key)
        entry_vs: dict = {"cae_nl_key": best_nl_key}

        if k in models["pca"]:
            _, err_pca = pca_pred_cache.get(k) or _pred("pca", k)
            entry_vs["dm_pca_vs_cae_nl"] = diebold_mariano_test(err_pca, err_nl)

        rescae_key = _best_rescae_for_k(k)
        if rescae_key is not None:
            rh_rescae, err_rescae = _pred("rescae", k, rescae_key)
            entry_vs["dm_rescae_vs_cae_nl"] = diebold_mariano_test(err_rescae, err_nl)
            entry_vs["boot_sharpe_rescae_vs_cae_nl"] = bootstrap_sharpe_test(
                get_portfolio_returns(test_ret, rh_rescae),
                get_portfolio_returns(test_ret, rh_nl),
                n_bootstrap)

        vs_cae_nl[k] = entry_vs
    sig_results["vs_cae_nl"] = vs_cae_nl

    # print DM table (primary: ResCAE vs IPCA first)
    print("\n--- Diebold-Mariano Results  (positive DM = right model better) ---")
    print("  PRIMARY: ResCAE vs IPCA — does the nonlinear residual add value?")
    hdr = f"{'Comparison':<26} {'K':>3} {'DM':>8} {'p':>8} {'sig':>4}"
    print(hdr)
    print("-" * len(hdr))

    def _dm_row(lbl, r):
        dm, p = r.get("dm_stat", np.nan), r.get("p_value", np.nan)
        return dm, p

    for k in k_values:
        if k not in sig_results:
            continue
        if "dm_rescae_vs_ipca" in sig_results[k]:
            r = sig_results[k]["dm_rescae_vs_ipca"]
            dm, p = _dm_row("ResCAE vs IPCA [PRIMARY]", r)
            if np.isfinite(dm):
                print(f"{'ResCAE vs IPCA [PRIMARY]':<26} {k:>3} {dm:>8.3f} {p:>8.4f} {_sig(p):>4}")
        for lbl, key in [("ResCAE vs PCA",    "dm_rescae_vs_pca"),
                          ("ResCAE vs CAE-NL", None)]:
            if key:
                r = sig_results[k].get(key, {})
                dm, p = _dm_row(lbl, r)
                if np.isfinite(dm):
                    print(f"{lbl:<26} {k:>3} {dm:>8.3f} {p:>8.4f} {_sig(p):>4}")
        r = sig_results.get("vs_cae_nl", {}).get(k, {}).get("dm_rescae_vs_cae_nl", {})
        dm, p = _dm_row("ResCAE vs CAE-NL", r)
        if np.isfinite(dm):
            print(f"{'ResCAE vs CAE-NL':<26} {k:>3} {dm:>8.3f} {p:>8.4f} {_sig(p):>4}")

    if have_fixed:
        print("\n  FIXED PRIMARY: ResCAE-Fixed vs IPCA — frozen nonlinear residual add value?")
        for k in k_values:
            if k not in sig_results:
                continue
            for lbl, key in [("ResCAE-Fixed vs IPCA [PRI-F]", "dm_rescae_fixed_vs_ipca"),
                              ("ResCAE-Fixed vs PCA",           "dm_rescae_fixed_vs_pca")]:
                r = sig_results[k].get(key, {})
                dm, p = _dm_row(lbl, r)
                if np.isfinite(dm):
                    print(f"{lbl:<30} {k:>3} {dm:>8.3f} {p:>8.4f} {_sig(p):>4}")
        print("\n  FIXED SECONDARY: ResCAE vs ResCAE-Fixed — does relaxing W_skip help?")
        for k in k_values:
            if k not in sig_results:
                continue
            r = sig_results[k].get("dm_rescae_vs_rescae_fixed", {})
            dm, p = _dm_row("ResCAE vs ResCAE-Fixed", r)
            if np.isfinite(dm):
                print(f"{'ResCAE vs ResCAE-Fixed [SEC]':<30} {k:>3} {dm:>8.3f} {p:>8.4f} {_sig(p):>4}")

    # print bootstrap Sharpe table
    print("\n--- Bootstrap Sharpe Test Results ---")
    print(f"{'Comparison':<26} {'K':>3} {'SR1':>6} {'SR2':>6} "
          f"{'ΔSR':>7} {'p':>7} {'CI 95%':>20} {'sig':>4}")
    print("-" * 86)
    for k in k_values:
        if k not in sig_results:
            continue
        for lbl, key in [("ResCAE vs IPCA [PRIMARY]", "boot_sharpe_rescae_vs_ipca"),
                          ("ResCAE vs PCA",            "boot_sharpe_rescae_vs_pca")]:
            r = sig_results[k].get(key, {})
            p = r.get("p_value", np.nan)
            if not np.isfinite(r.get("sharpe_diff", np.nan)):
                continue
            ci = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            print(f"{lbl:<26} {k:>3} {r['sharpe_1']:>6.3f} {r['sharpe_2']:>6.3f} "
                  f"{r['sharpe_diff']:>7.3f} {p:>7.4f} {ci:>20} {_sig(p):>4}")
        r = (sig_results.get("vs_cae_nl", {})
                        .get(k, {})
                        .get("boot_sharpe_rescae_vs_cae_nl", {}))
        p = r.get("p_value", np.nan)
        if np.isfinite(r.get("sharpe_diff", np.nan)):
            ci = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            print(f"{'ResCAE vs CAE-NL':<26} {k:>3} {r['sharpe_1']:>6.3f} "
                  f"{r['sharpe_2']:>6.3f} {r['sharpe_diff']:>7.3f} "
                  f"{p:>7.4f} {ci:>20} {_sig(p):>4}")

    if have_fixed:
        for k in k_values:
            if k not in sig_results:
                continue
            for lbl, key in [
                ("ResCAE-Fixed vs IPCA [PRI-F]", "boot_sharpe_rescae_fixed_vs_ipca"),
                ("ResCAE vs ResCAE-Fixed [SEC]",  "boot_sharpe_rescae_vs_rescae_fixed"),
            ]:
                r = sig_results[k].get(key, {})
                p = r.get("p_value", np.nan)
                if not np.isfinite(r.get("sharpe_diff", np.nan)):
                    continue
                ci = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
                print(f"{lbl:<30} {k:>3} {r['sharpe_1']:>6.3f} {r['sharpe_2']:>6.3f} "
                      f"{r['sharpe_diff']:>7.3f} {p:>7.4f} {ci:>20} {_sig(p):>4}")

    pkl_path = os.path.join(results_dir, "significance_tests.pkl")
    with open(pkl_path, "wb") as _f:
        pickle.dump(sig_results, _f)
    print(f"\n  Significance results → {pkl_path}")

    return sig_results


def plot_significance_results(sig_results: dict,
                               figures_dir: str = "paper/figures") -> None:
    """produce figures 2 and 3: DM bar chart (ResCAE vs IPCA first) and dot-whisker R² CIs with IPCA reference line."""
    import os
    os.makedirs(figures_dir, exist_ok=True)

    k_values = sorted(k for k in sig_results if isinstance(k, int))

    # figure 2: DM bar chart (primary comparison first)
    labels, stats, pvals, bar_colors = [], [], [], []
    _gold           = "#E6AC00"
    _steelblue      = "steelblue"
    _dimgray        = "dimgray"
    _mediumpurple   = "mediumpurple"
    _plum           = "plum"

    def _collect_dm(lbl, dm_val, p_val, color):
        if np.isfinite(dm_val):
            labels.append(lbl); stats.append(dm_val)
            pvals.append(p_val); bar_colors.append(color)

    for k in k_values:
        r = sig_results.get(k, {}).get("dm_rescae_vs_ipca", {})
        _collect_dm(f"ResCAE vs IPCA  K={k}",
                    r.get("dm_stat", np.nan), r.get("p_value", np.nan), _gold)
    for k in k_values:
        r = sig_results.get(k, {}).get("dm_rescae_fixed_vs_ipca", {})
        _collect_dm(f"ResCAE-Fixed vs IPCA  K={k}",
                    r.get("dm_stat", np.nan), r.get("p_value", np.nan), _mediumpurple)
    for k in k_values:
        r = sig_results.get(k, {}).get("dm_rescae_vs_rescae_fixed", {})
        _collect_dm(f"ResCAE vs ResCAE-Fixed  K={k}",
                    r.get("dm_stat", np.nan), r.get("p_value", np.nan), _plum)
    for k in k_values:
        r = sig_results.get(k, {}).get("dm_rescae_vs_pca", {})
        _collect_dm(f"ResCAE vs PCA  K={k}",
                    r.get("dm_stat", np.nan), r.get("p_value", np.nan), _steelblue)
    for k in k_values:
        r = sig_results.get("vs_cae_nl", {}).get(k, {}).get("dm_rescae_vs_cae_nl", {})
        _collect_dm(f"ResCAE vs CAE-NL  K={k}",
                    r.get("dm_stat", np.nan), r.get("p_value", np.nan), _dimgray)

    if labels:
        n   = len(labels)
        fig, ax = plt.subplots(figsize=(9, max(4, n * 0.6 + 1)))
        y   = np.arange(n)
        bars = ax.barh(y, stats, color=bar_colors, alpha=0.85,
                       edgecolor="black", lw=0.5)
        ax.axvline( 1.645, color="dimgray", lw=1.0, ls="--", label="±1.645  (10%)")
        ax.axvline(-1.645, color="dimgray", lw=1.0, ls="--")
        ax.axvline( 1.960, color="black",   lw=1.0, ls=":",  label="±1.960  (5%)")
        ax.axvline(-1.960, color="black",   lw=1.0, ls=":")
        ax.axvline(0,      color="black",   lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("DM statistic  (positive = right model better)", fontsize=11)
        ax.set_title("Diebold-Mariano Tests: Does the Nonlinear Residual Add Value?",
                     fontsize=13)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc="lower right")

        x_range = max(abs(s) for s in stats) if stats else 1.0
        offset  = x_range * 0.03
        for bar, p in zip(bars, pvals):
            w = bar.get_width()
            ax.text(w + (offset if w >= 0 else -offset),
                    bar.get_y() + bar.get_height() / 2,
                    f"p={p:.3f}", va="center",
                    ha="left" if w >= 0 else "right", fontsize=7)

        fig.tight_layout()
        _save_fig(fig, os.path.join(figures_dir, "dm_test_results.png"))

    # figure 3: R² dot-whisker CI
    cmap = {"pca":          "steelblue",
            "ipca":         "darkorange",
            "rescae_fixed": "mediumpurple",
            "rescae":       "seagreen",
            "cae_nl":       "dimgray"}
    r2_labels, r2_obs, r2_lo, r2_hi, r2_cols = [], [], [], [], []

    for k in k_values:
        if k not in sig_results:
            continue
        for mtype, key in [("pca",          "boot_r2_pca"),
                            ("ipca",         "boot_r2_ipca"),
                            ("rescae_fixed", "boot_r2_rescae_fixed"),
                            ("rescae",       "boot_r2_rescae")]:
            r   = sig_results[k].get(key, {})
            obs = r.get("r2_observed", np.nan)
            if not np.isfinite(obs):
                continue
            r2_labels.append(f"{mtype.upper()}  K={k}")
            r2_obs.append(obs)
            r2_lo.append(r.get("ci_lower", np.nan))
            r2_hi.append(r.get("ci_upper", np.nan))
            r2_cols.append(cmap[mtype])

    if r2_labels:
        from matplotlib.lines import Line2D
        n   = len(r2_labels)
        fig, ax = plt.subplots(figsize=(8, max(4, n * 0.5 + 1)))
        y   = np.arange(n)

        for i, (lo, hi, obs, col) in enumerate(zip(r2_lo, r2_hi, r2_obs, r2_cols)):
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [i, i], color=col, lw=2.5, alpha=0.7)
            ax.scatter(obs, i, color=col, zorder=3, s=55)

        ipca_r2s = [r for r, l in zip(r2_obs, r2_labels) if "IPCA" in l]
        if ipca_r2s:
            best_ipca_r2 = max(ipca_r2s)
            ax.axvline(best_ipca_r2, color="darkorange", lw=1.5, ls="--", alpha=0.7,
                       label=f"Best IPCA R² = {best_ipca_r2:.4f}")
            ax.legend(fontsize=8, loc="lower right")

        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(r2_labels, fontsize=9)
        ax.set_xlabel("Predictive R²  (95% bootstrap CI)", fontsize=11)
        ax.set_title("Predictive R² with 95% Bootstrap Confidence Intervals",
                     fontsize=13)
        ax.tick_params(labelsize=9)

        lbl_map = {"pca": "PCA", "ipca": "IPCA",
                   "rescae_fixed": "ResCAE-Fixed", "rescae": "ResCAE"}
        legend_els = [Line2D([0], [0], color=cmap[k], lw=2.5, label=lbl_map[k])
                      for k in ["pca", "ipca", "rescae_fixed", "rescae"]]
        ax.legend(handles=legend_els, fontsize=8, loc="lower right")

        fig.tight_layout()
        _save_fig(fig, os.path.join(figures_dir, "r2_bootstrap_ci.png"))


def plot_residual_improvement(splits: dict,
                               models: dict,
                               sig_results: dict,
                               figures_dir: str = "paper/figures") -> None:
    """bar chart of sequential R² improvement PCA → IPCA → ResCAE (figure 6). uses best K; includes bootstrap CI error bars and significance brackets."""
    import os
    os.makedirs(figures_dir, exist_ok=True)

    k_values = sorted(k for k in sig_results if isinstance(k, int))
    if not k_values:
        print("  No sig_results for residual improvement plot; skipping.")
        return

    best_k, best_r2 = None, -np.inf
    for k in k_values:
        r2 = sig_results[k].get("boot_r2_rescae", {}).get("r2_observed", -np.inf)
        if np.isfinite(r2) and r2 > best_r2:
            best_r2, best_k = r2, k

    if best_k is None:
        print("  No valid ResCAE R² for residual improvement plot; skipping.")
        return

    entry = sig_results[best_k]
    model_labels = ["PCA", "IPCA", "ResCAE-Fixed", "ResCAE"]
    r2_keys      = ["boot_r2_pca", "boot_r2_ipca",
                    "boot_r2_rescae_fixed", "boot_r2_rescae"]
    colors       = ["steelblue", "darkorange", "mediumpurple", "seagreen"]

    r2_obs, r2_lo, r2_hi, col_plot, lbl_plot = [], [], [], [], []
    for lbl, k_r2, col in zip(model_labels, r2_keys, colors):
        r = entry.get(k_r2, {})
        obs = r.get("r2_observed", np.nan)
        if not np.isfinite(obs):
            continue
        r2_obs.append(obs); r2_lo.append(r.get("ci_lower", np.nan))
        r2_hi.append(r.get("ci_upper", np.nan))
        col_plot.append(col); lbl_plot.append(lbl)

    if not r2_obs:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    x    = np.arange(len(lbl_plot))
    ax.bar(x, r2_obs, color=col_plot, edgecolor="black", lw=0.7, alpha=0.85)

    for i, (lo, hi, obs) in enumerate(zip(r2_lo, r2_hi, r2_obs)):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar(i, obs, yerr=[[obs - lo], [hi - obs]],
                        fmt="none", color="black", capsize=4, lw=1.2)

    # significance brackets
    hi_finite = [h for h in r2_hi if np.isfinite(h)]
    y_top     = (max(hi_finite) if hi_finite else max(r2_obs)) * 1.08
    brk_h     = max(r2_obs) * 0.04

    def _bracket(x1, x2, y, p_val):
        sig_str = f"p={p_val:.3f}" if np.isfinite(p_val) else "N/A"
        ax.plot([x1, x1, x2, x2], [y, y + brk_h * 0.5, y + brk_h * 0.5, y],
                color="black", lw=0.8)
        ax.text((x1 + x2) / 2, y + brk_h * 0.55,
                sig_str, ha="center", va="bottom", fontsize=8)

    if "IPCA" in lbl_plot and "PCA" in lbl_plot:
        p = entry.get("dm_ipca_vs_pca", {}).get("p_value", np.nan)
        _bracket(lbl_plot.index("PCA"), lbl_plot.index("IPCA"), y_top, p)

    if "ResCAE-Fixed" in lbl_plot and "IPCA" in lbl_plot:
        p = entry.get("dm_rescae_fixed_vs_ipca", {}).get("p_value", np.nan)
        _bracket(lbl_plot.index("IPCA"), lbl_plot.index("ResCAE-Fixed"),
                 y_top + brk_h * 1.2, p)

    if "ResCAE" in lbl_plot and "ResCAE-Fixed" in lbl_plot:
        p = entry.get("dm_rescae_vs_rescae_fixed", {}).get("p_value", np.nan)
        _bracket(lbl_plot.index("ResCAE-Fixed"), lbl_plot.index("ResCAE"),
                 y_top + brk_h * 2.4, p)

    ax.set_xticks(x)
    ax.set_xticklabels(lbl_plot, fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Predictive R²  (Out-of-Sample)", fontsize=11)
    ax.set_title(f"Sequential Improvement: Does Each Step Add Value?  (K={best_k})",
                 fontsize=13)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    _save_fig(fig, os.path.join(figures_dir, "residual_improvement.png"))


def format_paper_story(summary_df: pd.DataFrame, sig_results: dict) -> str:
    """format the paper's central narrative as a terminal block. uses the K with the highest ResCAE Pred_R² as the representative config."""
    rescae_rows = summary_df[summary_df["Model"] == "ResCAE"].copy()
    rescae_rows["_pr2"] = pd.to_numeric(rescae_rows["Pred_R2"], errors="coerce")
    if rescae_rows.empty or rescae_rows["_pr2"].isna().all():
        return "[No ResCAE results available]"

    best_k = int(rescae_rows.loc[rescae_rows["_pr2"].idxmax(), "K"])

    def _row(model):
        r = summary_df[(summary_df["Model"] == model) & (summary_df["K"] == best_k)]
        return r.iloc[0] if not r.empty else None

    pca_row    = _row("PCA")
    ipca_row   = _row("IPCA")
    fixed_row  = _row("ResCAE-Fixed")
    rescae_row = _row("ResCAE")
    nl_row     = _row("CAE-NL (robustness)")
    sig_k      = sig_results.get(best_k, {})

    def _fv(val, fmt=".4f"):
        try:    return format(float(val), fmt)
        except: return "N/A"

    def _dm(key, src=None):
        r = (src or sig_k).get(key, {})
        return _fv(r.get("dm_stat"), ".3f"), _fv(r.get("p_value"), ".4f")

    lines = ["=" * 66, "RESULTS — Test Set 2020–2024", "=" * 66, ""]

    lines.append(f"Step 1: Do characteristics matter?  (PCA → IPCA)  [K={best_k}]")
    if pca_row is not None:
        lines.append(f"  PCA    Pred R²: {pca_row['Pred_R2']}   Sharpe: {pca_row['Sharpe']}")
    if ipca_row is not None and pd.notna(ipca_row.get("Pred_R2")):
        lines.append(f"  IPCA   Pred R²: {ipca_row['Pred_R2']}   Sharpe: {ipca_row['Sharpe']}")
    else:
        lines.append("  IPCA   [not yet trained]")
    dm_stat, p_val = _dm("dm_ipca_vs_pca")
    lines.append(f"  DM test (IPCA vs PCA):              stat={dm_stat},  p={p_val}")
    lines.append("")

    lines.append(f"Step 2a: Frozen nonlinear residual add value?  (IPCA → ResCAE-Fixed)  [K={best_k}]")
    if fixed_row is not None and pd.notna(fixed_row.get("Pred_R2")):
        phi_f = fixed_row.get("NL_frac", "N/A")
        lines.append(f"  ResCAE-Fixed  Pred R²: {fixed_row['Pred_R2']}   "
                     f"Sharpe: {fixed_row['Sharpe']}   φ={phi_f}  (W_skip=Γ_IPCA frozen)")
    else:
        lines.append("  ResCAE-Fixed  [not trained]")
    dm_stat, p_val = _dm("dm_rescae_fixed_vs_ipca")
    lines.append(f"  DM test (ResCAE-Fixed vs IPCA):     stat={dm_stat},  p={p_val}  "
                 f"← PRIMARY-FIXED RESULT")
    lines.append("")

    lines.append(f"Step 2b: Does relaxing W_skip add more?  (ResCAE-Fixed → ResCAE)  [K={best_k}]")
    if rescae_row is not None:
        phi   = rescae_row.get("NL_frac",  "N/A")
        drift = rescae_row.get("drift_rel", "N/A")
        lines.append(f"  ResCAE  Pred R²: {rescae_row['Pred_R2']}   "
                     f"Sharpe: {rescae_row['Sharpe']}   φ={phi}")
    dm_stat, p_val = _dm("dm_rescae_vs_ipca")
    lines.append(f"  DM test (ResCAE vs IPCA):           stat={dm_stat},  p={p_val}  "
                 f"← PRIMARY RESULT")
    dm_stat_sec, p_val_sec = _dm("dm_rescae_vs_rescae_fixed")
    lines.append(f"  DM test (ResCAE vs ResCAE-Fixed):   stat={dm_stat_sec},  p={p_val_sec}  "
                 f"← SECONDARY RESULT")
    if rescae_row is not None:
        lines.append(f"  W_skip drift from IPCA init: {drift}")
    lines.append("")

    lines.append("Robustness: ResCAE vs CAE-NL (no IPCA init, no linear branch)")
    if nl_row is not None and pd.notna(nl_row.get("Pred_R2")):
        nl_sharpe_raw = nl_row.get("Sharpe", "N/A")
        try:
            nl_sharpe_v = float(nl_sharpe_raw)
            nl_sharpe_str = ("COLLAPSED (flat predictions — validates IPCA initialization)"
                             if (np.isnan(nl_sharpe_v) or nl_sharpe_v == 0.0)
                             else f"{nl_sharpe_v:.4f}")
        except (TypeError, ValueError):
            nl_sharpe_str = (nl_sharpe_raw if nl_sharpe_raw != "COLLAPSED"
                             else "COLLAPSED (flat predictions — validates IPCA initialization)")
        lines.append(f"  CAE-NL Pred R²: {nl_row['Pred_R2']}   Sharpe: {nl_sharpe_str}")
    vs_nl   = sig_results.get("vs_cae_nl", {}).get(best_k, {})
    r_vs_nl = vs_nl.get("dm_rescae_vs_cae_nl", {})
    dm_stat = _fv(r_vs_nl.get("dm_stat"), ".3f")
    p_val   = _fv(r_vs_nl.get("p_value"), ".4f")
    lines.append(f"  DM test (ResCAE vs CAE-NL):         stat={dm_stat},  p={p_val}")
    lines.append("=" * 66)

    return "\n".join(lines)


def plot_seed_stability(stability_df: pd.DataFrame,
                        figures_dir: str = "paper/figures") -> None:
    """four-panel boxplot of φ, drift, Pred R², and Sharpe across seeds. best config (highest mean Pred R²) highlighted in seagreen."""
    import os
    os.makedirs(figures_dir, exist_ok=True)

    if stability_df.empty:
        print("  plot_seed_stability: empty dataframe; skipping.")
        return

    df = stability_df.copy()
    df["label"] = df.apply(
        lambda r: f"K={int(r.K)} λL={r.lam_lin:.0e} λNL={r.lam_nonlin:.0e}",
        axis=1)

    mean_r2 = df.groupby("config_key")["pred_r2"].mean()
    best_cfg = mean_r2.idxmax() if not mean_r2.empty else None

    metrics = [
        ("phi",     "φ  (Nonlinear Fraction)"),
        ("drift",   "drift_rel"),
        ("pred_r2", "Predictive R²"),
        ("sharpe",  "Annualized Sharpe"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (metric, ylabel) in zip(axes, metrics):
        sub = df[["label", "config_key", metric]].dropna(subset=[metric])
        if sub.empty:
            ax.set_visible(False)
            continue

        labels_ord = (sub.groupby("label")["config_key"]
                      .first().reset_index()
                      .sort_values("label")["label"].tolist())
        data_by_label = [sub[sub["label"] == lbl][metric].values
                         for lbl in labels_ord]

        bp = ax.boxplot(data_by_label, patch_artist=True, medianprops={"color": "black", "lw": 1.5})

        for i, (patch, lbl) in enumerate(zip(bp["boxes"], labels_ord)):
            cfg = sub[sub["label"] == lbl]["config_key"].iloc[0]
            patch.set_facecolor("seagreen" if cfg == best_cfg else "steelblue")
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(labels_ord) + 1))
        ax.set_xticklabels(labels_ord, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Seed Stability: {ylabel}", fontsize=11)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Metric Distributions across {df['seed'].nunique()} Seeds  "
        f"(seagreen = best config by mean Pred R²)",
        fontsize=13)
    fig.tight_layout()
    _save_fig(fig, os.path.join(figures_dir, "seed_stability.png"))


def print_stability_summary(stability_df: pd.DataFrame) -> str:
    """return mean ± std table for the best config per K (by mean Pred R²)."""
    if stability_df.empty:
        return "[No seed stability data available]"

    df = stability_df.copy()
    best_per_k: dict = {}
    for k in sorted(df["K"].unique()):
        sub = df[df["K"] == k]
        mean_r2 = sub.groupby("config_key")["pred_r2"].mean()
        if not mean_r2.empty:
            best_per_k[int(k)] = mean_r2.idxmax()

    col_w = 18
    header = (f"{'Config':<14} "
              + "  ".join(f"{'Pred R²':<{col_w}} {'Sharpe':<{col_w}} "
                           f"{'φ':<{col_w}} {'Drift':<{col_w}}".split()))
    sep    = "─" * (14 + 4 * (col_w + 2))

    lines = ["\n--- Seed Stability (best config per K, mean ± std) ---",
             sep,
             f"{'Config':<14}  {'Pred R²':<{col_w}}  {'Sharpe':<{col_w}}"
             f"  {'φ':<{col_w}}  {'Drift':<{col_w}}",
             sep]

    for k, cfg_key in best_per_k.items():
        sub = df[df["config_key"] == cfg_key]
        row = df[df["config_key"] == cfg_key].iloc[0]
        label = f"K={k} best"

        def _ms(col):
            vals = sub[col].dropna()
            if vals.empty:
                return "N/A"
            return f"{vals.mean():.4f} ± {vals.std():.4f}"

        lines.append(
            f"{label:<14}  {_ms('pred_r2'):<{col_w}}  {_ms('sharpe'):<{col_w}}"
            f"  {_ms('phi'):<{col_w}}  {_ms('drift'):<{col_w}}"
        )

    lines.append(sep)
    return "\n".join(lines)


def plot_fixed_vs_free_comparison(splits: dict,
                                   models: dict,
                                   figures_dir: str = "paper/figures",
                                   results_dir: str = "results") -> None:
    """two-panel scatter of ResCAE-Fixed vs ResCAE per (K, λ_nonlin) config. left: Pred R²; right: Sharpe. points above diagonal mean ResCAE wins."""
    import os
    os.makedirs(figures_dir, exist_ok=True)

    cae_fixed = models.get("cae_fixed", {})
    cae_free  = models.get("cae", {})
    if not cae_fixed or not cae_free:
        print("  plot_fixed_vs_free_comparison: missing cae_fixed or cae; skipping.")
        return

    test_ret    = splits["test"]["returns"].values.astype(np.float32)
    test_chars  = splits["test"]["chars"].astype(np.float32)
    train_ret   = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)

    import pickle
    hparam_path = os.path.join(results_dir, "cae_hparam_search.pkl")
    free_scores: dict = {}
    if os.path.exists(hparam_path):
        with open(hparam_path, "rb") as f:
            free_scores = pickle.load(f).get("pred_val_mse") or {}

    records = []
    for fixed_key, fixed_model in cae_fixed.items():
        k_f, lam_nl_f = fixed_key
        rh_f = fixed_model.predict(test_ret, test_chars,
                                   train_returns=train_ret, train_chars=train_chars)
        pr2_f = predictive_r2(test_ret, rh_f.astype(np.float32))
        sr_f  = factor_sharpe(test_ret,  rh_f.astype(np.float32))

        # match to same (K, lam_nonlin) free ResCAE
        k_free_cfgs = {key: free_scores.get(key, float("inf"))
                       for key in cae_free if key[0] == k_f}
        if not k_free_cfgs:
            continue
        free_key  = min(k_free_cfgs, key=k_free_cfgs.get)
        free_model = cae_free[free_key]
        rh_free = free_model.predict(test_ret, test_chars,
                                     train_returns=train_ret, train_chars=train_chars)
        pr2_free = predictive_r2(test_ret, rh_free.astype(np.float32))
        sr_free  = factor_sharpe(test_ret,  rh_free.astype(np.float32))

        records.append({"K": k_f, "lam_nonlin": lam_nl_f,
                        "pr2_fixed": pr2_f,  "pr2_free": pr2_free,
                        "sr_fixed":  sr_f,   "sr_free":  sr_free})

    if not records:
        print("  plot_fixed_vs_free_comparison: no matching configs; skipping.")
        return

    k_vals  = sorted({r["K"] for r in records})
    cmap_k  = plt.cm.tab10
    k_color = {k: cmap_k(i / max(len(k_vals) - 1, 1)) for i, k in enumerate(k_vals)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, xkey, ykey, title in [
        (axes[0], "pr2_fixed", "pr2_free", "Predictive R²"),
        (axes[1], "sr_fixed",  "sr_free",  "Annualized Sharpe"),
    ]:
        xs = [r[xkey] for r in records]
        ys = [r[ykey] for r in records]
        cols = [k_color[r["K"]] for r in records]

        ax.scatter(xs, ys, c=cols, s=70, edgecolors="black", lw=0.4, zorder=3)
        all_vals = [v for v in xs + ys if np.isfinite(v)]
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            pad     = (hi - lo) * 0.05
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    color="black", lw=0.8, ls="--", zorder=2, label="Diagonal (equal)")
        ax.set_xlabel(f"ResCAE-Fixed  {title}", fontsize=10)
        ax.set_ylabel(f"ResCAE  {title}", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.tick_params(labelsize=9)

        above = sum(y > x for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y))
        total = sum(np.isfinite(x) and np.isfinite(y) for x, y in zip(xs, ys))
        ax.text(0.04, 0.96, f"ResCAE above diag: {above}/{total}",
                transform=ax.transAxes, fontsize=8, va="top")

    for k in k_vals:
        axes[0].scatter([], [], color=k_color[k], s=50,
                        edgecolors="black", lw=0.4, label=f"K={k}")
    axes[0].legend(fontsize=8, loc="lower right")

    fig.suptitle("ResCAE-Fixed vs ResCAE: Does Relaxing W_skip Help?", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, os.path.join(figures_dir, "fixed_vs_free_comparison.png"))


def run_evaluation(splits: dict,
                   models: dict,
                   figures_dir: str = "paper/figures",
                   results_dir: str = "results",
                   stability_df: Optional[pd.DataFrame] = None,
                   ) -> Tuple[pd.DataFrame, dict]:
    """end-to-end evaluation: metrics, figures, summary table. params: {splits, models, figures_dir, results_dir, stability_df}. returns (summary_df, sig_results)."""
    import os
    print("\n=== Evaluation ===")

    summary_df = build_summary_table(splits, models, results_dir=results_dir)

    sig_results = run_significance_tests(splits, models, figures_dir=figures_dir,
                                         results_dir=results_dir)
    plot_significance_results(sig_results, figures_dir=figures_dir)

    # attach DM p-values to ResCAE/ResCAE-Fixed rows; others get "-"
    summary_df["p_dm_vs_pca"]    = None
    summary_df["p_dm_vs_ipca"]   = None
    summary_df["p_dm_vs_cae_nl"] = None

    for (k, lam_lin, lam_nonlin), pc_res in sig_results.get("per_config", {}).items():
        mask = ((summary_df["Model"] == "ResCAE") &
                (summary_df["K"] == k) &
                (summary_df["λ_lin"]    == f"{lam_lin:.0e}") &
                (summary_df["λ_nonlin"] == f"{lam_nonlin:.0e}"))
        summary_df.loc[mask, "p_dm_vs_pca"]  = pc_res["dm_vs_pca"].get("p_value")
        if "dm_vs_ipca" in pc_res:
            summary_df.loc[mask, "p_dm_vs_ipca"] = pc_res["dm_vs_ipca"].get("p_value")

    for (k, lam_nonlin), pc_res in sig_results.get("per_config_fixed", {}).items():
        mask = ((summary_df["Model"] == "ResCAE-Fixed") &
                (summary_df["K"] == k) &
                (summary_df["λ_nonlin"] == f"{lam_nonlin:.0e}"))
        summary_df.loc[mask, "p_dm_vs_pca"]  = pc_res["dm_vs_pca"].get("p_value")
        if "dm_vs_ipca" in pc_res:
            summary_df.loc[mask, "p_dm_vs_ipca"] = pc_res["dm_vs_ipca"].get("p_value")

    for k, entry in sig_results.get("vs_cae_nl", {}).items():
        if "dm_rescae_vs_cae_nl" not in entry:
            continue
        mask = (summary_df["Model"] == "ResCAE") & (summary_df["K"] == k)
        summary_df.loc[mask, "p_dm_vs_cae_nl"] = (
            entry["dm_rescae_vs_cae_nl"].get("p_value"))

    non_sig = ~summary_df["Model"].isin(["ResCAE", "ResCAE-Fixed"])
    for col in ["p_dm_vs_pca", "p_dm_vs_ipca", "p_dm_vs_cae_nl"]:
        summary_df.loc[non_sig, col] = "-"

    os.makedirs(figures_dir, exist_ok=True)

    plot_residual_improvement(splits, models, sig_results, figures_dir=figures_dir)
    plot_factor_portfolios(splits, models, figures_dir=figures_dir,
                           results_dir=results_dir)
    plot_phi_r2_scatter(splits, models, sig_results, figures_dir=figures_dir)
    plot_fixed_vs_free_comparison(splits, models, figures_dir=figures_dir,
                                  results_dir=results_dir)
    plot_loss_curves(results_dir=results_dir, figures_dir=figures_dir)
    plot_summary_heatmap(summary_df, figures_dir=figures_dir)
    plot_ipca_drift(summary_df, figures_dir=figures_dir)
    plot_lambda_interaction(summary_df, figures_dir=figures_dir)
    phi_decomposition_analysis(summary_df, figures_dir=figures_dir)

    if stability_df is not None and not stability_df.empty:
        plot_seed_stability(stability_df, figures_dir=figures_dir)
        print(print_stability_summary(stability_df))

    csv_path = os.path.join(results_dir, "summary_table.csv")
    os.makedirs(results_dir, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Summary table → {csv_path}")
    print(f"  Figures saved to: {figures_dir}/")

    # sanity checks for known failure modes
    print("\n=== BUG FIX VALIDATION ===")
    cae_nl_rows = summary_df[summary_df["Model"] == "CAE-NL (robustness)"]
    bad_sharpe = cae_nl_rows[cae_nl_rows["Sharpe"] == "0.0000"]
    if len(bad_sharpe) > 0:
        print(f"  WARNING: {len(bad_sharpe)} CAE-NL rows still show Sharpe=0.0000 "
              f"(should show COLLAPSED or NaN)")
    else:
        print("  Bug 1 confirmed: no CAE-NL rows show Sharpe=0.0000")

    bad_phi = cae_nl_rows[cae_nl_rows["NL_frac"] == "1.000"]
    if len(bad_phi) > 0:
        print(f"  WARNING: {len(bad_phi)} CAE-NL rows still show NL_frac=1.000")
    else:
        print("  Bug 2 confirmed: no collapsed CAE-NL rows show NL_frac=1.000")

    pca_rows = summary_df[summary_df["Model"] == "PCA"]
    pca_r2_unique = pca_rows["Pred_R2"].nunique()
    if pca_r2_unique == 1:
        print("  WARNING: PCA Pred R² is identical for all K — investigate Bug 3")
    else:
        print(f"  Bug 3 confirmed: PCA Pred R² has {pca_r2_unique} unique values across K")

    rescae_rows = summary_df[summary_df["Model"] == "ResCAE"]
    high_drift = rescae_rows[
        rescae_rows["drift_rel"].astype(str).str.contains(r"\*", na=False)]
    if len(high_drift) > 0:
        print(f"  Bug 5: {len(high_drift)} ResCAE configs have drift > 1.0 (capped+flagged with *)")
    else:
        print("  Bug 5: no drift > 1.0 found")
    print("=== END VALIDATION ===\n")

    return summary_df, sig_results
