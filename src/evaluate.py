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


def factor_sharpe(r_true_test, r_hat_test, annualize=12):
    """Annualized Sharpe of the same monthly decile spread used in all plots/tests."""
    port = get_portfolio_returns(r_true_test, r_hat_test)
    port = port[np.isfinite(port)]
    if len(port) < 2 or port.std(ddof=1) < 1e-10:
        return np.nan
    return float(port.mean() / port.std(ddof=1) * np.sqrt(annualize))


def predictions_collapsed(r_true, r_hat):
    """Flat cross sections, distinct from insufficient observations for Sharpe."""
    spreads = [np.std(pred[valid]) for ret, pred in zip(r_true, r_hat)
               if (valid := _safe_mask(ret, pred)).sum() >= 2]
    return bool(spreads) and float(np.mean(spreads)) < 1e-6


def evaluation_splits(splits):
    """Use one characteristic-complete test panel for every model family."""
    result = dict(splits)
    test = dict(splits["test"])
    valid_chars = np.isfinite(test["chars"]).all(axis=2).T
    test["returns"] = test["returns"].where(valid_chars)
    result["test"] = test
    return result


def nonlinear_contribution(model: CAEModel,
                           returns: np.ndarray,
                           chars: np.ndarray) -> Tuple[float, float, float]:
    """decompose ResCAE loading variance into linear and nonlinear parts. params: {model: CAEModel, returns: (T,N), chars: (N,T,P)}. returns (var_lin, var_nonlin, frac_nonlin).

    φ = var_nonlin / (var_lin + var_nonlin). This branch variance ratio excludes
    covariance, and is not a share of explained returns or proof of nonlinearity.
    """
    var_lin, var_nonlin = model.get_nonlinear_contribution(returns, chars)

    total = var_lin + var_nonlin
    if not np.isfinite(total) or total <= 0:
        return var_lin, var_nonlin, np.nan
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


def _best_key_per_k(model_dict, pkl_path):
    """Select by saved validation scores; never silently pick an arbitrary config."""
    import os, pickle
    if not model_dict:
        return {}
    scores = {}
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        scores = data.get("pred_val_mse") or data.get("val_losses", {})
    best = {}
    for k in sorted({key[0] for key in model_dict}):
        keys = [key for key in model_dict if key[0] == k]
        available = [key for key in keys if np.isfinite(scores.get(key, np.nan))]
        if available:
            best[k] = min(available, key=scores.get)
        elif len(keys) == 1:
            best[k] = keys[0]
        else:
            raise ValueError(f"Missing validation scores for K={k}: {pkl_path}")
    return best


def build_summary_table(splits, models, results_dir="results", all_configs=False):
    """Test metrics on one common panel; keep numeric metrics at full precision."""
    import os
    splits = evaluation_splits(splits)
    test_ret = splits["test"]["returns"].values.astype(np.float32)
    test_chars = splits["test"]["chars"].astype(np.float32)
    train_ret = splits["train"]["returns"].values.astype(np.float32)
    train_chars = splits["train"]["chars"].astype(np.float32)
    rows = []
    labels = {"pca": "PCA", "ipca": "IPCA", "ae": "AE", "cae_fixed": "ResCAE-Fixed",
              "cae": "ResCAE", "cae_nl": "CAE-NL (robustness)"}
    for family, label in labels.items():
        configs = models.get(family, {})
        conditional = family in ("cae", "cae_fixed", "cae_nl")
        best = (_best_key_per_k(configs, os.path.join(results_dir, f"{family}_hparam_search.pkl"))
                if conditional else {})
        keys = configs if all_configs or not conditional else best.values()
        for key in keys:
            model = configs[key]
            k = key[0] if conditional else key
            if family in ("pca", "ae"):
                rec = model.reconstruct(test_ret)
                pred = model.predict(test_ret, train_returns=train_ret)
            else:
                rec = model.reconstruct(test_ret, test_chars)
                pred = model.predict(test_ret, test_chars, train_ret, train_chars)
            collapsed = predictions_collapsed(test_ret, pred)
            phi = nonlinear_contribution(model, test_ret, test_chars)[2] if conditional else np.nan
            rows.append({"Model": label, "K": k,
                         "λ_lin": key[1] if family == "cae" else ("frozen" if family == "cae_fixed" else np.nan),
                         "λ_nonlin": key[-1] if conditional else np.nan,
                         "Total_R2": total_r2(test_ret, rec), "Pred_R2": predictive_r2(test_ret, pred),
                         "Sharpe": factor_sharpe(test_ret, pred), "CSPE": oos_pricing_error(test_ret, pred),
                         "NL_frac": np.nan if collapsed else phi,
                         "drift_rel": compute_ipca_drift(model) if conditional else np.nan,
                         "Collapsed": collapsed, "N_predictions": int(_safe_mask(test_ret, pred).sum()),
                         "selected_by_validation": not conditional or key == best[k]})
    df = pd.DataFrame(rows)
    df["_order"] = df["Model"].map({name: i for i, name in enumerate(labels.values())})
    return df.sort_values(["K", "_order"]).drop(columns="_order").reset_index(drop=True)


def _save_fig(fig: plt.Figure, base_path: str, dpi: int = 200) -> None:
    """save figure as PNG and PDF."""
    import os
    png_path = base_path if base_path.endswith(".png") else base_path + ".png"
    pdf_path = png_path.replace(".png", ".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {png_path}")


def plot_loss_curves(results_dir="results", figures_dir="paper/figures"):
    """Overlay training/validation histories by family without one subplot per seed."""
    import json
    from pathlib import Path
    path = Path(results_dir) / "loss_histories.json"
    if not path.exists():
        print("  No saved loss histories; skipping loss curves.")
        return
    histories = json.loads(path.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, family in zip(axes.flat, ("ae", "cae", "cae_fixed", "cae_nl")):
        for i, (name, history) in enumerate((name, history) for name, history in histories.items()
                                           if name.startswith(family + " ")):
            ax.plot(history["train_losses"], color="steelblue", alpha=0.3,
                    label="Train (includes L1 for CAE)" if i == 0 else None)
            ax.plot(history["val_losses"], color="darkorange", alpha=0.3,
                    label="Validation reconstruction MSE" if i == 0 else None)
        ax.set(title=family, xlabel="Epoch", ylabel="Loss (orthogonality step excluded)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, str(Path(figures_dir) / "loss_curves.png"))


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
                               values="Sharpe", aggfunc="mean").reindex(index=pivot_r2.index, columns=pivot_r2.columns)
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
        port = get_portfolio_returns(r_true, r_hat)
        if not np.isfinite(port).any():
            return
        cumulative = np.nancumsum(port)
        cumulative[~np.isfinite(port)] = np.nan
        ax.plot(test_dates, cumulative, label=label, color=color, lw=lw, linestyle=ls)

    # Compare every family at the globally validation-selected ResCAE K.
    k_vals = sorted(models["pca"].keys())
    k_rep = best_cae_key[0] if best_cae_key is not None else k_vals[len(k_vals) // 2]
    if models.get("cae_nl"):
        best_nl_key = _best_key_per_k(models["cae_nl"], nl_hparam_path).get(k_rep)
    if models.get("cae_fixed"):
        best_fixed_key = _best_key_per_k(models["cae_fixed"], fixed_hparam_path).get(k_rep)

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

    if k_rep in models.get("ae", {}):
        _port_ts(test_ret, models["ae"][k_rep].predict(test_ret, train_returns=train_ret),
                 f"AE K={k_rep}", "firebrick")

    bnh = np.nanmean(test_ret, axis=1)
    ax.plot(test_dates, np.nancumsum(bnh),
            label="Monthly equal-weight excess return", color="black", lw=1.0, linestyle=":")
    ax.axhline(0, color="black", lw=0.5)

    ax.set_title(f"Sum of Monthly Decile Spreads — {test_dates[0]:%Y-%m} to {test_dates[-1]:%Y-%m}",
                 fontsize=13)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Arithmetic sum of monthly returns (not compounded)", fontsize=11)
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
        improvement = (sig_results.get("per_config", {}).get(cae_key, {})
                       .get("dm_vs_ipca", {}).get("mean_loss_diff", np.nan))
        drift = compute_ipca_drift(model)
        points.append({"phi": phi, "pred_r2": pr2, "k": k,
                       "lam_nonlin": lam_nonlin, "p_vs_ipca": p_ipca,
                       "improvement": improvement,
                       "high_drift": (drift is not None and drift > 1.0)})

    if not points:
        print("  No ResCAE configs for φ–R² scatter; skipping.")
        return

    # ResCAE-Fixed keeps the linear weights frozen; encoder and g still vary.
    fixed_points = []
    for fixed_key, model in models.get("cae_fixed", {}).items():
        k_f, lam_nl_f = fixed_key
        rh_f = model.predict(test_ret, test_chars,
                             train_returns=train_ret, train_chars=train_chars)
        pr2_f = predictive_r2(test_ret, rh_f.astype(np.float32))
        _, _, phi_f = nonlinear_contribution(model, test_ret, test_chars)
        fixed_points.append({"phi": phi_f, "pred_r2": pr2_f,
                              "k": k_f, "lam_nonlin": lam_nl_f})

    def _col(p, improvement):
        if not np.isfinite(p): return "lightgray"
        if p < 0.05 and improvement > 0: return "seagreen"
        if p < 0.10 and improvement > 0: return "gold"
        return "firebrick"

    fig, ax = plt.subplots(figsize=(7, 5))
    for pt in points:
        marker = "X" if pt["high_drift"] else "o"
        ax.scatter(pt["pred_r2"], pt["phi"],
                   color=_col(pt["p_vs_ipca"], pt["improvement"]),
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
    ax.annotate(f"Highest test R² (descriptive): K={best['k']}, λNL={best['lam_nonlin']:.0e}",
                (best["pred_r2"], best["phi"]),
                fontsize=8, fontweight="bold",
                textcoords="offset points", xytext=(10, -12),
                arrowprops=dict(arrowstyle="->", lw=0.8))

    from matplotlib.lines import Line2D as _Line2D
    legend_els = [
        Patch(facecolor="seagreen",     edgecolor="black", label="ResCAE improves on IPCA, p < 0.05"),
        Patch(facecolor="gold",         edgecolor="black", label="ResCAE improves, 0.05 ≤ p < 0.10"),
        Patch(facecolor="firebrick",    edgecolor="black", label="No significant improvement at 10%"),
        Patch(facecolor="lightgray",    edgecolor="black", label="IPCA not available"),
        _Line2D([0], [0], marker="D", color="w", markerfacecolor="mediumpurple",
                markeredgecolor="black", markersize=8,
                label="ResCAE-Fixed (frozen linear weights)"),
        _Line2D([0], [0], marker="X", color="w", markerfacecolor="gray",
                markeredgecolor="black", markersize=8,
                label="× = relative weight drift > 1.0"),
    ]
    ax.legend(handles=legend_els, fontsize=9)
    ax.set_xlabel("Predictive R²", fontsize=11)
    ax.set_ylabel("φ  (Nonlinear Fraction)", fontsize=11)
    ax.set_title("Nonlinear Contribution (φ) vs. Predictive R²", fontsize=13)
    ax.tick_params(labelsize=9)

    fig.text(0.5, -0.06,
             "Each point is one configuration. φ is a branch variance ratio, excluding "
             "linear/nonlinear covariance; it is not a fraction of explained returns.\n"
             "Color indicates statistical significance of ResCAE vs IPCA (DM test).",
             ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    _save_fig(fig, os.path.join(figures_dir, "phi_decomposition.png"))


def plot_ipca_drift(summary_df: pd.DataFrame,
                    figures_dir: str = "paper/figures") -> None:
    """bar chart of relative W_skip drift from IPCA init per ResCAE config (figure 5). Drift is a parameter diagnostic, not an identification test."""
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
    # Relative parameter movement has no universal interpretation threshold.
    y_hi = max(max(drifts, default=1.2), 1.2) * 1.05
    ax.set_ylim(0, y_hi)

    ax.set_ylabel("‖W_skip − Γ_init‖_F / ‖Γ_init‖_F", fontsize=11)
    ax.set_title(
        "W_skip Drift from IPCA Initialization —\n"
        "How Much Does Joint Optimization Move the Linear Component?",
        fontsize=13)
    ax.set_xlabel("ResCAE configuration", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)

    fig.text(0.5, -0.04,
             "Drift measures movement in linear weights relative to their initial norm.\n"
             "It does not identify a unique linear/nonlinear decomposition.",
             ha="center", fontsize=8, style="italic")

    plt.xticks(rotation=30, ha="right", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, os.path.join(figures_dir, "ipca_drift.png"))


def get_portfolio_returns(r_true, r_hat):
    """Equal-weight top-minus-bottom predicted deciles; no portfolio for flat signals.

    Percentile boundaries include ties, but legs must be disjoint. Each month
    requires at least 20 observed stocks. Costs and financing are excluded.
    """
    port = np.full(len(r_true), np.nan)
    if predictions_collapsed(r_true, r_hat):
        return port
    for t, (ret, pred) in enumerate(zip(r_true, r_hat)):
        valid = _safe_mask(ret, pred)
        if valid.sum() < 20:
            continue
        pv, rv = pred[valid], ret[valid]
        if pv.std() < 1e-6:
            continue
        q10, q90 = np.percentile(pv, [10, 90])
        if q90 <= q10:
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
    max_lag = min(max(int(max_lag), 0), T_eff - 1)

    d_dev   = d_clean - d_bar
    gamma0  = float((d_dev ** 2).mean())
    hac_var = gamma0
    for h in range(1, max_lag + 1):
        gamma_h = float((d_dev[h:] * d_dev[:-h]).sum() / T_eff)
        hac_var += 2.0 * (1.0 - h / (max_lag + 1)) * gamma_h

    if hac_var < 0:
        print(f"    WARNING: Newey-West variance non-positive ({hac_var:.2e}); "
              f"falling back to sample variance.")
        hac_var = gamma0

    if hac_var == 0:
        dm_stat = 0.0 if d_bar == 0 else np.nan
        p_value = 1.0 if d_bar == 0 else np.nan
    else:
        dm_stat = d_bar / np.sqrt(hac_var / T_eff)
        p_value = 2.0 * float(_norm.sf(abs(dm_stat)))

    return {
        "dm_stat":        float(dm_stat),
        "p_value":        float(p_value),
        "mean_loss_diff": float(d_bar),
        "hac_variance":   float(hac_var),
        "n_lags":         max_lag,
    }


def _bootstrap_months(rng, n_months, block_length=6):
    """Circular blocks preserve short-run monthly dependence."""
    if n_months < 1 or block_length < 1:
        raise ValueError("Bootstrap requires observations and a positive block length.")
    starts = rng.integers(0, n_months, size=int(np.ceil(n_months / block_length)))
    return ((starts[:, None] + np.arange(block_length)) % n_months).ravel()[:n_months]


def bootstrap_r2_ci(r_true: np.ndarray,
                     r_hat: np.ndarray,
                     n_bootstrap: int = 1000,
                     seed: int = 42,
                     block_length: int = 6,
                     ) -> dict:
    """bootstrap 95% CI for predictive R² by resampling the time dimension. params: {r_true: (T,N), r_hat: (T,N), n_bootstrap: int, seed: int}. returns dict with r2_observed, ci_lower, ci_upper, bootstrap_distribution."""
    T      = r_true.shape[0]
    r2_obs = float(predictive_r2(r_true, r_hat))
    rng    = np.random.default_rng(seed)
    boot_r2 = np.full(n_bootstrap, np.nan)

    for b in range(n_bootstrap):
        idx        = _bootstrap_months(rng, T, block_length)
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
                           block_length: int = 6,
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
    if T < 30:
        return nan_result

    sr1       = _sr(r1)
    sr2       = _sr(r2)
    delta_obs = sr1 - sr2 if (np.isfinite(sr1) and np.isfinite(sr2)) else np.nan

    rng        = np.random.default_rng(seed)
    boot_delta = np.full(n_bootstrap, np.nan)
    for b in range(n_bootstrap):
        idx           = _bootstrap_months(rng, T, block_length)
        boot_delta[b] = _sr(r1[idx]) - _sr(r2[idx])

    bv = boot_delta[np.isfinite(boot_delta)]
    if len(bv) == 0 or not np.isfinite(delta_obs):
        return {**nan_result, "sharpe_1": sr1, "sharpe_2": sr2,
                "sharpe_diff": delta_obs, "bootstrap_distribution": boot_delta}

    centered = bv - delta_obs
    p_value = float((1 + (np.abs(centered) >= np.abs(delta_obs)).sum()) / (len(bv) + 1))

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
    """Describe φ under equal versus unequal penalty weights; this is not a controlled parameter-count experiment."""
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
        return _best_key_per_k(models["cae"], hparam_path).get(k)

    # resolve per-K best ResCAE-Fixed
    fixed_hparam_path = os.path.join(results_dir, "cae_fixed_hparam_search.pkl")
    fixed_scores: dict = {}
    if os.path.exists(fixed_hparam_path):
        with open(fixed_hparam_path, "rb") as _f:
            _fd = pickle.load(_f)
        fixed_scores = _fd.get("pred_val_mse") or _fd.get("val_losses", {})

    def _best_fixed_for_k(k: int):
        return _best_key_per_k(models.get("cae_fixed", {}), fixed_hparam_path).get(k)

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
    test_dates = splits["test"]["returns"].index
    sig_results: dict = {
        "test_period": f"{test_dates[0]:%Y-%m} to {test_dates[-1]:%Y-%m} ({len(test_dates)} months)",
        "bootstrap": {"method": "circular blocks", "block_length": 6, "samples": n_bootstrap},
        "dm_convention": "positive favors first named model",
    }

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
    avail = {k: v for k, v in scores_by_config.items() if k in models["cae"] and np.isfinite(v)}
    if not avail and len(models["cae"]) > 1:
        raise ValueError("Validation scores are required to choose an overall ResCAE configuration.")
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
            entry_vs["dm_pca_vs_cae_nl"] = diebold_mariano_test(err_nl, err_pca)

        rescae_key = _best_rescae_for_k(k)
        if rescae_key is not None:
            rh_rescae, err_rescae = _pred("rescae", k, rescae_key)
            entry_vs["dm_rescae_vs_cae_nl"] = diebold_mariano_test(err_nl, err_rescae)
            entry_vs["boot_sharpe_rescae_vs_cae_nl"] = bootstrap_sharpe_test(
                get_portfolio_returns(test_ret, rh_rescae),
                get_portfolio_returns(test_ret, rh_nl),
                n_bootstrap)

        vs_cae_nl[k] = entry_vs
    sig_results["vs_cae_nl"] = vs_cae_nl

    # print DM table (primary: ResCAE vs IPCA first)
    print("\n--- Diebold-Mariano Results  (positive DM = first named model better) ---")
    print("  PRIMARY: ResCAE vs IPCA — does the residual architecture improve forecasts?")
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
        print("\n  FIXED PRIMARY: ResCAE-Fixed vs IPCA — does the frozen-linear architecture improve forecasts?")
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

    # Holm correction within each comparison across the prespecified K grid.
    # All-grid per-config tests remain exploratory and unadjusted.
    comparison_keys = {key for k in k_values for key in sig_results.get(k, {})
                       if key.startswith("dm_")}
    for comparison in comparison_keys:
        entries = [sig_results[k][comparison] for k in k_values
                   if comparison in sig_results.get(k, {})
                   and np.isfinite(sig_results[k][comparison].get("p_value", np.nan))]
        entries.sort(key=lambda entry: entry["p_value"])
        adjusted = 0.0
        for rank, entry in enumerate(entries):
            adjusted = max(adjusted, min(1.0, (len(entries) - rank) * entry["p_value"]))
            entry["p_value_holm"] = adjusted
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
        ax.set_xlabel("DM statistic  (positive = first named model better)", fontsize=11)
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

    best_key = sig_results.get("overall", {}).get("rescae_key")
    best_k = best_key[0] if best_key is not None else None

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
            # Percentile intervals need not contain the point estimate.
            ax.plot([i, i], [lo, hi], color="black", lw=1.2)
            ax.plot([i - 0.05, i + 0.05], [lo, lo], color="black", lw=1.2)
            ax.plot([i - 0.05, i + 0.05], [hi, hi], color="black", lw=1.2)

    # significance brackets
    hi_finite = [h for h in r2_hi if np.isfinite(h)]
    span = max(np.ptp(r2_obs), 0.001)
    y_top = (max(hi_finite) if hi_finite else max(r2_obs)) + span * 0.08
    brk_h = span * 0.04

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


def format_paper_story(summary_df, sig_results):
    """Report the validation-selected model, without picking K on the test set."""
    best = sig_results.get("overall", {}).get("rescae_key")
    if best is None:
        return "No validation-selected ResCAE configuration available. See the per-K summary."
    k = best[0]
    rows = summary_df[summary_df["K"] == k]
    period = sig_results.get("test_period", "held-out period")
    lines = [f"RESULTS — {period}", f"Representative ResCAE chosen on validation data: {best}",
             rows[["Model", "K", "Pred_R2", "Sharpe", "NL_frac", "drift_rel"]].to_string(index=False),
             "", "DM tests: positive statistic favors the first named model; two-sided p-values."]
    for label, key in [("ResCAE vs IPCA", "dm_rescae_vs_ipca"),
                       ("ResCAE vs PCA", "dm_rescae_vs_pca"),
                       ("ResCAE-Fixed vs IPCA", "dm_rescae_fixed_vs_ipca"),
                       ("ResCAE vs ResCAE-Fixed", "dm_rescae_vs_rescae_fixed")]:
        result = sig_results.get(k, {}).get(key, {})
        lines.append(f"  {label}: DM={result.get('dm_stat', np.nan):.3f}, "
                     f"p={result.get('p_value', np.nan):.4f}, "
                     f"Holm p across K={result.get('p_value_holm', np.nan):.4f}")
    lines.extend(["", "Predictive R² and portfolio Sharpe measure different objectives.",
                  "ResCAE-Fixed freezes W_skip; its encoder and nonlinear branch both train.",
                  "The comparison does not isolate nonlinearity with fixed factors.",
                  "φ is a branch variance ratio excluding covariance; it is not explained return variance.",
                  "Flat predictions provide no stock ranking; this does not validate an architecture."])
    return "\n".join(lines)


def plot_seed_stability(stability_df: pd.DataFrame,
                        figures_dir: str = "paper/figures") -> None:
    """Four-panel distribution of test metrics across seeds at validation-selected configurations."""
    import os
    os.makedirs(figures_dir, exist_ok=True)

    if stability_df.empty:
        print("  plot_seed_stability: empty dataframe; skipping.")
        return

    df = stability_df.copy()
    if "selected_by_validation" in df:
        df = df[df["selected_by_validation"]].copy()
    df["label"] = df.apply(
        lambda r: f"{r.get('Model', 'ResCAE')} K={int(r.K)}",
        axis=1)

    best_cfg = None

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
        f"(validation-selected configurations)",
        fontsize=13)
    fig.tight_layout()
    _save_fig(fig, os.path.join(figures_dir, "seed_stability.png"))


def print_stability_summary(stability_df):
    """Summarize seed variation within each validation-selected configuration."""
    if stability_df.empty:
        return "No multi-seed stability results available."
    if "selected_by_validation" not in stability_df:
        return "Legacy stability file has no validation selection markers; see the historical audit."
    selected = stability_df[stability_df["selected_by_validation"]]
    table = selected.groupby(["Model", "K"])[["pred_r2", "sharpe", "phi", "drift"]].agg(["mean", "std"])
    return "Seed stability at validation-selected configurations (test metrics):\n" + table.to_string()


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
                       for key in cae_free if key[0] == k_f and key[-1] == lam_nl_f}
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


def run_evaluation(splits, models, figures_dir="paper/figures", results_dir="results",
                   stability_df=None, n_bootstrap=1000):
    """Evaluate and save numeric summaries, all-grid diagnostics, and figures."""
    import os
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    splits = evaluation_splits(splits)
    print("\n=== Evaluation on the common characteristic-complete test panel ===")
    all_metrics = build_summary_table(splits, models, results_dir, all_configs=True)
    summary = all_metrics[all_metrics["selected_by_validation"]].copy()
    sig = run_significance_tests(splits, models, figures_dir=figures_dir,
                                 results_dir=results_dir, n_bootstrap=n_bootstrap)
    for column in ("p_dm_vs_pca", "p_dm_vs_ipca", "p_dm_vs_cae_nl", "p_dm_vs_ipca_holm"):
        summary[column] = np.nan
    for idx, row in summary.iterrows():
        k = int(row["K"])
        prefix = {"ResCAE": "rescae", "ResCAE-Fixed": "rescae_fixed"}.get(row["Model"])
        if prefix is None:
            continue
        for baseline in ("pca", "ipca"):
            result = sig.get(k, {}).get(f"dm_{prefix}_vs_{baseline}", {})
            summary.loc[idx, f"p_dm_vs_{baseline}"] = result.get("p_value", np.nan)
            if baseline == "ipca":
                summary.loc[idx, "p_dm_vs_ipca_holm"] = result.get("p_value_holm", np.nan)
        if prefix == "rescae":
            summary.loc[idx, "p_dm_vs_cae_nl"] = sig.get("vs_cae_nl", {}).get(k, {}).get(
                "dm_rescae_vs_cae_nl", {}).get("p_value", np.nan)
    # Save tabular evidence before plotting, so a plotting failure cannot lose results.
    summary.to_csv(os.path.join(results_dir, "summary_table.csv"), index=False)
    all_metrics.to_csv(os.path.join(results_dir, "all_config_metrics.csv"), index=False)
    plot_significance_results(sig, figures_dir)
    plot_residual_improvement(splits, models, sig, figures_dir)
    plot_factor_portfolios(splits, models, figures_dir, results_dir)
    plot_phi_r2_scatter(splits, models, sig, figures_dir)
    plot_fixed_vs_free_comparison(splits, models, figures_dir, results_dir)
    plot_loss_curves(results_dir, figures_dir)
    plot_summary_heatmap(summary, figures_dir)
    plot_ipca_drift(summary, figures_dir)
    plot_lambda_interaction(all_metrics, figures_dir)
    phi_decomposition_analysis(all_metrics, figures_dir)
    if stability_df is not None and not stability_df.empty:
        plot_seed_stability(stability_df, figures_dir)
        print(print_stability_summary(stability_df))
    print(f"\nSummary: {results_dir}/summary_table.csv")
    return summary, sig
