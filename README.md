# Latent Factor Models

A reproducible comparison of PCA, instrumented PCA (IPCA), a plain autoencoder, and three conditional autoencoders for monthly stock returns. The project studies whether a flexible correction to characteristic-based factor loadings improves held-out forecasts.

The complete experiment was run in September 2026 using newly downloaded data, **365 stocks**, **19 lagged price/volume features**, and five seeds for each conditional architecture. The test period is **January 2015–December 2024**. Results, model selection, and the limitations of the experiment are documented below and in the [research report](docs/report.md).

## Results

Hyperparameters were selected on 2010–2014 validation data. The overall selected ResCAE is **K = 8, λ_linear = 0.1, λ_nonlinear = 5e-05**. The table compares all families at this same factor dimension, using each conditional family's validation-selected penalties.

| Model | Reconstruction R² | Predictive R² | Sharpe | Pricing error |
| :--- | :--- | :--- | :--- | :--- |
| PCA | 38.75% | 1.52% | 0.6081 | 0.0075 |
| IPCA | 12.31% | -0.04% | 0.8291 | 0.0130 |
| AE | 32.48% | -0.33% | -0.2646 | 0.0136 |
| ResCAE-Fixed | 21.73% | 0.90% | 0.7643 | 0.0108 |
| ResCAE | 18.62% | 1.73% | 0.7103 | 0.0077 |
| CAE-NL | 14.34% | 1.43% | Undefined | 0.0079 |

At the validation-selected ResCAE dimension **K = 8**, ResCAE's predictive R² is **1.73%**, versus **-0.04%** for IPCA (difference: +1.77 percentage points). The two-sided DM p-value is **0.166**, or **0.832** after Holm adjustment across K. The portfolio Sharpe difference is **-0.119**, with block-bootstrap p = **0.461**. Forecast accuracy and portfolio ranking measure different objectives. This selected comparison does not establish lower ResCAE forecast error at the adjusted 5% level. The Sharpe difference is not significant at 5%.

At K = 8, ResCAE has **φ = 1e-07** and relative linear-weight drift **0.841**. Its nonlinear branch contributes negligible loading variation; constant effects are not measured by φ. **5 of 5** validation-selected CAE-NL configurations produce flat cross-sectional forecasts, so their portfolio Sharpe is undefined. Positive forecast R² for these flat signals does not demonstrate stock-ranking ability. This diagnostic alone cannot identify the source of a forecast gain.

A simple control predicts the pooled training mean, **1.165% per month**, for every stock and month. Its test predictive R² is **1.67%**; ResCAE differs by **+0.06 percentage points**. This control has no stock ranking and no portfolio Sharpe. It shows why beating the zero-return R² benchmark alone is insufficient evidence of useful conditional prediction. [Control calculation](results/full-2026-09-22-clean/constant_mean_benchmark.json).

![Forecast accuracy and portfolio Sharpe](docs/figures/results_overview.png)

Source: [full-precision summary](results/full-2026-09-22-clean/summary_table.csv), [run configuration](results/full-2026-09-22-clean/config.json), and [report with statistical comparisons](docs/report.md). The full K grid is reported; the highest test score is not used to choose the headline model. ResCAE-Fixed, ResCAE, and CAE-NL each average five separately trained models. PCA, IPCA, and AE are single fits per K; AE uses seed 42.

This is an exploratory study of a survivorship-biased stock universe. It does not establish that nonlinear effects are uniquely identified, or that the portfolios would remain profitable after costs.

## Models

| Model | Structure and forecast |
| :--- | :--- |
| PCA | Static principal-component directions; training mean returns projected into the K-component space |
| IPCA | Characteristic-driven loadings `β = Γz`, estimated by alternating least squares; forecast uses the mean training factor |
| AE | Full-return-vector encoder `N → 128 → 64 → K` and mirrored decoder; forecast decodes the mean training encoding |
| ResCAE | Loading map `β = W_skip z + g(z)`; linear branch initialized from IPCA, both branches learned |
| ResCAE-Fixed | Same loading map with `W_skip` frozen at IPCA weights; encoder and `g` are still learned |
| CAE-NL | Loading map `β = g(z)`, with no linear skip branch |

The conditional encoder maps characteristic-managed returns, `Z[t]ᵀr[t]/N[t]`, through `P → 32 → K`. The loading network `g` uses `P → 32 → 32 → K`. Hidden layers use ReLU activations.

**Reconstruction uses realized returns; forecasting does not.** All forecasts use training-period factor means. Conditional models combine those means with characteristics computed through the preceding month. Parameters and factor means remain fixed during validation and testing. PCA and AE produce time-constant forecasts in this protocol.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The full run used Python 3.14.3, NumPy 2.4.4, pandas 3.0.3, SciPy 1.17.1, and PyTorch 2.11.0 on CPU. Minimum dependency bounds are in `requirements.txt`; actual run versions are saved in `config.json`.

## Run

Check the pipeline without downloading data:

```bash
python main.py --smoke-test
python -m unittest discover -s tests -v
```

A smoke test is short synthetic training for software verification. To reproduce the full protocol:

```bash
python main.py --multi-seed --seeds 10 30 45 99 2048 --device cpu
```

Fresh empirical runs download current S&P 500 constituents from Wikipedia, adjusted prices/volume from Yahoo Finance through `yfinance`, and [TB3MS from FRED](https://fred.stlouisfed.org/series/TB3MS). No API key is configured. Network access/source availability are required; raw data are cached locally and excluded from Git.

The grid is K = `2 3 5 8 10`, λ_linear = `0.001 0.005 0.01 0.05 0.1`, and λ_nonlinear = `0.000001 0.00001 0.00005 0.0001 0.0005`. This means 125 ResCAE, 25 fixed, and 25 nonlinear-only configurations: **875 conditional fits** with five seeds, plus five AE fits and the PCA/IPCA baselines. Allow substantial runtime for the full search.

For a smaller empirical experiment:

```bash
python main.py --k 2 --lambda-lin 0.001 --lambda-nonlin 0.00001 \
  --max-epochs 20 --patience 5 --bootstrap-samples 100
```

Training uses Adam at learning rate 0.001, batches of 32 months, gradient clipping at norm 1, maximum 400 epochs, and early-stopping patience 25. Epoch selection uses validation reconstruction MSE; configuration selection uses **mean individual-seed validation predictive MSE**. Ensemble predictions are averaged after selection; the selection criterion is not ensemble validation MSE. There is no train-plus-validation refit.

Conditional training minimizes observed-return reconstruction MSE plus `λ_linear × sum(abs(W_skip)) + λ_nonlinear × sum(abs(parameters of g))`. The nonlinear penalty includes biases. The encoder is not subject to this L1 penalty. Positive penalties throughout the tested grid can suppress the flexible branch; a collapsed model does not rule out useful nonlinear forecasts under other specifications.

A separate full-sample update applies the `||FᵀF/T − I||²` penalty with weight 0.001. This encourages scale and orthogonality but does not identify rotations. IPCA fits and static training inputs are reused to reduce repeated work.

`--device cuda` and `--device mps` are available when supported; CPU is the verified path. `--threads` defaults to 1. `--run-dir` selects a new output directory, and `--figures-dir` overrides its plot directory. Existing directories cannot be overwritten by a new training run.

Re-evaluate a completed trusted checkpoint using the same cache:

```bash
python main.py --skip-train --run-dir results/full-2026-09-22-clean \
  --cache data/raw/data_cache_v4.pkl
```

Model checkpoints and data caches are local and ignored by Git; a clone needs a training run before `--skip-train` is usable. Checkpoint loading verifies the data fingerprint. Re-evaluation replaces that run's tables/plots while preserving its training log/configuration. Only load pickle/PyTorch files you trust. Run `python main.py --help` for all options.

## Dataset

| Split | Dates | Months | Observed stock-months |
| :--- | :--- | ---: | ---: |
| Training | 1985-02-28 to 2009-12-31 | 299 | 83,296 |
| Validation | 2010-01-31 to 2014-12-31 | 60 | 21,900 |
| Test | 2015-01-31 to 2024-12-31 | 120 | 43,800 |

Prices are requested from 1980 through 2024. Long-window features delay the first eligible month until 1985. Stocks require at least 60 characteristic-complete **training** observations. Every model uses the same eligibility mask in all splits. Test coverage is never used to select stocks or tune hyperparameters.

Quotes with zero or missing trading volume are excluded before resampling and bounded forward filling. NVR's price history begins on October 1, 1993: its [reorganization](https://www.sec.gov/Archives/edgar/data/906163/000095012310100067/w79957e10vq.htm) changed the underlying security, so predecessor partnership prices are not joined to corporate shares. These rules apply before return and feature calculation; no return winsorization or test-score-based trimming is applied. Coverage and return-distribution checks are saved with the run.

The target is monthly adjusted-price return minus an approximate monthly risk-free rate. [FRED TB3MS](https://fred.stlouisfed.org/series/TB3MS) is a monthly average of discount-basis yields; this project approximates a monthly rate using `(1 + TB3MS/100)^(1/12) − 1`. It is not a realized Treasury holding-period return. Monthly price gaps are filled for at most three months.

Features are ranked cross-sectionally to `[-1, 1]`:

| Group | Features |
| :--- | :--- |
| Momentum | `mom1`, `mom3`, `mom6`, `mom9`, `mom12`, `mom24_13` |
| Volatility | `vol1`, `vol6`, `vol12`, `ivol12` |
| Market exposure | `beta12`, `beta36`, `beta_down` |
| Price level relative to history | `high52` |
| Return shape | `max_ret`, `min_ret`, `skew1`, `skew3` |
| Liquidity proxy | `amihud` |

`mom1` uses the prior month. Shorter multi-month momentum windows skip that month; `mom24_13` compounds `returns[t−37:t−13]`. `vol1` scales daily volatility by `sqrt(21)` to monthly units. Missing compounding windows remain missing. Market regressions use paired finite stock/market excess returns and an intercept. Shares-dependent features are excluded because reliable historical shares are unavailable.

## Metrics and inference

- **Reconstruction/predictive R²:** `1 − Σ(r − r_hat)² / Σr²`, relative to zero excess returns. A stored `0.01` means 1%. Reconstruction and forecasts use different inputs.
- **Sharpe:** `sqrt(12) × mean(monthly spread) / sample_std(monthly spread)`. Portfolios are equal-weight top predicted decile minus bottom decile, with at least 20 eligible stocks. Flat signals and overlapping tied legs produce no portfolio. Long and short notionals are one unit each.
- **CSPE:** root mean squared per-stock mean forecast residual, in monthly decimal return units.
- **φ / NL_frac:** nonlinear loading variance divided by the sum of linear and nonlinear loading variances, averaged within factor coordinates. It excludes cross-branch covariance and is not a share of explained return variance. Ensemble component variances are averaged before division.
- **drift_rel:** relative Frobenius movement of `W_skip` from its IPCA initialization; averaged across seeds and zero for frozen weights. Values above 1 are retained.

DM statistics are positive when the **first named model** has lower squared forecast error. Tests use two-sided normal p-values with a Bartlett/Newey–West long-run variance estimate. Selected comparisons include Holm adjustment across the K grid within each comparison family. Full-grid tests remain exploratory. Bootstrap inference uses 1,000 draws of six-month circular blocks. Portfolio plots show arithmetic sums of monthly spreads, not compounded wealth.

The current-constituent universe excludes delisted firms and introduces survivorship bias. Adjusted-price/dollar-volume features are proxies. No turnover, trading-cost, financing, or short-borrow model is included. Freezing a loading branch does not freeze factors: the encoder and `g` can change, and `g` can also represent linear/constant effects. These limits constrain interpretation even when a statistical comparison is significant.

## Files

```text
main.py                       CLI, run metadata, model checkpoints
src/data.py                   Downloads, 19 lagged features, common panel
src/models.py                 PCA/IPCA/AE/conditional autoencoders
src/train.py                  Training and validation-based selection
src/ensemble.py               Prediction averaging and seed diagnostics
src/evaluate.py               Metrics, inference, full-grid plots
scripts/build_report.py        Generate documentation from a completed run
scripts/verify_run.py          Check grids, selection, and checkpoint metrics
scripts/build_paper.py         Optional local LaTeX/PDF export of the report
notebooks/exploration.ipynb    Data and run exploration (Jupyter optional)
tests/test_regressions.py      Offline regression tests
docs/report.md                Detailed current experiment report
results/full-2026-09-22-clean/        Full empirical outputs
```

A run includes `config.json`, `evaluation_config.json`, `summary_table.csv`, `all_config_metrics.csv`, validation-search pickles, `seed_stability.csv`, `significance_tests.pkl`, `loss_histories.json`, `results_summary.txt`, `data_manifest.json`, `environment.json`, execution logs, and PNG/PDF plots. Verification adds `verification.json`, `data_quality.csv`, and `constant_mean_benchmark.json`. `models/checkpoint.pt` stores all fitted models locally. Data, model bundles, smoke-test outputs, and bytecode are excluded from Git.

Verify saved selections and metrics, then regenerate this README and the report:

```bash
python scripts/verify_run.py --run-dir results/full-2026-09-22-clean
python scripts/build_report.py --run-dir results/full-2026-09-22-clean
```

Optional: `python scripts/build_paper.py` exports the report to `paper/paper.tex` and `paper/paper.pdf`. This requires Pandoc, XeLaTeX, and the Times New Roman, Arial, and Menlo fonts; those tools are not required for training or the Markdown report. The generated paper directory is excluded from Git.

## Research background

The characteristic-based loading model follows the IPCA framework of [Kelly, Pruitt, and Su, *Characteristics Are Covariances*](https://www.nber.org/system/files/working_papers/w24540/w24540.pdf). Conditional autoencoders are motivated by [Gu, Kelly, and Xiu, *Autoencoder Asset Pricing Models*](https://www.sciencedirect.com/science/article/pii/S0304407620301998). This project uses its own dataset, residual architecture, and evaluation protocol; it is not a reproduction of either paper's empirical results.
