# Project and results audit — September 2026

The repository's original empirical output is preserved in [`results/05-14-2026-1407`](../results/05-14-2026-1407/). The audit read the source, notebook, local research drafts, cached panel, validation-search files, summary, seed diagnostics, significance results, and training log. Historical CSVs, pickles, and figures have not been rewritten to resemble outputs from corrected code.

The corrected pipeline has been checked with offline regression tests, short synthetic training/evaluation runs, and a short run on the existing cached stock panel. These checks establish software behavior, not a replication of the full empirical experiment. No new full-grid performance claim is made.

## Historical run: verified facts

| Item | Evidence |
| :--- | :--- |
| Universe | 486 surviving/current-constituent stocks in the cached panel |
| Training | February 1980–December 2009: 359 calendar months |
| Validation | January 2010–December 2014: 60 months |
| Test | January 2015–December 2024: 120 months |
| Features | 21; local notebook previously displayed only five |
| Factor grid | 2, 3, 5, 8, 10 |
| ResCAE grid | 125 configurations, each with seeds 10, 30, 45, 99, 2048 |
| CAE-NL grid | 25 configurations, each with those five seeds |
| ResCAE-Fixed grid | 25 configurations, one training realization each |
| Overall ResCAE selected on validation | `(5, 0.1, 0.00005)` |
| Overall ResCAE-Fixed selected on validation | `(8, 0.00001)` |
| Overall CAE-NL selected on validation | `(10, 0.0005)` |
| Checkpoints | None saved in the historical run |

The original stability file contains 650 rows: 625 ResCAE seed/configuration rows and 25 single-run fixed rows. The string `fixed` was counted as a sixth “seed” in some displays. CAE-NL seed diagnostics were omitted.

The cache has 104,776 observed training stock-month returns, of which 93,117 have a complete feature vector; 299 of the 359 training months contain any eligible observation, beginning February 1985. Validation has 26,390 eligible pairs versus 26,591 observed returns. Test has 57,434 eligible pairs versus 57,588 observed returns. Old PCA summaries included a slightly different test panel than conditional models; paired DM comparisons intersected observations, so their sample need not match the standalone R² rows.

## Conclusions supported by the archived outputs

| K | ResCAE predictive R² | ResCAE vs IPCA p | Fixed vs IPCA p | ResCAE vs Fixed p |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1.49% | 0.0747 | 0.0174 | 0.4865 |
| 3 | 1.55% | 0.0198 | 0.5473 | 0.00000130 |
| 5 | 1.83% | 0.0120 | 0.0879 | 0.000908 |
| 8 | 1.61% | 0.0443 | 0.0543 | 0.9674 |
| 10 | 1.98% | 0.00152 | 0.00910 | 0.5054 |

All p-values above are the original unadjusted, two-sided DM results from `significance_tests.pkl`. They are not recomputed using the revised HAC estimator or repaired characteristics.

ResCAE has positive reported predictive R² at each **validation-selected per-K configuration**. Its improvement over IPCA is significant at the unadjusted 5% level for four of five K values. Fixed versus IPCA is significant at 5% for **two of five**, or at 10% for four of five. Conflating those thresholds produced an overstated conclusion in the notes.

The archived fixed-versus-free comparison uses a single fixed model against a five-seed free ensemble, so it does not isolate the effect of releasing the linear weights. Differences in the learned encoder, regularization selection, and seed averaging also matter. Failure to reject a difference at K = 10 does not establish equivalence.

The five-factor ResCAE's reported 1.076 Sharpe exceeds IPCA's 0.929, but its archived Sharpe-difference bootstrap p-value is 0.133. Reconstruction R², forecast R², and portfolio Sharpe answer different questions. No blanket claim that nonlinear models dominate on all objectives is supported.

At the actual validation-selected ResCAE configuration, historical individual-seed diagnostics are:

| Metric | Mean | Sample standard deviation |
| :--- | ---: | ---: |
| Predictive R² | 0.017055 | 0.001132 |
| Sharpe | 1.009332 | 0.127920 |
| Old φ definition | 0.353912 | 0.264568 |
| Relative linear-weight drift | 0.869999 | 0.064854 |

These are individual-seed statistics, not ensemble metrics. The terminal's old stability summary instead selected configurations using mean **test** R², so it describes different models from the validation-selected summary.

## Corrections made

| Area | Verified inconsistency | Current behavior |
| :--- | :--- | :--- |
| Data basis | Excess returns were passed to a feature builder that subtracted the risk-free rate again | Features take raw returns; market regressions subtract the rate once |
| Market regressions | Covariance and variance used mismatched normalizations and unpaired missing values | Per-stock OLS with paired observations, intercept, and residual degrees of freedom |
| Missing momentum | `nanprod` made all-missing windows look like zero returns | Incomplete compounding windows remain missing |
| Date endpoint | Download end date could omit the final 2024 trading day | Exclusive endpoint is January 1, 2025 |
| Cache validity | Only feature count identified a “valid” cache | Version 2 defaults; legacy data require explicit opt-in and remain marked legacy |
| Runtime | Device string entered `use_linear` position in a single-seed call | Keyword arguments/shared training routine |
| Factor grid | `k_list` affected baselines but not conditional search | Every family uses the requested grid |
| IPCA factors | Stored factors preceded the final Γ update; empty months entered the factor mean as zeros | Recompute final factors; unavailable months remain missing |
| CAE factor mean | Empty managed-return months still supplied encoder outputs | Exclude empty months from training and factor means |
| Forecast contract | CAE output availability depended on realized test returns | Predictions use characteristics and training inputs only |
| PCA imputation | Test-set means filled missing observations during projection | Fitted training means fill missing observations |
| Baseline reporting | AE was trained but omitted | Include AE and full-precision numeric metrics |
| Evaluation coverage | Model summaries used different available test observations | Common characteristic-complete test panel |
| Fixed model | Single-seed even when free/NL models were ensembled; zero drift returned without measuring it | Same conditional-model seeds; actual weight equality and measured drift |
| Seed selection | Stability and narrative highlighted highest test scores | Validation-based representative configurations and separate per-family seed groups |
| Portfolios | Sharpe, bootstrap, and plotting had different flat-signal logic | Shared portfolio routine; flat predictions/overlapping tied legs are undefined |
| φ | Pooling factor coordinates counted differences between constant factor loadings as variance | Average within-factor variance; undefined display for flat prediction signals |
| DM direction | Most labeled comparisons used one sign convention, CAE-NL comparisons the opposite | Positive favors the first named model throughout |
| HAC calculation | Lag covariance divided by `T−h`; excessive lags/identical errors could cause undefined operations | Common `T` denominator, bounded lags, explicit degenerate handling |
| Inference | Individual-month bootstrap ignored temporal dependence; only raw p-values reported | Six-month circular blocks; Holm p-values across K for selected comparison families |
| Charts | Wrong test dates, mismatched K/penalties, one-cell “grid” heatmaps, capped drift | Dates from data, matched dimensions/penalties, all-grid heatmaps, uncapped drift |
| Cumulative returns | Arithmetic sum labeled like a buy-and-hold wealth curve | Explicit arithmetic-sum labels and monthly equal-weight comparator |
| Checkpoints | `--skip-train` exited without doing anything; no losses/checkpoints saved | Save complete model bundle and histories; re-evaluate after fingerprint verification |
| Reproducibility | Minute-level paths could collide; no resolved configuration metadata | Unique run directories, recorded settings/versions/data/source hashes |
| Notebook/dependencies | Broken summary path, stale error output, five feature names, missing explicit SciPy dependency | Explicit run path, all 21 features, cleared outputs, direct SciPy requirement |

## Interpretation limits that remain

The current S&P 500 universe, missing delisted stocks, full-sample coverage filter, and historical use of current shares prevent a point-in-time investable backtest. A corrected rerun will not eliminate those source-data limitations. The downloader's adjusted prices and approximate TB3MS conversion further limit financial precision.

`W_skip = Γ_IPCA` fixes one loading branch, not the factors. The encoder and `g` are learned jointly, `g` can represent linear functions, and its biases permit constant effects absent from the baseline's characteristic-only linear map. Thus improved forecasts cannot uniquely be attributed to nonlinear characteristic interactions. An experiment that holds factors fixed and compares nested, matched-capacity loading maps would be needed for that narrower claim.

The branch ratio φ excludes covariance between branches and is not a unique allocation of return variance. The orthogonality penalty does not identify rotations. Weight drift describes movement from one parameter initialization; thresholds such as 0.9 or 1.0 do not establish that the IPCA component has been “discarded.”

Multiple-testing adjustments here cover K within each selected comparison family, not every exploratory plot, specification, or research decision. Bootstrap block length is fixed at six months and needs sensitivity analysis for empirical inference. Missing test months, where present, are removed for the monthly loss/Sharpe inference series; gaps shorten effective time spacing. No transaction-cost or portfolio-capacity analysis is included.

Local `paper/` and `notes/` files were marked as archived planning drafts. Their old references to 27 configurations, 2020–2024 test data, uniform AE dominance, or “proof” of nonlinearity are not established by the saved experiment. These local drafts remain excluded from Git; this audit and the README provide the public interpretation.

## Verification performed

- **19 offline regression tests passed**, covering raw/excess return handling, paired OLS, missing momentum, legacy-cache rejection, training-only PCA imputation, forecast independence from realized returns, empty factor months, frozen-weight drift, φ, portfolio consistency, DM direction, Holm adjustment, block sampling, requested grids/devices, seed parity, and validation-based selection.
- A single-seed synthetic run completed all six families and generated all applicable figures.
- A synthetic two-seed run with K = 2 and 3 and two linear-penalty settings exercised the grid, all three conditional ensembles, stability summaries, and plotting.
- `--skip-train` successfully reloaded that multi-seed checkpoint and regenerated evaluation outputs with matching data.
- A three-epoch K = 2 diagnostic completed on the original 486-stock cache, explicitly opting into legacy features. Its results are not a corrected empirical experiment.
- All seven notebook code cells executed with an explicit legacy-cache override under a noninteractive plotting backend. The historical chart was visually reviewed, documentation links checked, and `git diff --check` passed.

Fresh external downloads and GPU execution were not exercised. Download inputs and raw-return forwarding were tested with mocks. The large corrected empirical search remains to be run before replacing the archived results. Disposable verification runs are ignored under `results/smoke-*/`.
