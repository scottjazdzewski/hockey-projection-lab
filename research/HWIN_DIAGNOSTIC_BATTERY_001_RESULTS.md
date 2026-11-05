# HWIN Diagnostic Battery 001 — Verified First Pass

**Status:** Research-only. No production probabilities, weights, or scorer files changed.

**Data:** 6,993 frozen chronological holdout games from Schedule Context 007 (2022–2026), scoring the existing `baseline_b2b` predictions. I ran the updated branch script against that exact frozen Parquet input and verified that all 1,000 bootstrap replicates completed.

**Initial workflow run:** [GitHub Actions run 42045215658](https://github.com/scottjazdzewski/hockey-projection-lab/actions/runs/42045215658)  
**Research PR:** [#3](https://github.com/scottjazdzewski/hockey-projection-lab/pull/3)  
**Verified interval table:** [hwin_diagnostic_battery_001_schedule_cluster_bootstrap.csv](hwin_diagnostic_battery_001_schedule_cluster_bootstrap.csv)

## Overall probability quality

- Brier score: **0.23456**; log loss: **0.66119**; accuracy: **60.00%**.
- The model averaged **54.08%** home-win probability; home teams won **53.65%**. That is about **0.42 percentage points of aggregate home-probability overprediction**.
- Calibration slope was **1.002**, so the aggregate issue does not look like a general “probabilities too extreme” problem.
- The error changed direction by season: home wins were overpredicted by about **2.1–2.2 points** in 2023 and 2026, and underpredicted by about **2.4 points** in 2025. That is a regime stability question, not a reason to apply one global correction.

## Schedule-state diagnostics

Intervals below are for the difference in calibration residual (observed minus predicted home-win rate) between flagged and unflagged games. Negative values mean the flagged games were more overpredicted. The bootstrap resampled dates within seasons and applied shared team-cluster weights for both home and away roles.

| Slice | Flagged vs unflagged residual gap | 95% interval | Read |
|---|---:|---:|---|
| Either team in 3-in-4 | −0.87 pp | −7.62 to +5.54 pp | Includes zero |
| Either team in 4-in-6 | −0.03 pp | −6.52 to +5.98 pp | Essentially no raw gap |
| Either team in 5-in-8 | −0.11 pp | −6.84 to +5.40 pp | Includes zero |
| Away road-trip finale proxy | −1.08 pp | −8.44 to +5.71 pp | Includes zero |

These multiway intervals are broad. They do not support promoting any of these schedule flags as a probability adjustment. The road-trip finale remains a plausible hypothesis to investigate with a sharper, predeclared test, not a confirmed edge.

## Other calibration slices

- Home probabilities from **35–40%** were low by **3.37 points**; **55–60%** were high by **2.20 points**; **70%+** were low by **1.86 points**. The changing direction across bands argues against a blanket offset. Those bins still need uncertainty-adjusted, season-aware confirmation.
- Corrected favorite-confidence diagnostics also move in both directions: favorites at **60–65%** were high by **2.35 points**, while **70–75%** were low by **2.58 points**. The **80%+** band has only 65 games and is too small for a confident read.

## Team-season signal worth checking

Home-team slices are noisy when pooled over changing roster eras. On a complete 32-team by five-season scan (about 41–54 home games per team-season), Philadelphia was the only team whose home residual had the same sign in every season: its home-win probability was overpredicted by **13.1 points** in 2022, **1.1** in 2023, **5.4** in 2024, **0.7** in 2025, and **0.4** in 2026. The pooled gap is about **4.2 points across 210 games**, but its magnitude is mostly driven by 2022. Because Philadelphia emerged from scanning all teams, this is a post-hoc lead requiring a multiple-comparison correction and forward-season validation before any team-specific feature is considered.

## Decision and next test

**No production change.** The diagnostics do not show a stable rest-state or getaway-game correction. The next useful modeling test is a preregistered forward-validation comparison for season/context drift and team-era strength, with block-aware uncertainty and correction for searching across teams and probability bins. Only a candidate that improves held-out Brier and log loss and persists in later seasons should advance.

Market odds remain benchmark-only and are not used as model inputs.


## One-percentage-point calibration: does “60%” win 60%?

I ran a separate pointwise audit on the same frozen holdout, converting each game into both teams’ predicted win probabilities and outcomes. That gives **13,986 team forecasts across 6,993 games**. Probabilities are assigned to the nearest whole-percent bucket; for example, the **60%** bucket contains forecasts from **59.5% through 60.5%**. The full 0–100 table is [HWIN_POINTWISE_CALIBRATION_001.csv](HWIN_POINTWISE_CALIBRATION_001.csv), with the reproducible script at [hwin_pointwise_calibration_001.py](hwin_pointwise_calibration_001.py).

| Rounded forecast | Games | Mean forecast | Actual win rate | Actual minus forecast | 95% date/team block interval for gap |
|---:|---:|---:|---:|---:|---:|
| 54% | 425 | 53.98% | 55.06% | +1.08 pp | −10.20 to +13.07 pp |
| 55% | 390 | 54.99% | 51.79% | −3.20 pp | −15.76 to +9.00 pp |
| 56% | 397 | 56.01% | 56.42% | +0.41 pp | −11.62 to +12.53 pp |
| 57% | 393 | 57.02% | 54.20% | −2.82 pp | −15.13 to +8.91 pp |
| 58% | 361 | 58.00% | 57.34% | −0.65 pp | −13.78 to +11.94 pp |
| 59% | 308 | 58.99% | 60.39% | +1.40 pp | −12.57 to +14.28 pp |
| **60%** | **317** | **59.98%** | **59.31%** | **−0.67 pp** | **−15.85 to +14.15 pp** |
| 61% | 289 | 61.01% | 62.28% | +1.27 pp | −12.27 to +14.05 pp |
| 62% | 316 | 62.03% | 61.71% | −0.32 pp | −15.13 to +13.38 pp |
| 63% | 249 | 63.00% | 55.42% | −7.58 pp | −24.51 to +8.16 pp |
| 64% | 248 | 63.99% | 61.29% | −2.70 pp | −17.81 to +12.76 pp |
| 65% | 211 | 65.00% | 66.82% | +1.82 pp | −17.51 to +18.96 pp |

So the exact rounded **60%** group won **59.3%** of the time in this sample, only **0.7 points below** its forecast. The earlier **60–65% favorite band** is a different, wider bucket: it averaged **62.36% forecast** and **60.01% observed**, a **2.35-point overprediction**. Both findings are true; the wider bucket should not be read as the exact 60% result.

The per-point results jump around, as expected with narrow bins. The largest absolute gap among bins with at least 100 games was the 63% bucket (mirrored by the 37% bucket): **63.00% forecast vs 55.42% observed**, but its block-bootstrap interval includes zero. **None of the pointwise intervals excludes zero.** This means the data does not establish that any individual percentage point is miscalibrated; it does not prove exact calibration either. The intervals are intentionally block-aware and broad, and are pointwise rather than simultaneous across all bins.

The 50% row is marked structural: a game contributes both teams at 50%, guaranteeing a 50/50 observed rate. Do not count that row as evidence that the model is calibrated at 50%. Empty extreme bins are shown in the CSV, and bins with fewer than 100 games are flagged.


## Calibration method test: can a learned layer improve this?

I tested three probability maps with expanding chronological folds. For each test season (2023–2026), each map used only earlier seasons to fit. All mapped both teams symmetrically so each matchup still sums to 100%.

| Method | Out-of-fold Brier | Change vs raw | Out-of-fold log loss | Change vs raw |
|---|---:|---:|---:|---:|
| Raw probabilities | 0.236200 | — | 0.664460 | — |
| Symmetric logit-slope scaling | 0.236400 | +0.000199 | 0.664872 | +0.000412 |
| Symmetric isotonic curve | 0.237408 | +0.001208 | 0.666959 | +0.002499 |
| Shrunken 5-point favorite bands, all prior seasons | 0.236757 | +0.000556 | 0.665612 | +0.001151 |
| Shrunken 5-point favorite bands, last two seasons | 0.236605 | +0.000405 | 0.665281 | +0.000821 |

Positive changes mean worse scores. None of these candidates improved the raw model on either proper score across the four forward folds. The differences are small and this was exploratory, so it does not prove all recalibration will fail. It does mean **we should not add any of these calibration layers now**.

The most actionable lead is a changing favorite-band error: for favorites in the **60–65%** range, the actual-minus-predicted gap was **+4.34 pp** in 2022 (underprediction), **+1.02** in 2023, then **−3.58** in 2024, **−6.26** in 2025, and **−7.12** in 2026 (overprediction). Sample sizes were 254–293 games per later season. This sign change points to season/context drift; it is not a stable correction factor.

**Next calibration work:** trace the 2024–2026 drift to inputs and data freshness (team-strength updates, home-ice treatment, goalie estimate versus confirmation, and schedule context), then retest only a predeclared candidate on forward seasons. Keep the one-point table as a monitor, but do not fit a separate adjustment to each integer percentage. The forward-fold scoring table and repeatable script are [HWIN_ROLLING_CALIBRATION_001_FOLDS.csv](HWIN_ROLLING_CALIBRATION_001_FOLDS.csv) and [hwin_rolling_calibration_001.py](hwin_rolling_calibration_001.py).
