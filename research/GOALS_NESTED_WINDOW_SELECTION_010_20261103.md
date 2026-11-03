# Goals Model EXP-0028 — Nested 20/40/60-Date Team-History Selection

**Model ID:** `HOCKEY-LAB-GOALS-NESTED-WINDOW-SELECTION-010-RESEARCH`  
**Status:** REJECTED as a promotion candidate; diagnostic result retained  
**Run date:** 2026-11-03  
**Champion protection:** Hockey Lab Game v1.0.0 remains frozen. No website export or deployment.

## Question

EXP-0027's fixed 60-date team-history model looked better than the fixed 20-date baseline, but the 60-date choice followed review of the same development seasons. This experiment asks whether a strictly prior-season selector still favors longer histories when it must choose among 20, 40, and 60 dates before each target season.

## Design

For target season S, train through S−2, then use only S−1 to tune Poisson regularization and select the team-history window. Window selection minimizes independent-Poisson regulation score log loss on S−1. With that window selected, choose a score-distribution family/parameter on S−1, refit through S−1, and score S. The tested target seasons are 2022–2026, 1,312 games each (6,560 OOF games total).

The league scoring-environment window stays fixed at 20 dates. Team/opponent features include lagged goals, 5-on-5 expected goals and shots, power-play production/suppression, penalties, rest and schedule density, and online opponent-adjusted expected-goal/shot ratings. Every game on a date uses the same prior-date state. No market or HWIN input is used. Historical pregame goalie and line-combination snapshots are not available in the current feature warehouse, so goalie identity and lineups remain excluded.

## Window choices

| Target season | Selection season | Selected team-history window | Validation score log loss (20 / 40 / 60) | Selected score law |
|---:|---:|---:|---:|---|
| 2022 | 2021 | 60 | 3.79832 / 3.79116 / **3.78951** | Diagonal-inflated Poisson, 1.45 |
| 2023 | 2022 | 60 | 3.85774 / 3.85291 / **3.84949** | Diagonal-inflated Poisson, 1.45 |
| 2024 | 2023 | 60 | 3.82740 / 3.82591 / **3.82096** | Diagonal-inflated Poisson, 1.60 |
| 2025 | 2024 | 60 | 3.86500 / 3.85755 / **3.85540** | Diagonal-inflated Poisson, 1.30 |
| 2026 | 2025 | 40 | 3.85802 / **3.85461** / 3.85616 | Diagonal-inflated Poisson, 1.30 |

The selector chose 60 dates in four folds and 40 dates in one. It did not select the same window unconditionally.

## Out-of-fold results

| Model | Regulation score log loss | Regulation 3-way Brier | Regulation accuracy | Home / away regulation-goal MAE | Final-total log loss | OT Brier |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 20-date history | 3.83752 | 0.62564 | 47.67% | 1.39765 / 1.35111 | 2.17984 | 0.17320 |
| Fixed 40-date history | 3.83407 | 0.62421 | 47.68% | 1.39605 / 1.35073 | 2.17863 | 0.17320 |
| Fixed 60-date history | **3.83180** | **0.62306** | **47.90%** | **1.39548 / 1.34878** | 2.17814 | 0.17308 |
| Nested 20/40/60 selector | 3.83208 | 0.62328 | 47.82% | 1.39619 / 1.34867 | **2.17806** | 0.17311 |

The nested selector beats fixed 20 dates on score log loss by 0.00544 and on three-way Brier by 0.00237. It is slightly worse than fixed 60 dates on those metrics (+0.00028 log loss; +0.00022 Brier). Its final-total log loss and away-goal MAE are marginally better, while home-goal MAE and regulation accuracy are slightly worse. Overtime Brier is effectively unchanged.

The first four target-season forecasts are identical to the fixed-60 model because the prior-season selector chose 60. In the 2026 target fold, the selector chose 40 from 2025 validation data, but that choice did not improve next-season results over fixed 60: the nested fold's regulation score log loss was 3.81805 versus 3.81668 for fixed 60, and its regulation Brier was 0.64641 versus 0.64530. This is a useful counterexample to treating validation winner as a guaranteed improvement.

## Calibration and limitations

The score model's mean regulation-goal errors are −0.056 home and −0.094 away goals per team-game. Final total goals are underpredicted by 0.158 goals per game on average. The predicted over-5.5 rate is 53.70% versus 57.16% observed; over-6.5 is 43.25% versus 46.17% observed. Conditional home regulation-win Brier is 0.22967 (log loss 0.65085; calibration slope 1.13). Full-game home-win Brier is 0.23547 (log loss 0.66325; accuracy 59.73%; slope 1.21). Overtime is forecast at 21.49% versus 22.24% observed; OT Brier is 0.17311 and its probabilities have a narrow range, limiting discrimination.

The model remains a development system, not a production-ready site forecast. Expected-goal and total-goal calibration, overtime discrimination, and absent timestamped goalie/lineup inputs need more work. The 2022–2026 rolling-origin scores are development evidence; they are not a future-season live test or an untouched final promotion holdout.

## Decision

Reject EXP-0028 as a promotion candidate: it does not improve on fixed 60-date history, and the single 40-date choice did not improve the 2026 target fold. Keep EXP-0027's fixed-60 model at CANDIDATE for further research only. Its prior-season window was selected in four of five nested folds, but the 2026 counterexample and missing goalie/lineup layer prevent promotion. Do not publish predictions from this experiment. Hockey Lab Game v1.0.0 remains unchanged.

## Next highest-value work

1. Test goalie strength and starter identity using only a defensible pregame availability path; then measure its incremental goal-forecast value by rolling origin.
2. Investigate the systematic total-goal underprediction through rate/pace and scoring-environment features, with all choices confined to prior validation seasons.
3. Continue score-law tests focused on total tails, low-score dependence, and overtime/tie probability calibration.
4. Establish a frozen 2026–27 shadow ledger only after the goal model clears a documented candidate gate; preserve each pregame prediction and its input cutoff.

## Artifacts

The experiment package contains the Python pipeline, pinned requirements, run manifest, fold selection metrics, 6,560-game OOF predictions, full regulation and final score PMFs (676 and 729 cells per game, float32 and Zstandard-compressed), season/context diagnostics, reliability bins, and this report. A second full run reproduced predictions and metrics exactly. Maximum PMF sum error is 2.2e-8, and the largest PMF-to-ledger expected-goal difference is 1.5e-6. Input snapshots are hashed in the manifest.
