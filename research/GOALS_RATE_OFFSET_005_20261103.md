# Hockey Goals Rate Offset 005

**Experiment:** EXP-0023  
**Model ID:** `HOCKEY-LAB-GOALS-RATE-OFFSET-005-RESEARCH`  
**Status:** RESEARCH  
**Run date:** 2026-11-03  
**Champion protection:** Hockey Lab Game v1.0.0 remains frozen. No HWIN or betting-market inputs. No website export.

## Question and design

The EXP-0020 goals model reduced total-goal underprediction versus its prior baseline, but still forecast about 0.186 fewer final goals per game than observed. This experiment tests a single validation-tuned rate-level correction without changing the hockey-process feature set.

For each target season S, fit the hockey-process-plus-league-scoring count model through S−2 and predict S−1. Estimate separate home and away log-rate residuals as `log(sum(observed goals) / sum(predicted goals))`. Choose one common shrinkage strength from `{0, .25, .5, .75, 1}` by combined home/away Poisson count log loss on S−1. Refit the rate model through S−1, apply the selected strength and preceding-season residuals to S, then score S. The target-season outcomes are not used to estimate or select the correction.

The comparison uses the same 6,560 rolling-origin target games in source seasons 2022–2026. For score-distribution evaluation, the corrected model reselects its score family on S−1 using corrected S−1 rates. No goalie or lineup identities were available as timestamped pregame inputs.

## Results

| Metric | Unadjusted EXP-0020 selected distribution | Rate-offset 005 |
|---|---:|---:|
| Home / away regulation-goal MAE | **1.3977 / 1.3511** | 1.4017 / 1.3531 |
| Home / away mean regulation-goal error | −0.0715 / −0.1081 | **−0.0109 / −0.0248** |
| Mean final-total forecast error | −0.1859 | **−0.0446** goals/game |
| Final-total MAE | **1.8543** | 1.8557 |
| Regulation exact-score log loss | 3.8375 | **3.8354** |
| Final exact-score log loss | 3.7136 | **3.7104** |
| Final-total log loss | 2.17984 | **2.17723** |
| Regulation 3-way Brier | **0.62564** | 0.62582 |
| OT Brier | **0.17320** | 0.17323 |

The validation procedure selected full-strength offsets (`λ=1`) in all five folds. Validation residuals were positive for both home and away scoring, ranging from roughly 0.8% to 5.0% on the rate scale. On the held-out target seasons, the adjustment reduced systematic underprediction and modestly improved score likelihood, but did not improve expected-goal MAE or regulation/OT Brier. It therefore has not demonstrated a stable all-output improvement.

## Decision

Keep at **RESEARCH**. Do not promote or publish. EXP-0020's unadjusted rate remains the safer reference until the correction is challenged with a wider, predeclared shrinkage grid and additional prospective evidence. The fact that every fold selected the grid maximum is a reason to test the range, not evidence that a larger correction should be used.

Hockey Lab Game v1.0.0 remains the frozen full-game win-probability champion. This experiment changed no public files and generated no live predictions.

## Next experiment

Test a wider predeclared common-shrinkage grid `{0, .5, 1, 1.25, 1.5, 1.75, 2}` with the same rolling-origin design. Compare mean error, MAE, count/score likelihood, regulation Brier, and OT Brier by season. Do not promote a correction that only removes aggregate bias while degrading point accuracy or probability quality.
