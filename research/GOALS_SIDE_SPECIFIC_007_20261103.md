# Goals Model EXP-0025 — Separate Home and Away Scoring Rates

**Model ID:** `HOCKEY-LAB-GOALS-SIDE-SPECIFIC-RATES-007-RESEARCH`  
**Status:** REJECTED  
**Run date:** 2026-11-03  
**Champion protection:** Hockey Lab Game v1.0.0 remains frozen. This is a separate goals/distribution experiment. No public export or website change.

## Question

Does fitting one Poisson goal-rate regression for home teams and another for away teams improve pregame expected goals or the score distribution compared with the pooled home/away regression?

## Design and leakage controls

The run covers five complete target seasons (source seasons 2022–2026), with 1,312 regular-season games per season and 6,560 out-of-fold games total. For target season S, models are trained on seasons through S−2, regularization is tuned on S−1, and the chosen model is refit through S−1 before scoring S. The separate fits use the same lagged hockey-process and prior-date league-scoring predictors as the pooled rate model, with the constant `is_home` feature removed. Home and away regularization strengths are selected independently using only their validation-season goal-count log loss. The score-distribution family and parameter are also selected on S−1.

Pregame team and opponent rolling features exclude the entire game date. The target-season outcomes are used only for scoring. No win-model probabilities, markets, same-game goalie identity, or same-game lineup information enters the model. The canonical winner cross-check passed, regulation ties matched overtime status, and all 2022–2026 regular-season target labels passed validation. The source warehouse does not yet contain reliable timestamped historical goalie/lineup snapshots, so those inputs remain excluded.

## Results

The registered baseline is the pooled-rate hockey-process model with the validation-selected score law. The candidate uses separate home/away Poisson rate fits and selects the score law on the preceding validation season.

| Metric | Pooled baseline | Separate home/away | Change (candidate − baseline) |
|---|---:|---:|---:|
| Home regulation-goal MAE | 1.39765 | 1.39793 | +0.00028 |
| Away regulation-goal MAE | 1.35111 | 1.35239 | +0.00128 |
| Home regulation-goal RMSE | 1.74298 | 1.74626 | +0.00329 |
| Away regulation-goal RMSE | 1.68530 | 1.68615 | +0.00085 |
| Regulation score log loss | 3.83752 | 3.83980 | +0.00229 |
| Final score log loss | 3.71356 | 3.71611 | +0.00255 |
| Final-total log loss | 2.17984 | 2.18144 | +0.00160 |
| Regulation 3-way Brier | 0.62564 | 0.62635 | +0.00071 |
| Regulation outcome accuracy | 47.67% | 47.50% | −0.17 pp |
| Overtime Brier | 0.17320 | 0.17314 | −0.00006 |
| Non-tied regulation home-win Brier | 0.23141 | 0.23209 | +0.00067 |
| Full-game home-win Brier from the score model | 0.23670 | 0.23740 | +0.00070 |

Lower is better except accuracy. Overall regulation overtime frequency was 22.24%; the validation-selected model predicted 21.62% with Brier 0.17320, while the candidate predicted 21.72% with Brier 0.17314. That tiny OT difference does not offset the regressions in goal error, score likelihood, regulation outcome calibration, and full-game win Brier.

The directly fitted independent-Poisson rates tell the same story: home raw-rate MAE changed from 1.39888 to 1.39881 (effectively flat), away raw-rate MAE worsened from 1.35058 to 1.35195, and both side-specific Poisson count log losses were slightly worse. The candidate's separate alphas varied substantially across folds (home: 0.1 or 1.0; away: 0.0001 to 1.0), which is consistent with unstable side-specific coefficients rather than a repeatable home/road scoring relationship. The selected score law was diagonal-inflated Poisson in all five folds, but its selected multiplier varied from 1.30 to 1.60.

Season-level comparisons show no durable advantage: regulation score log loss was worse in four folds and tied in one; regulation 3-way Brier was worse in four folds and better only in the 2026 fold. The one better fold does not justify promotion.

## Decision

Reject the separate home/away rate structure for this feature set. Keep the pooled-rate model as the research benchmark; keep Hockey Lab Game v1.0.0 as the unchanged win-probability champion. Do not publish these projections.

## Next experiments

1. Test alternate leakage-safe time decay for 5-on-5 xG creation/suppression and scoring form (e.g. rolling windows versus exponentially weighted histories), choosing decay only on the prior season.
2. Test offensive and defensive component shrinkage / hierarchical partial pooling, with emphasis on whether opponent-adjusted 5-on-5 xG contributes beyond team rolling xG and shots.
3. Audit whether a legally accessible historical source provides timestamped pregame goalie starters; only then run a goalie-identity add-on with starter snapshots frozen before puck drop.
4. Test goal-rate calibration by season and scoring environment using validation-only calibration, while retaining the raw count model as the comparison and avoiding market-derived targets.

## Reproducibility

Code: `research/hockey_goals_side_specific_007.py`  
Dependencies: `research/requirements-goals-side-specific-007.txt`  
Validation outputs: summary JSON, fold metrics, out-of-fold predictions, context metrics, and reliability bins. Source and artifact checksums are in the companion manifest.
