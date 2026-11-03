# Goals Model EXP-0026 — Longer Team-Form Window

**Model ID:** `HOCKEY-LAB-GOALS-LONGER-TEAM-FORM-008-RESEARCH`  
**Status:** CANDIDATE for further testing; not promoted  
**Run date:** 2026-11-03  
**Champion protection:** Hockey Lab Game v1.0.0 remains frozen. No public export or website change.

## Question and isolated change

Does a slower team-history window improve goal-rate forecasts and score probabilities? This experiment changes team and opponent rolling features from the previous 20 team-game dates to 40. It keeps the league-scoring environment at 20 dates, uses the same hockey-process feature set, opponent-adjusted ratings, regularization grid, and score-distribution selection procedure.

For each target season S, the model trains through S−2, tunes on S−1, refits through S−1, and scores S. The five target seasons are 2022–2026, 1,312 games each, 6,560 out-of-fold games total. Every target-date feature excludes all games on that date. Markets, the existing win model, and same-game goalie or lineup identity are not used. The source lacks a reliable timestamped historical goalie/lineup snapshot. Canonical winner checks and regulation-tie/overtime label checks passed.

## Results against the 20-game pooled-rate baseline

| Metric | 20-game baseline | 40-game candidate | Change |
|---|---:|---:|---:|
| Home regulation-goal MAE | 1.39765 | 1.39605 | −0.00160 |
| Away regulation-goal MAE | 1.35111 | 1.35073 | −0.00038 |
| Home regulation-goal RMSE | 1.74298 | 1.73964 | −0.00334 |
| Away regulation-goal RMSE | 1.68530 | 1.68277 | −0.00253 |
| Regulation score log loss | 3.83752 | 3.83407 | −0.00344 |
| Final score log loss | 3.71356 | 3.71007 | −0.00349 |
| Final-total log loss | 2.17984 | 2.17863 | −0.00121 |
| Regulation 3-way Brier | 0.62564 | 0.62421 | −0.00143 |
| Regulation outcome accuracy | 47.67% | 47.68% | +0.02 pp |
| Non-tied regulation home-win Brier | 0.23141 | 0.23024 | −0.00117 |
| Overtime Brier | 0.173195 | 0.173203 | +0.000007 |
| Full-game home-win Brier from score model | 0.23670 | 0.23590 | −0.00080 |

Raw independent-Poisson goal rates also improve slightly: home/away goal-count log loss changes from 1.94229/1.90614 to 1.94051/1.90473. Their MAEs change from 1.39888/1.35058 to 1.39723/1.35030, and mean underprediction narrows from −0.0457/−0.0980 to −0.0375/−0.0897 goals per team-game.

The validation procedure selected diagonal-inflated Poisson for both models in all five folds, with the same multiplier in every corresponding fold (1.45, 1.45, 1.60, 1.30, 1.30). This makes the score-distribution improvement attributable to the rate features and their selected regularization, rather than a different score-family choice. The longer-window model improved regulation score log loss and regulation 3-way Brier in four of five target seasons; the 2026 fold worsened on both. Overtime Brier was effectively unchanged, with predicted OT frequency 21.54% versus an observed 22.24%.

Goal-rate calibration slopes remain broadly similar. The home rate mean error improves from −0.0715 to −0.0633 and away from −0.1081 to −0.0999. The pooled improvement is repeatable enough to keep testing, but the point-error gains are small, accuracy is essentially unchanged, and the latest fold regressed.

## Decision

Keep this version at **CANDIDATE** for continued research. Do not promote or publish it. These are development rolling-origin folds, not an untouched promotion test, and the 2026 fold is a meaningful counter-signal. The independent win-probability champion remains unchanged.

## Next test

EXP-0027 will test a 60-game team-form window with the league-scoring window held at 20. Compare 20, 40, and 60 on the same chronological folds; retain the 40-game candidate only if its improvement survives this additional challenge and season/context diagnostics. Any later promotion still needs a separate forward/shadow test and a documented calibration review.
