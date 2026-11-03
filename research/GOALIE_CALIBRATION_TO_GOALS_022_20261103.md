# EXP-0036 — Calibrated Named-Goalie Adjustment to Goals and Win Probability

**Status:** RESEARCH; no promotion or public export  
**Model:** `HOCKEY-LAB-GOALIE-CALIBRATION-TO-GOALS-022-RESEARCH`  
**Run date:** 2026-11-03

## Question

Does a calibrated rating for the exact named goalie improve expected goals and the full score distribution beyond the team-only Goals 009 model? Do the resulting regulation and full-game probabilities improve as well?

## Method and leakage controls

The base is frozen Goals Team Form 60 009 out-of-fold rates and its validation-selected score distribution. The candidate multiplies the opposing team’s goal rate by a single shared, nonpositive log-rate coefficient times the named goalie’s prior calibrated GSAx/30 rating. Coefficient regularization is selected on the preceding season and tested on the next season (2024–2026). Goalie skill uses only starts before the target game date.

The target game’s exact starter ID comes from the postgame goalie box, so this remains an upper-bound retrospective diagnostic rather than a live pregame model. The existing win-probability champion is not changed.

## Out-of-sample full-game win results

| Model | Games | Home-win Brier | Home-win log loss | Accuracy |
|---|---:|---:|---:|---:|
| No-goalie Goals 009 score distribution | 3,936 | 0.238724 | 0.669875 | 57.60% |
| Calibrated named-goalie adjustment | 3,936 | 0.239222 | 0.670917 | 57.90% |

All three target seasons had slightly worse full-game Brier and log loss with the absolute goalie-skill adjustment. Accuracy rose about 0.3 percentage points, but the proper probability scores worsened. Mean absolute home-win probability movement was about **1.12 percentage points** pooled; the 2026 fold moved forecasts more, yet its Brier and log loss deteriorated.

Both home and away expected-goal MAE worsened in every test season, and regulation score likelihood did not improve consistently. Thus the rating changes projections but the current tests do not validate this absolute additive goalie layer as a better goals or win model.

## Decision

Do not promote this adjustment. Keep Hockey Lab Game v1.0.0 frozen. Test a replacement effect relative to the goalie quality already embedded in team history and explicitly test the team’s prior win percentage with the same goalie (EXP-0037). Production remains blocked on time-stamped pregame starter identity.

## Artifacts

- Reproducible script: `research/hockey_goalie_calibration_to_goals_022.py`
- OOF score-distribution probabilities: archived with EXP-0036 artifacts
