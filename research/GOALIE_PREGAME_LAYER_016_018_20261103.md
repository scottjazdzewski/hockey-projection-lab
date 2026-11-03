# Pregame Goalie Projection and Goal Forecast Layer — 016–018

**Run date:** 2026-11-03  
**Status:** RESEARCH; no model promotion or site export  
**Champion protection:** Hockey Lab Game v1.0.0 remains frozen and unchanged.

## Decision

The starter-choice model identifies the goalie better than a last-five-start frequency baseline, and early-season calibration modestly improves the current forward slice. Probability-weighted goalie skill/workload features reduce score log loss slightly in two held-out seasons, but expected-goal error and win/tie probability metrics do not improve consistently. Keep this layer in research; do not update production projections.

## Data and leakage controls

- Canonical game spine V3 and goalie features V1 from the Hockey Lab Drive warehouse, snapshots dated 2026-11-02.
- Starter-choice outcomes use the explicit same-game starter flag only as a label. Candidate goalies and rotation features use starts on strictly earlier dates; all games on the target date are excluded.
- Goalie quality/workload features are computed from prior starts only. The current game's actual starter is used only for the outcome label and a separately labeled retrospective diagnostic.
- The goals comparison uses frozen out-of-fold outputs from Goals Team Form 60 009. The score family and parameter for each game remain the prior-validation selection from that model. Candidate goal rates do not use win probabilities, markets, or same-game goalie identity.
- Source season 2022 is omitted from the goalie impact comparison because the starter-choice OOF begins at canonical game-id season 2022, aligning with Goals source season 2023. Complete overlap is source seasons 2023–2026.
- Game v1.0.0 and all public website exports remain unchanged.

## Experiment 016 — Starter-choice probabilities

An expanding chronological conditional-logit choice model assigns probabilities to each goalie appearing in a team's prior 20 starts plus an explicit unknown/returning-goalie option. It uses recent rotation shares, the latest starter, days since a goalie last started, and back-to-back interactions. L2 strength is fixed at 1.0; there is no tuning against the target season.

| Forward 2026–27 slice through Nov. 1 | Model 016 | Prior-five frequency baseline |
|---|---:|---:|
| Team-side choices | 462 | 462 |
| Top-choice accuracy | 66.0% | 58.2% |
| Multiclass log loss | 0.8458 | 0.9873 |
| Multiclass Brier | 0.4923 | 0.5710 |
| Actual starter covered by prior-20 candidate set | 94.2% | — |

The forward sample is only 231 games. The model predicted 3.2% average unknown-goalie probability while 5.8% of actual starters were outside the candidate set, so new or returning starters remain underweighted early in a season.

## Experiment 017 — Chronological probability calibration

Temperature and an unknown-goalie logit shift are fitted on prior seasons' out-of-fold candidate logits, then applied to a later season. The current early-season check uses matching first-34-day windows from older seasons.

On 2026–27 through Nov. 1, early-window calibration changed log loss from **0.8458 to 0.8369**, multiclass Brier from **0.4923 to 0.4908**, and left top-choice accuracy at **66.0%**. Mean unknown probability moved from 3.2% to 4.8%, still below the 5.8% observed rate. This calibration was not uniformly beneficial: the historical 2024 early-window fold worsened from 0.8963 to 0.9071 log loss. Treat the adjustment as a research candidate, not a promoted calibrator.

## Experiment 018 — Probability-weighted goalie effect on goals

The tested feature families include career-shrunk, 20-start and 40-start GSAx per 30 shots, 5-on-5 GSAx, recent-versus-established performance, save percentage, and workload/rest. A ridge-regularized Poisson rate model uses the frozen baseline log rate as an offset. The feature family is selected on the immediately preceding season; its coefficients are refitted through that validation season and evaluated on the next season.

| Goal-model source season | Selected pregame feature family | Test games | Regulation score log loss: baseline → goalie | 3-way regulation Brier: baseline → goalie | Home / away goal MAE: baseline → goalie | OT Brier: baseline → goalie |
|---|---|---:|---:|---:|---:|---:|
| 2025 | Career-shrunk GSAx | 1,312 | 3.85077 → 3.84675 | 0.62125 → 0.62069 | 1.4089 / 1.3594 → 1.4090 / 1.3570 | 0.16400 → 0.16413 |
| 2026 | GSAx quality + workload | 1,312 | 3.81667 → 3.81485 | 0.64530 → 0.64670 | 1.3866 / 1.3294 → 1.3950 / 1.3289 | 0.18836 → 0.18859 |

Across the two held-out folds, score log loss improved from **3.83372 to 3.83080**, but regulation Brier worsened from **0.63328 to 0.63369** and pick accuracy fell from **56.71% to 56.44%**. Full-game home-win Brier was effectively unchanged (0.24045 to 0.24040). The by-game 95% bootstrap interval for the score-log-loss delta was [-0.00721, -0.00083] in 2025 and [-0.00652, 0.00288] in 2026; it does not account for team/season clustering.

The same feature coefficients applied to postgame actual-starter identities are explicitly retrospective. Their pooled score log loss was 3.83029, only 0.00051 better than the pregame-probability version; neither result establishes a reliable probability improvement.

## Model registry

| Experiment | Status | Decision |
|---|---|---|
| HOCKEY-LAB-GOALIE-STARTER-PROBABILITY-016 | RESEARCH | Keep the rotation-choice probabilities for further forward evaluation; not production-ready. |
| HOCKEY-LAB-GOALIE-STARTER-CALIBRATION-017 | RESEARCH | Early calibration is promising in the current slice but inconsistent across prior early-season folds. |
| HOCKEY-LAB-GOALS-PREDICTED-GOALIE-018 | RESEARCH | Do not promote; scoring likelihood improves slightly while goal MAE and outcome calibration are mixed. |

## Next tests

1. Continue the current-season frozen forward evaluation as more games arrive, especially unknown/new-goalie coverage and calibration.
2. Repeat goalie-impact evaluation on additional rolling test seasons and report team-clustered uncertainty.
3. Test confirmed-starter snapshots only when a lawful, attributable pregame timestamp source is established; never substitute postgame starter flags.
4. Revisit goalie-specific score-distribution integration only if the next folds show stable gains in expected-goal accuracy and regulation/OT probability quality together.

