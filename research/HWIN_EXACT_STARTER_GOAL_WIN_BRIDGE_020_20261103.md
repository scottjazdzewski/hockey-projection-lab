# Exact-Starter Goal-to-Win Bridge — EXP-0034

**Model ID:** `HOCKEY-LAB-EXACT-STARTER-GOAL-WIN-BRIDGE-020-RESEARCH`  
**Run date:** 2026-11-03  
**Status:** RESEARCH; no promotion or site export  
**Champion:** Hockey Lab Game v1.0.0 remains frozen.

## Question

Does using the named goalie who actually started, through that goalie's lagged performance and workload, improve the expected-goals score distribution and the win probabilities derived from that distribution? Does the score-derived probability add information to the frozen win model?

The goal model estimates each team's goal distribution directly from hockey features. Full-game win percentages are calculated from that score distribution and overtime treatment; they are not backed out from the existing win percentages or market prices.

## Inputs and timing

The goal-model probabilities are out-of-fold outputs for 2025 and 2026 regular seasons. They use the fixed score model from Goals 009 and its goalie-effect experiments. The exact-starter diagnostic uses same-game starter identity from the postgame goalie box. Goalie quality and workload use only starts on earlier dates, but the identity itself has no pregame timestamp. Results therefore measure an **upper-bound retrospective signal**, not a deployable pregame forecast.

The frozen HWIN comparator is the `team_strength_004_plus_b2b` out-of-fold probability, joined by game ID. We filtered to the same regular-season games before scoring. A convex blend weight was selected on 2025 by full-game win log loss, then applied unchanged to 2026. A separate HWIN challenger adds exact-starter goalie features directly and selects its feature family on the prior season.

## Exact-starter goal model evidence

The five-season goalie ablation (EXP-0029) directly included the opposing actual starter's prior-date, career-shrunk goalie performance. Across 6,560 out-of-fold games from source seasons 2022–2026:

| Metric | No-goalie model | Exact-starter goalie layer | Change |
|---|---:|---:|---:|
| Mean home/away goal-count log loss | 1.92155 | 1.92083 | −0.00072 |
| Regulation score log loss | 3.83180 | 3.83019 | −0.00161 |
| Regulation 3-way Brier | 0.62306 | 0.62268 | −0.00038 |
| Home regulation-goal MAE | 1.39548 | 1.39562 | +0.00014 (worse) |
| Away regulation-goal MAE | 1.34878 | 1.34809 | −0.00069 |
| Overtime Brier | 0.17308 | 0.17309 | +0.00001 (worse) |

Regulation score log loss improved in all five seasonal folds; regulation 3-way Brier improved in four of five. The result supports retaining exact goalie identity and established goalie skill as a candidate goals-model input. It does not show a large reduction in expected-goal error: home and away goal MAE move in opposite directions and only slightly.

## Win-probability bridge

The exact-starter goal model's full-game probabilities are derived from its score distributions. In the 2026 untouched regular-season fold, the 2025-selected blend used a 47% weight on the exact-starter score-model probability and 53% on frozen HWIN.

| 2026 regular season, 1,312 games | Log loss | Brier | Winner accuracy |
|---|---:|---:|---:|
| Frozen HWIN v1.0.0 | 0.68441 | 0.24577 | **55.18%** |
| Exact-starter goals score probability alone | 0.68375 | 0.24545 | 54.65% |
| HWIN + exact-starter score probability, 2025-selected blend | **0.68296** | **0.24509** | 54.27% |
| HWIN + no-goalie score probability, 2025-selected blend | 0.68327 | 0.24523 | 54.57% |

The exact-starter blend's point estimates improve log loss by 0.00145 and Brier by 0.00069 relative to HWIN, but its 14-day paired block-bootstrap 95% intervals include zero: **log loss −0.00352 to +0.00085; Brier −0.00169 to +0.00042**. Accuracy is lower than HWIN. The exact-starter score probabilities alone also vary by season: the 2025 fold improves Brier but loses accuracy; in 2026 they are worse than the no-goalie score probabilities.

The separate exact-starter HWIN challenger did not improve probabilistic scoring in the matched 2026 regular-season games: log loss **0.68587 vs 0.68441** and Brier **0.24641 vs 0.24577**. Accuracy increased by 0.08 percentage points, which is not enough to offset worse probability scores.

## Decision

1. Keep exact named-starter identity and that goalie's prior, shrinkage-controlled skill/workload in the **goals-model candidate**. It adds small, repeatable historical score-likelihood signal, though expected-goal MAE is mixed.
2. Keep Hockey Lab Game v1.0.0 as the **win-probability champion**. The exact-starter win layer and the score-derived blend do not yet show repeatable, statistically clear out-of-sample improvement in win probabilities.
3. Do not publish retrospective-starter probabilities as pregame outputs. The current warehouse's actual-starter box has no availability timestamp. A prediction run must receive a time-stamped named goalie ID with `projected` or `confirmed` status; a goalie change before puck drop creates a new immutable prediction revision. The postgame starter identity is used only to score the frozen forecast.

This is not evidence that goalie identity is unimportant. It means the goals layer has a modest score-distribution signal, while the win-probability lift remains small and not stable enough to replace the current champion.

## Next work

1. Add the pregame named-goalie snapshot to the scoring input path, preserving goalie ID, source, status, retrieval time, prediction cutoff, and each update before puck drop.
2. Re-run the goalie-adjusted expected-goals model after each starter change; regenerate regulation, tie/OT, and full-game probabilities from its score distribution.
3. Freeze each pregame probability revision and continue a forward test. Compare exact confirmed starter against projected starter, and evaluate probability lift on matched games over additional untouched seasons.
4. Promote neither the goal model nor the win model until timestamped inputs and repeatable rolling-origin tests satisfy the existing promotion standard.

## Artifacts

- `hockey_exact_starter_goal_win_bridge_020.py` — reproducible comparison and paired block bootstrap.
- `paired_predictions.parquet` — game-level matched probabilities and outcomes for the 2025/2026 folds.
- `summary.json` — metrics, blend weights, calibration diagnostics, and bootstrap intervals.
- `manifest.json` — versions, input hashes, and artifact hashes.
