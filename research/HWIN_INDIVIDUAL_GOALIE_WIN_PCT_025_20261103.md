# EXP-0040 — Individual Goalie Start Win Record on Game Probabilities

**Model ID:** `HOCKEY-LAB-HWIN-INDIVIDUAL-GOALIE-WIN-PCT-025-RESEARCH`  
**Run date:** 2026-11-03  
**Status:** RESEARCH; no promotion or public export  
**Champion:** Hockey Lab Game v1.0.0 remains frozen.

## Question and feature

Does the actual named starter's own prior team win record improve the frozen game win probability? Each goalie is tracked by individual goalie ID across every team he has played for. The feature is that goalie's prior team wins above the frozen champion's expected wins, shrunk by his own prior starts. It does not compare him with the club's prior goalie rotation.

The home-minus-away individual-goalie residual adds to the frozen HWIN champion's home-win logit. No xG, goal-rate, market, or site output is used.

## Validation

- Baseline: frozen Team Strength 004 + B2B out-of-fold probabilities.
- Target: regular seasons 2024–2026, 1,312 games per season (3,936 total).
- For target season S, fit through S−2, select the goalie history window, shrinkage and ridge penalty on S−1, refit through S−1, and score S.
- All goalie-start outcomes and expected wins are lagged strictly by date. The team cannot appear twice on the same date.
- The exact target starter comes from the postgame goalie box, so this is an upper-bound retrospective result until timestamped pregame starters are available.

## Results

| Metric | Frozen champion | Individual-goalie W% residual | Change |
|---|---:|---:|---:|
| Home-win log loss | 0.668622 | 0.668701 | +0.000079 (worse) |
| Home-win Brier | 0.238209 | 0.238253 | +0.000044 (worse) |
| Winner accuracy | 58.10% | 57.85% | −0.25 pp |
| Mean absolute probability shift | — | 0.51 pp | — |

Paired 14-day block-bootstrap intervals cross zero for log loss (**−0.000424 to +0.000577**) and Brier (**−0.000196 to +0.000284**). The point forecast moves by a small amount based on the specific goalie, but the historical test does not show improved probability quality: 2024 and 2025 proper scores worsened, while 2026 was nearly neutral.

## Decision

**Do not promote this adjustment.** The individual goalie identity and his own prior start record are now directly represented and tested, but the probability lift is uncertain and the pooled proper scores are slightly worse. Keep Game v1.0.0 unchanged; use a time-stamped pregame starter before any live/public update.

The companion EXP-0041 tests a different individual goalie measure—calibrated prior GSAx/30 directly against the same frozen win probabilities—without combining the two signals.

## Reproducibility and artifacts

The second run exactly reproduced the summary and all 3,936 prediction rows. See `research/exports/HWIN_INDIVIDUAL_GOALIE_WIN_PCT_025_20261103/` for the script, summary, manifest and frozen Parquet predictions.
