# Exact Starter Update to NHL Win Probabilities — 019

**Run date:** 2026-11-03  
**Status:** RESEARCH; no promotion or site export  
**Champion protection:** Hockey Lab Game v1.0.0 remains frozen.

## Question

Does the identity and prior quality of the actual starting goalie improve the independent win probabilities, and does the effect increase when the two starters have a large quality gap?

The goalie input is the named starter from the same-game postgame goalie box, used as a retrospective proxy for a goalie confirmed before puck drop. Goalie skill uses information from earlier game dates only. This tests the exact-goalie path; a distribution over possible starters is not used in this challenger.

## Design

- Baseline: out-of-fold Team Strength 004 + Schedule 007 B2B home-win probabilities.
- Goalie feature: home-minus-away career-shrunk save percentage for the exact starters, shrunk toward .900; all same-date starter results are excluded from the rating.
- Candidate: baseline logit held fixed as an offset plus a nonnegative goalie effect. Rolling-origin selection compares no-goalie, linear, and linear-plus-high-gap hinge forms. Regularization and feature family are selected on the immediately preceding season; the next season is scored.
- Test seasons: 2024–2026, 4,192 games. These remain development folds, not an untouched final test.
- No markets, same-game goalie performance, or Hockey Lab Game v1.0.0 predictions enter the fit.

## Results

| Test season | Selected goalie form using prior validation | Games | Log loss: baseline → candidate | Brier: baseline → candidate | Accuracy: baseline → candidate |
|---|---|---:|---:|---:|---:|
| 2024 | No goalie effect | 1,400 | 0.65876 → 0.65876 | 0.233477 → 0.233477 | 60.07% → 60.07% |
| 2025 | High-gap hinge | 1,398 | 0.66252 → 0.66104 | 0.235265 → 0.234595 | 59.23% → 60.01% |
| 2026 | Linear effect | 1,394 | 0.68308 → 0.68460 | 0.245150 → 0.245823 | 55.09% → 55.24% |
| **Pooled** | — | **4,192** | **0.668104 → 0.668110** | **0.2379552 → 0.2379552** | **58.13% → 58.44%** |

The candidate changed home-win probability by an average absolute **0.92 percentage points**. The pooled log-loss change was +0.000007 and Brier change was +0.00000005, effectively flat. The paired game bootstrap intervals include zero for both metrics: log loss **[−0.000991, +0.000976]**, Brier **[−0.000469, +0.000461]**. Pooled calibration slope moved from **0.908 to 0.884**, a small regression. The candidate improved both probability scores in 2025 and worsened both in 2026; the 2024 validation process selected no goalie adjustment.

## Related exact-starter goals result

The separate exact-starter goals experiment EXP-0029/Goals Layer 011 tested the named actual starter's prior career-shrunk quality in the fixed-60 goals model across 6,560 games in 2022–2026. Regulation score log loss improved from **3.83180 to 3.83019** and three-way regulation Brier from **0.623055 to 0.622676**. Score log loss improved in all five seasons; regulation Brier improved in four. Home-goal MAE worsened by 0.00014, away-goal MAE improved by 0.00070, and OT Brier was essentially unchanged. Its score distribution also generates regulation and full-game win probabilities, so this remains the stronger exact-starter candidate for the goals-derived probability layer.

## Decision

Use the exact named goalie as the primary goalie input when a pregame starter has been recorded; retain the rotation distribution only while the starter is unknown. The goals model's exact-starter layer remains a candidate because its score-distribution and regulation-probability gains repeat across seasons, though the expected-goal MAE effect is very small and mixed.

The separate win-probability adjustment in this experiment is **not promoted**: pooled log loss and Brier do not improve, and the fold results reverse direction. Keep Hockey Lab Game v1.0.0 frozen. Continue exact-starter research and capture the named goalie before puck drop; when that identity changes, create a new prediction version and preserve the earlier forecast. Do not use postgame starter flags as if they were time-stamped pregame evidence.

## Reproducibility and artifacts

The script writes `summary.json` and game-level `oof_predictions.parquet`. The source manifest identifies the 011 OOF win predictions, canonical game spine V3, and goalie feature snapshot V1. Game-level bootstrap intervals are exploratory and do not account for team/season clustering; only three target seasons are available to experiment 019.
