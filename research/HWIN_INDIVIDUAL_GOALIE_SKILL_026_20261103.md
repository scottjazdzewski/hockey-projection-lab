# EXP-0041 — Individual Goalie Skill on Game Win Probabilities

**Model ID:** `HOCKEY-LAB-HWIN-INDIVIDUAL-GOALIE-SKILL-026-RESEARCH`  
**Run date:** 2026-11-03  
**Status:** RESEARCH; no promotion or public export  
**Champion:** Hockey Lab Game v1.0.0 remains frozen.

## Question and feature

Does the actual named starter's calibrated individual goalie ability change and improve the frozen game win probability? For each game, the sole feature is the home starter's EXP-0035 prior-only calibrated GSAx per 30 shots minus the away starter's rating. This tests the two individual goalies in that game directly; it does not compare either goalie with his team's goalie rotation.

A single nonnegative coefficient adjusts the frozen champion's home-win logit, with away probability set as the complement. The goal rates and score distribution are not changed.

## Validation

- Baseline: frozen Team Strength 004 + B2B out-of-fold probabilities.
- Target: regular seasons 2024–2026, 1,312 games per season (3,936 total).
- For target season S, fit through S−2, select ridge strength on S−1, refit through S−1, and score S. The no-goalie baseline is retained when it wins validation.
- Goalie ratings use earlier starts. One zero-shot historical starter row used that goalie's latest prior rating.
- The target game's actual starter ID comes from postgame goalie records. These results are retrospective until timestamped pregame starters are available.
- No team-goalie mix, team-goalie win record, sportsbook market, or goal-model probability is used.

## Results

| Metric | Frozen champion | Individual-goalie skill | Change |
|---|---:|---:|---:|
| Home-win log loss | 0.668622 | 0.668702 | +0.000080 (worse) |
| Home-win Brier | 0.238209 | 0.238245 | +0.000036 (worse) |
| Winner accuracy | 58.10% | 58.05% | −0.05 pp |
| Mean absolute probability shift | — | 0.09 pp | — |

The candidate was selected for 2024 only; validation selected the frozen baseline for 2025 and 2026. The 2024 skill adjustment worsened both Brier (0.233402→0.233510) and log loss (0.658592→0.658831). Pooled paired 14-day block-bootstrap intervals include no change: log loss **−0.000034 to +0.000207**, Brier **−0.000020 to +0.000097**.

## Decision

**Do not promote.** Named goalie identity is a direct model input in this experiment, but the validated adjustment averages only 0.09 percentage points and has no demonstrated out-of-sample win-probability improvement. Do not force a larger shift; keep Hockey Lab Game v1.0.0 frozen and await timestamped pregame goalie identity for a live test.

## Reproducibility and artifacts

The second run exactly reproduced the summary and all 3,936 prediction rows. See `research/exports/HWIN_INDIVIDUAL_GOALIE_SKILL_026_20261103/` for the script, summary, manifest and frozen Parquet predictions.
