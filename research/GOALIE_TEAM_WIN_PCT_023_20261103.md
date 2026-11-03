# EXP-0037 — Team Winning Percentage With the Named Goalie

**Status:** RESEARCH; no promotion, production export, or site change  
**Model:** `HOCKEY-LAB-GOALIE-TEAM-WIN-PCT-023-RESEARCH`  
**Run date:** 2026-11-03

## Question

Does the specific goalie's prior team win percentage add information to a game's full-game win probability, after accounting for what the frozen hockey score model already expected? Does it improve on the named-goalie expected-goals adjustment tested in EXP-0036?

## Data and pregame rule

- Base probabilities: out-of-fold Goals Team Form 60 009 validation-selected score distribution, regular seasons 2022–2026.
- Goalie identity: the same-game starter flag and named goalie ID from the postgame goalie box. This is retrospective identity, so this test is an upper-bound diagnostic until a timestamped pregame starter feed is available.
- Target evaluation: rolling-origin regular seasons 2024, 2025, and 2026; 1,312 games each (3,936 total).
- For every team-goalie pair, prior starts are date-lagged. The goalie-specific team record uses only starts before the target game date. Same-day team duplicates are checked and absent.
- No sportsbook data or Hockey Lab Game v1.0.0 probabilities are used in the goals model. The full-game outcome is the actual final winner, including overtime/shootouts.

## Feature tested

Raw team win percentage with a goalie can mostly reflect the team's strength and the situations in which the coach starts that goalie. So the predictor is the team's wins with that goalie **above the frozen score model's expected wins**, divided by prior starts plus a shrinkage prior. This is a smoothed goalie-team win-percentage residual; it shrinks new or lightly used goalie-team pairs toward no added effect.

The rolling validation season selected a career-history feature with a 60-start prior for 2024 and 2025, and a 30-start prior for 2026. A ridge-regularized coefficient was fit on older seasons, selected on the preceding season, and refit before each test season. This feature adds a logit adjustment to the team's full-game probability. It does not change expected goals.

## Out-of-sample results

| Model | Games | Home-win Brier | Home-win log loss | Accuracy |
|---|---:|---:|---:|---:|
| Frozen Goals 009 probability | 3,936 | 0.238724 | 0.669875 | 57.60% |
| Goalie-team win-history residual | 3,936 | 0.238380 | 0.669012 | 57.95% |
| Absolute named-goalie score adjustment (EXP-0036) | 3,936 | 0.239222 | 0.670917 | 57.90% |

For the team-goalie win-history model, pooled loss deltas are small and uncertainty includes no improvement: Brier delta **−0.000343** (paired 14-day block-bootstrap 95% interval **−0.001177 to +0.000439**); log-loss delta **−0.000863** (95% interval **−0.002564 to +0.000735**, 42 calendar blocks). The calibration slope also moved from 0.267 to 0.226, so this is not a calibration win.

| Test season | Frozen Brier | Goalie-history Brier | Frozen log loss | Goalie-history log loss | Mean probability shift |
|---|---:|---:|---:|---:|---:|
| 2024 | 0.235265 | 0.233737 | 0.662626 | 0.659219 | 3.32 pp |
| 2025 | 0.236399 | 0.236853 | 0.665175 | 0.665932 | 2.81 pp |
| 2026 | 0.244507 | 0.244551 | 0.681825 | 0.681886 | 0.78 pp |

The improvement is concentrated in 2024. It reverses slightly in 2025 and is effectively neutral in 2026. The absolute named-goalie GSAx adjustment moved win probabilities, but worsened the pooled Brier score and log loss in this test. Thus the probability should move when a different named goalie is selected, but the current retrospective evidence does not show that the absolute skill adjustment is better than the team-only forecast.

## Interpretation and decision

The team's historical win percentage with the named goalie is now represented explicitly in the research model as a shrinkage-adjusted, expected-win residual. It changes win probabilities by a mean absolute **2.30 percentage points** across the pooled test, but its gain is not statistically clear and does not repeat across seasons. A second run reproduced the summary and all 3,936 game predictions exactly. The goalie adjustment is **not promoted**; Hockey Lab Game v1.0.0 remains frozen, and no production export or site value changes.

This feature is predictive context, not a causal estimate of goalie skill: residual team-goalie records can still contain coaching, lineup, schedule, and selection effects. We should keep comparing it with shot-quality goalie skill and team-relative replacement effects rather than presenting raw goalie W–L records as ability.

## Next experiment

Use timestamped pregame starter snapshots to make the identity genuinely available before puck drop. Then compare a named goalie's calibrated shot-quality rating against the team's prior goalie mix, while measuring score-distribution accuracy and the resulting full-game win probability together. Keep the W% residual as a challenger feature and require another untouched rolling season plus block-uncertainty checks before promotion.

## Artifacts

- Reproducible script: `research/hockey_goalie_team_win_pct_023.py`
- Summary: `summary.json`
- Frozen test predictions: `oof_team_goalie_win_pct.parquet`
- Input model: Goals Team Form 60 009 OOF ledger
- Goalies: NHL/PBP goalie feature table with actual starter IDs; target-game identity is retrospective
