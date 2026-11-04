# Model Accuracy Battery Run 001 — Goalie Uncertainty and Goal Totals

**Date:** 2026-11-04  
**Status:** Research only; no production model or public projection changed  
**Scope:** Paired uncertainty and goalie-exposure checks, followed by score-distribution total-goal diagnostics

## Runs and data

Both GitHub Actions runs completed successfully on the research branch:

- [HWIN Goalie Uncertainty Audit 006](https://github.com/scottjazdzewski/hockey-projection-lab/actions/runs/41866125088), code commit `789a5c373ba33f08390b599d0daa623e79cbabd2`
- [Goals Rate Tail Audit 007](https://github.com/scottjazdzewski/hockey-projection-lab/actions/runs/41867507889), code commit `f3cdc7a2614f1764b38764fbdd757d0b17888091`

The goalie comparison used 6,993 matched games from target seasons 2022–2026; source runs were PBP `41766609805`, schedule spine `41766612176`, and goalie history `41850348678`. The active HWIN comparison reference was `team-strength-004+b2b-007`. Goalie performance features were prior-only, but each game's goalie identity came from its postgame box score. Treat this as a retrospective upper-bound diagnostic, not a pregame starter validation.

The score-distribution comparison used 6,560 matched out-of-fold games across 2022–2026 from the same PBP and spine runs. Its OOF input SHA-256 was `da95708555bb57be541bfc6298067d9c519deebe62fd4f7ae09ae8fbe88c23f8`. The candidate applies a league-scoring rate offset selected on the prior validation season. Goalie identity was excluded.

## Goalie uncertainty and exposure

Negative score deltas favor the goalie candidate. All 95% intervals include zero, including the team-season block-bootstrap intervals, so none establishes a reliable historical gain.

| Goalie signal | Δ Brier (paired game 95% CI) | Δ Brier (team-season 95% CI) | Δ log loss (paired game 95% CI) | Folds improved, Brier / log loss |
|---|---|---|---|---:|
| Save percentage | −0.000141 [−0.000598, 0.000323] | [−0.000906, 0.000583] | −0.000280 [−0.001279, 0.000740] | 3/5, 3/5 |
| Age-adjusted GAA | −0.000037 [−0.000310, 0.000234] | [−0.000454, 0.000391] | −0.000060 [−0.000681, 0.000549] | 3/5, 2/5 |
| SOG-based GSAx | +0.000075 [−0.000378, 0.000545] | [−0.000660, 0.000799] | +0.000137 [−0.000838, 0.001113] | 2/5, 2/5 |
| Save percentage + GAA | +0.000129 [−0.000472, 0.000730] | [−0.000845, 0.001134] | +0.000320 [−0.000952, 0.001628] | 3/5, 3/5 |
| Save percentage + SOG-GSAx | +0.000152 [−0.000460, 0.000760] | [−0.000824, 0.001176] | +0.000322 [−0.001006, 0.001710] | 2/5, 2/5 |

The current SV% signal remains the strongest of these candidates, but its pooled lift is small and uncertain. GAA and SOG-based GSAx did not add value when combined with SV%.

The SV%-only results by the less-experienced starting goalie's prior-start tier were:

| Lower-exposure goalie starts | Games | Δ Brier | Δ log loss |
|---|---:|---:|---:|
| 0–4 | 270 | +0.001355 | +0.002616 |
| 5–14 | 443 | −0.000399 | −0.000979 |
| 15–39 | 993 | −0.000948 | −0.001975 |
| 40+ | 5,287 | −0.000044 | −0.000051 |

This is descriptive, with no subgroup confidence intervals. The smallest-exposure group is a warning that the current adjustment may be too aggressive for a goalie with fewer than five prior starts; it is not enough evidence to choose a new shrinkage value. The promoted goalie model remains unchanged pending timestamped pregame starters and prospective results.

## Goal-total bias and upper tail

Negative deltas favor the rate-offset candidate. Lower total bias means less signed underprediction; lower loss is better.

| Metric | Baseline | Rate offset | Candidate delta and paired 95% CI |
|---|---:|---:|---:|
| Expected final-total bias (projected − actual) | −0.1859 goals | −0.0446 goals | +0.1413 goals toward zero |
| Expected final-total MAE | 1.85432 | 1.85568 | +0.00136 [−0.00218, 0.00489] |
| Final score log loss | 3.71356 | 3.71040 | −0.00315 [−0.00465, −0.00164] |
| Final-total log loss | 2.17984 | 2.17723 | −0.00261 [−0.00396, −0.00124] |
| Over 5.5 Brier | 0.24547 | 0.24415 | −0.00131 [−0.00188, −0.00074] |
| Over 6.5 Brier | 0.24887 | 0.24786 | −0.00101 [−0.00154, −0.00043] |

For over 5.5, the team-season bootstrap interval was [−0.00236, −0.00029], also below zero. For over 6.5 it was [−0.00209, +0.00009], which includes zero. Both over-probability forecasts moved closer to the observed rates: over 5.5 from 53.26% to 55.56% versus 57.16% observed; over 6.5 from 42.81% to 45.11% versus 46.17% observed.

The separate win-probability check did not improve: full-game home-win Brier moved from 0.236704 to 0.236849 (Δ +0.000145; paired 95% CI [−0.000092, +0.000374]) and log loss worsened by 0.000313 (paired 95% CI [−0.000163, +0.000786]). This score-distribution result is not evidence to change the independently trained HWIN probabilities.

The offset improved score log loss, total log loss, and both over-threshold Brier scores in all five target seasons. Total-goal MAE worsened in three of five seasons. The environment slices expose a useful boundary: the baseline already had almost no bias in the highest projected-total decile (+0.028 goals), while the global offset pushed that group to +0.192 goals of overprediction. The same global correction reduces underprediction through most lower and middle scoring environments, but it should be tested conditionally before any use.

## Decision and next experiment

- Keep production HWIN and published cards unchanged. No candidate was promoted.
- Continue the prospective goalie ledger. Do not retune goalie exposure on postgame-selected starter identities; first collect timestamped projected and confirmed starter snapshots.
- Next score-model test: fit a small, predeclared scoring-environment-conditioned rate correction on each prior validation season, with a taper or no correction where the baseline already calibrates the highest-total range. Evaluate total bias, MAE, score/total log loss, over 5.5/6.5 calibration, and full-game/regulation win scores on the next untouched season.
- Keep the score-distribution and HWIN conclusions separate. A better total-goal forecast does not by itself justify changing team win percentages.

## Reproducibility

The workflow artifacts contain the OOF/game-level outputs, fold metrics, paired deltas, uncertainty results, exposure strata, and manifests. The GitHub Actions run pages above preserve the successful run records; the uploaded artifacts are retained for 30 days.
