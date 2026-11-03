# Goals and Score Distribution Model 002

**Experiment:** EXP-0020  
**Model ID:** `HOCKEY-LAB-GOALS-DISTRIBUTION-002-RESEARCH`  
**Status:** RESEARCH  
**Run date:** 2026-11-03  
**Champion protection:** Hockey Lab Game v1.0.0 remains frozen and unchanged. No HWIN probabilities or betting-market inputs were used. No website export was generated.

## Research question

Does a strictly lagged league scoring environment improve pregame team-goal forecasts, and does a flexible diagonal score law improve regulation-tie/OT estimates without damaging the full score distribution?

## Data and validation

- Same canonical game spine V3 and SportsDataverse/fastRhockey PBP feature snapshot as EXP-0019.
- Five completed regular seasons, source seasons 2022–2026; 1,312 games per season and 6,560 rolling-origin test predictions.
- For each target season S, features and count-model tuning used only seasons before S. Parameters for the score distribution were selected on S−1 and scored on S. Results are development out-of-fold metrics; these five periods have all been inspected and are not an untouched promotion test.
- Regulation score labels were reconstructed and cross-checked against the canonical spine. Shootout games remain tied in regulation; the official shootout winner receives one final-score goal. Regulation ties match the overtime flag.
- Team performance features are shifted through the previous game date. The new league scoring feature is the mean regulation goals per team-game over the prior 20 unique game dates, shifted one full date and requiring at least eight dates. Games on the target date cannot affect that date's forecasts.
- No timestamped historical pregame starter or lineup snapshots were available, so goalie and lineup identity are excluded. Actual postgame starter data was not used.
- Current-season 2026–27 predictions were not generated: the research script and tested ledger stop at source season 2026, and a leakage-safe live inference/export path is not yet built.

## Model changes

The team-goal rates use the hockey-process feature bundle from EXP-0019 plus the lagged league scoring environment. The score law now compares independent Poisson, negative binomial, bivariate Poisson, Dixon–Coles, and diagonal-inflated Poisson. Distribution family and its parameter are chosen on the immediately preceding season, then frozen for the target season. The diagonal multiplier selected in all five folds (1.30–1.60).

Regulation and full-game probabilities are kept separate. If `p_ot` is the regulation-tie probability and `q_home_ot` is the training-only home OT win rate, then:

- `P(home full-game win) = P(home regulation win) + p_ot × q_home_ot`
- `P(away full-game win) = P(away regulation win) + p_ot × (1 − q_home_ot)`

The two full-game probabilities sum to one. This makes explicit why a team's win probability is not its share of expected goals. The expected-goal share is a ratio of expected scores; win probability depends on the complete score distribution and how tied regulation games resolve in OT.

## Pooled out-of-sample results

The comparison below is validation-selected score law in EXP-0019 (Dixon–Coles) versus validation-selected score law in EXP-0020 (diagonal-inflated Poisson). The outcome Brier uses away win / regulation tie / home win.

| Metric | EXP-0019 | EXP-0020 | Change |
|---|---:|---:|---:|
| Regulation exact-score log loss | 3.8480 | **3.8375** | Improved |
| Final exact-score log loss | **3.7046** | 3.7136 | Worse |
| Final total-goal log loss | 2.1829 | **2.1798** | Slightly improved |
| Regulation-outcome Brier | 0.6291 | **0.6256** | Improved |
| Regulation-outcome log loss | 1.0477 | **1.0397** | Improved |
| Regulation most-likely-class accuracy | 47.7% | 47.7% | No change |
| Home/away regulation-goal MAE | 1.398 / 1.351 | 1.398 / 1.351 | Essentially unchanged |
| Mean final-total forecast error | −0.240 | **−0.186** goals/game | Less underprediction |
| Predicted / observed OT rate | 17.0% / 22.2% | **21.6% / 22.2%** | Closer in the aggregate |
| OT Brier | 0.1756 | **0.1732** | Improved, but still slightly worse than constant-rate baseline (~0.1729) |

EXP-0020 has a full-game home-win Brier of **0.2367**, log loss **0.6659**, accuracy **59.6%**, and calibration intercept/slope **−0.046 / 1.228**. These figures describe the score model's full-game output and do not replace or benchmark against the frozen Hockey Lab Game v1.0.0 win model.

The candidate still underpredicts official final scoring. Its over rates were:

| Final total threshold | Predicted | Observed | Brier |
|---|---:|---:|---:|
| Over 4.5 | 75.3% | 79.2% | 0.1658 |
| Over 5.5 | 53.3% | 57.2% | 0.2455 |
| Over 6.5 | 42.8% | 46.2% | 0.2489 |

Regulation expected-goal MAE is **1.398 home / 1.351 away**. Mean prediction errors are −0.072 home and −0.108 away goals. Final home/away goal MAE is 1.400 / 1.366.

## Stability

Against EXP-0019's selected model, EXP-0020 improved regulation exact-score log loss and three-class Brier in all five target seasons. OT Brier also improved in each season, but the pooled result remains a hair worse than the constant observed-rate forecast. Final-total log loss improved in four of five seasons and regressed slightly in 2025. The aggregate OT-rate improvement should therefore not be mistaken for a production-ready probability calibration.

The broad process feature bundle still has not isolated which hockey mechanisms create the gain. Direct 5v5 xG/shot-pace-only remains weaker than the richer bundle in the earlier experiment. A feature ablation is required before attributing the improvement specifically to xG, special teams, or the new league environment.

## Data and code quality note

The first EXP-0020 run exposed a variant-label collision between the `hockey_process` baseline and the plus-league distribution loop. The source was corrected to emit unique feature-set-qualified names, then the complete rolling-origin pipeline was rerun. The verified ledger now preserves both feature sets as distinct variants; the selected-model metrics are unchanged. Saved OOF checks confirm unique game/model rows, date-safe cutoffs, regulation probabilities summing to one, full-game probabilities summing to one, and overtime probability equaling regulation-tie probability.

## Decision

Keep EXP-0020 at **RESEARCH**. It is an improvement candidate for regulation scoring and regulation outcome probabilities, but not ready for public Game Cards. The full final-score likelihood regressed, total-goal thresholds remain underpredicted, OT Brier does not beat simple OT-rate climatology, and no current-season inference path or timestamped goalie/lineup layer exists. Keep Hockey Lab Game v1.0.0 as the frozen full-game win-probability champion. Leave expected-goal, regulation-win, and overtime website fields blank/pending until a candidate clears the documented production gate.

## Highest-value next experiments

1. Test a leakage-safe latent pace/scoring-environment mixture and constrained overdispersed score laws against the full score and total-goal likelihoods. Tune only on the prior season.
2. Test a training-only scoring-rate intercept/shrinkage layer and attack/defense decomposition. Diagnose residual total bias without a fixed post-hoc uplift.
3. Run a predeclared process ablation: recent goals, opponent-adjusted 5v5 xG, SOG pace/suppression, special teams, penalties, and league environment, each against the same chronological folds.
4. Build a timestamped pregame goalie/lineup snapshot dataset from a legally accessible free source, then test whether starter strength/workload and available lineup strength add value.
5. Build a live 2026–27 inference path and shadow ledger only after historical candidate checks pass; preserve each prediction and its input cutoff before puck drop.
