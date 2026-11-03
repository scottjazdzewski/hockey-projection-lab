# Goals and Score Distribution Model 001

**Experiment:** EXP-0019  
**Model ID:** `HOCKEY-LAB-GOALS-DISTRIBUTION-001-RESEARCH`  
**Status:** RESEARCH  
**Run date:** 2026-11-03  
**Champion protection:** Hockey Lab Game v1.0.0 remains frozen and unchanged. This experiment used no HWIN probabilities and creates no website export.

## Question

Can pregame hockey information predict each team's regulation goals and a coherent game score distribution well enough to produce useful regulation win, regulation tie, and overtime probabilities?

## Data and validation

- Canonical game spine V3 joined to the SportsDataverse/fastRhockey team-game PBP feature table.
- Five completed regular seasons, source seasons 2022–2026: 1,312 games per season, 6,560 scored games total.
- Rolling-origin folds: for target season S, tune count-model regularization and score-law parameters on S-1; refit count models through S-1; evaluate on S. No target-season outcomes enter its predictions.
- Targets are regulation scores from PBP. Overtime winners' extra goal is removed; shootout scoreboards remain tied in regulation and receive the official shootout goal only in the final-score target. Target ties match the PBP overtime flag, and regulation winners cross-check against the canonical spine for all 2022–2026 regular-season games.
- 605 source rows were rejected by label-quality checks overall; 8 were in source seasons 2016 onward. No games were excluded from the five scored target seasons.
- Team rolling features use only games before the target calendar date. Opponent-adjusted xG/SOG states are frozen across each date, then updated after all games on that date. Schedule density, rest, and back-to-back features are pregame schedule data.
- Starting goalie and lineup identity were excluded. The warehouse has no timestamped historical pregame starter/lineup snapshots; postgame starter identity would be hindsight leakage.
- No bookmaker data, market totals, or Game v1 probabilities were inputs.

## Model construction

The count layer uses regularized Poisson regression to estimate regulation goals for the home and away sides independently. Tested feature sets include home ice only; recent regulation goals for/against; direct 5v5 xG and shot-pace form; and a broader hockey-process bundle with lagged 5v5 xG/SOG creation and suppression, PP xG/goals, penalties, opponent-adjusted xG/SOG offense and defense, rest, B2B, and schedule density.

The score layer converts both goal rates to a joint regulation score matrix. Rolling-origin validation compared independent Poisson, independent negative binomial, bivariate Poisson, and Dixon–Coles low-score correction. The official final-score distribution adds overtime winner outcomes to regulation-tie mass using a training-only home overtime-win rate with a symmetric Beta(10,10) prior. Each saved game prediction contains the goal rates and selected family parameters needed to reconstruct its full score distribution.

## Pooled out-of-sample results

All metrics below pool the 6,560 rolling-origin target games. Score log losses are for exact scorelines; 3-class outcome metrics use away win / regulation tie / home win.

| Model variant | Regulation score log loss | Final total log loss | 3-class Brier | 3-class accuracy | Home/away regulation goal MAE |
|---|---:|---:|---:|---:|---:|
| Home-ice-only Poisson | 3.8937 | 2.1926 | 0.6504 | 42.1% | 1.402 / 1.388 |
| Recent goal-form Poisson | 3.8628 | **2.1810** | 0.6395 | 45.4% | 1.412 / 1.361 |
| 5v5 xG + shot-pace Poisson | 3.8741 | 2.1955 | 0.6373 | 45.6% | 1.400 / 1.371 |
| Hockey process, no recent goals | 3.8651 | 2.1950 | 0.6320 | 47.0% | 1.398 / 1.361 |
| Full hockey process, independent Poisson | 3.8498 | 2.1834 | 0.6302 | 47.6% | 1.398 / 1.351 |
| Full hockey process, validation-selected Dixon–Coles | **3.8480** | 2.1829 | **0.6291** | 47.6% | 1.398 / 1.351 |

The full hockey-process feature bundle beat recent goal form in regulation score log loss and 3-class Brier in **all five** target seasons. It also improved home and away regulation-goal MAE in each season. The direct 5v5 xG/shot-pace-only set did not beat recent goal form on score likelihood, so the broad-bundle gain cannot yet be attributed to direct xG features alone. The selected full model's final-total log loss is slightly worse than the recent-goal baseline; the scoring-rate component still needs work.

### Expected goals and score calibration

For the validation-selected model:

- Regulation goals: home MAE **1.398** (RMSE 1.743; mean prediction error −0.068); away MAE **1.351** (RMSE 1.686; mean error −0.119).
- Official final goals: home MAE **1.400**; away MAE **1.367**. Final combined goals are underpredicted by **0.240 goals/game** on average (MAE 1.856; RMSE 2.320).
- Predicted versus observed final total-over rates: over 4.5, **74.2% vs 79.2%**; over 5.5, **52.6% vs 57.2%**; over 6.5, **41.5% vs 46.2%**. The model is low on total scoring at each cut.

### Regulation outcome and overtime probabilities

- Regulation away/tie/home probabilities have mean multiclass Brier **0.6291**, log loss **1.0477**, and most-likely-class accuracy **47.6%**.
- Conditional on regulation not tying, home-win Brier is **0.2314**, log loss **0.6546**. Logistic calibration intercept/slope are **−0.060 / 1.136**; the slope indicates predictions are somewhat too compressed.
- The score-derived tie/overtime probability averages **17.0%**, versus **22.2% observed**. Its Brier is **0.1756**, slightly worse than the constant observed-rate Brier of about **0.1730**. Predicted OT probabilities have too little spread to estimate a useful calibration slope; the fixed-slope calibration offset is +0.336 on the log-odds scale.
- Regulation home-win Brier by target season ranges from **0.2240 to 0.2380**. OT Brier ranges from **0.1660 to 0.1924**; predicted OT rates remain below observed rates in each season.

The current score distribution is not calibrated enough for production. In particular, it underestimates scoring totals and regulation ties/overtime. The home/away regulation result signal is useful compared with the home-ice baseline, but this is not a promotion result.

## Distribution comparison

Dixon–Coles was selected on the preceding validation season in all five folds and modestly reduced exact regulation score log loss from 3.8498 to 3.8480 pooled. The gain is small. Independent negative binomial was worse than Poisson on validation score likelihood; bivariate Poisson selected a zero shared component in every fold and therefore reduced to independent Poisson. Dixon–Coles parameters often hit the edge of the tested range, so its result needs a broader constrained search and additional held-out evaluation before it is treated as a stable distribution improvement.

## Data-quality and reproducibility notes

- Code: `research/hockey_goals_distribution_001.py`.
- Model features depend on the existing `hwin_team_strength_004.py` and `hwin_schedule_007.py` feature builders; their online states are used only as hockey-feature inputs.
- Inputs are versioned as `sportsdataverse-pbp-features-20261102` and `canonical-spine-v3-20261102`; retrieval date 2026-11-03.
- The source archive is free at this stage; commercial redistribution rights were not audited as part of this experiment.
- This is a historical development battery, not an untouched test and not a prospective live test. No current predictions were written to `public/data/`.

## Decision and next experiments

Keep this version at **RESEARCH**. Keep Hockey Lab Game v1.0.0 as the frozen win-probability champion.

1. Add a leakage-safe league scoring-environment and era adjustment, then test attack/defense shrinkage and finishing residuals. Diagnose the persistent −0.24 final-goal total bias and underprediction at 4.5/5.5/6.5.
2. Fit a coherent low-score/tie distribution that improves OT calibration while preserving the full score PMF. Compare constrained Dixon–Coles, latent shared-pace mixtures, and a separate generative score-difference construction; do not apply a manual tie uplift.
3. Run a predeclared ablation of opponent-adjusted 5v5 xG versus SOG, direct rolling xG, special-teams quality, recent goals, and finishing residuals. The present bundle improves outcomes, but the responsible hockey mechanism is not isolated.
4. Establish a timestamped historical pregame starter and lineup source, then test goalie and lineup effects on these same chronological folds. Do not infer their value from postgame identities.
5. Add deeper stability cuts for teams, predicted total-goal environment, rest and schedule density before any candidate promotion or live export.
