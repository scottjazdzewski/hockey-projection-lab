# Hockey Goals / xG Improvement Battery — 2026-11-05

**Scope:** Research evaluation of the fixed-60 NHL team-goal model and the score probabilities derived from its expected goal rates. This does not change Hockey Lab production, the existing win model, public cards, or market inputs.

## How close is the model?

The best available evidence is five rolling held-out regular seasons (2022–2026, 6,560 games) on the rebuilt 2026-11-04 snapshot. For the current hockey-process + league-scoring independent-Poisson reference, team-goal MAE is **1.372 goals per team-game**, final-total MAE is **1.852 goals per game**, and full-game home-win Brier is **0.2352**. The home-win Brier is 0.0135 lower than a simple constant-prevalence benchmark (0.2486, computed on the same held-out outcomes); that is useful signal, but it is not a comparison with a closing market and the baseline is descriptive.

This is **a promising research candidate, not “done” or ready to call reliable for wagering**. Its win-probability skill varies by season, its goal-rate predictions still have material game-level error, and the existing data rebuild does not exactly match the frozen source hashes. Preserve the existing snapshot caveat and require a prospective 2026–27 ledger before any promotion decision.

## Battery: ordered tests

| Priority | Test | Question answered | Status / decision rule |
|---:|---|---|---|
| 0 | Input, labels, timestamp and leakage audit | Are every feature and label valid as of puck drop? | Partly passed for the rebuilt snapshot; frozen EXP-0027 source-hash mismatch remains documented. Recheck every new feature. |
| 1 | Paired rolling-origin feature ablation | Does a feature group improve next-season predictions, not just validation fit? | **Started:** existing OOF feature variants rescored over 2022–2026, with paired season-stratified bootstrap intervals. |
| 2 | Process-component ablation | Which of 5v5 xG for/against, shots, power-play xG/goals, penalties, recent goals, and schedule inputs adds unique value? | **Completed first pass:** removed one coherent family at a time and retrained across all five rolling folds. Component-specific xG evidence is mixed; shots, recent goals, opponent adjustments, and schedule inputs show stronger support. |
| 3 | xG calibration and conversion persistence | Do recent xG levels predict goals, and does goals-minus-xG finishing persist enough to help future rates? | **Component isolation complete:** team 5v5 attack xG and team 5v5 xG-against help expected-goal MAE modestly; opponent-side 5v5 xG-against, power-play xG, and opponent-adjusted xG add little unique value. Next test separate time windows for attack and defense xG. |
| 4 | Form-window and decay sensitivity | Is 60 games too slow or noisy? | **First pass complete:** 20/40/60/100-game windows retrained in identical rolling folds; 60 games leads. Exponential decay and xG-only windows remain open. |
| 5 | Matchup nonlinearity and interactions | Do attack-versus-defense, home/away, pace, special teams, or schedule-state interactions help? | Open after component ablation; test predeclared interactions with shrinkage to limit overfit. |
| 6 | Residual map | Where and when is error concentrated? | Partial diagnostics exist. Extend by expected-goal decile, team strength, home/away, rest, 3-in-4, 4-in-6, and strictly pregame getaway label; report sample sizes and season stability. |
| 7 | Score-law and tie-cell calibration | Are marginal goal rates right but joint score/tie probabilities wrong? | Candidate score families were screened on prior validation seasons; exact 0–0/1–1/2–2 diagonal probability calibration remains open. |
| 8 | Probability calibration and decision stability | Are win probabilities calibrated, especially at extremes and by season? | Partial OOF diagnostics exist; check reliability, calibration slope/intercept, Brier/log loss, and uncertainty by season before any probability-layer change. |
| 9 | Prospective shadow ledger | Does a frozen, pregame forecast hold up on unseen games? | Open and required before promotion. Freeze inputs, timestamps, code version, and metrics; do not revise against outcomes already observed. |

**Promotion gate:** no single season or one metric is enough. A candidate must improve the primary proper scoring rule on held-out games, avoid material regression in goal MAE and win probabilities, show directionally stable results across seasons, survive paired uncertainty checks, and then pass the prospective ledger. Markets remain benchmark-only and are never predictors.

## First test result: rolling held-out xG feature comparison

All variants below are existing rolling-origin OOF predictions. Each target season used parameters selected on the immediately preceding season. Pooled comparisons cover 6,560 games; paired uncertainty resamples games within each target season (10,000 draws). Negative candidate-minus-baseline values are better.

| Variant | Team goal MAE | Team count log loss | Final-total MAE | Final-score log loss | Full home-win Brier |
|---|---:|---:|---:|---:|---:|
| Recent goal form | 1.3833 | 1.9253 | 1.8541 | 3.7028 | 0.2382 |
| Goal form + recent xG | 1.3782 | 1.9264 | 1.8555 | 3.7083 | 0.2376 |
| 5v5 xG pace only | 1.3845 | 1.9360 | 1.8753 | 3.7304 | 0.2395 |
| Full hockey-process bundle | **1.3724** | **1.9216** | 1.8521 | **3.6978** | **0.2352** |

Adding recent xG to goal form lowers team-goal MAE by 0.0051 (paired 95% interval −0.0075 to −0.0027) and improves full-game home-win Brier by 0.00053 (−0.00085 to −0.00022), but worsens team count log loss by 0.0011 (−0.0003 to +0.0025) and final-score log loss by 0.0055 (+0.0027 to +0.0084). So the narrow xG addition is **mixed**, not a clean upgrade. The 5v5 xG pace-only feature set is clearly worse than goal form on count and score log loss.

The full hockey-process bundle improves over goal form by 0.00366 team-goal log loss (95% interval −0.00565 to −0.00162), 0.0109 team-goal MAE (−0.0142 to −0.0075), 0.00505 final-score log loss (−0.00902 to −0.00096), and 0.00302 home-win Brier (−0.00420 to −0.00183). The bundle is better on count log loss in 4/5 seasons and on home-win Brier in 5/5. Because it combines xG, shots, special teams, goals, and schedule features, this result **does not identify xG as the source of the gain**. That is the reason to run the component ablation next.

The extra league-scoring feature adds almost nothing to the full bundle (team count log-loss delta −0.00005; interval −0.00020 to +0.00010). Keep it only if later tests show it helps in more targeted scoring-regime checks.


## Second test result: retrained process-component ablation

Retrained each variant in all five rolling folds. Every target season had 1,312 games. The full process reference was compared against models with one feature family removed; negative ablation-minus-full values mean the removed-feature model did better.

| Removed feature family | Team-goal log-loss delta | Team-goal MAE delta | Final-score log-loss delta | Home-win Brier delta | Read |
|---|---:|---:|---:|---:|---|
| All xG | −0.00073 (−0.00148, +0.00001) | +0.00117 (−0.00013, +0.00250) | −0.00252 (−0.00400, −0.00103) | +0.00004 (−0.00033, +0.00041) | Mixed, small; removal improves exact score log loss, while goal MAE slightly worsens but is uncertain. |
| 5v5 xG only | −0.00040 (−0.00094, +0.00013) | +0.00133 (+0.00041, +0.00228) | −0.00175 (−0.00281, −0.00069) | +0.00003 (−0.00017, +0.00023) | Helps goal MAE a little; not a stable overall scoring-probability gain. |
| Power-play process | −0.00021 (−0.00050, +0.00009) | −0.00028 (−0.00081, +0.00026) | −0.00034 (−0.00093, +0.00025) | −0.00012 (−0.00028, +0.00004) | No measurable unique gain yet; intervals include zero. |
| Shot volume | +0.00273 (+0.00144, +0.00404) | +0.00392 (+0.00168, +0.00619) | +0.00569 (+0.00311, +0.00831) | +0.00168 (+0.00078, +0.00257) | Strong, stable contribution. |
| Recent goal form | +0.00815 (+0.00621, +0.01005) | +0.00501 (+0.00174, +0.00814) | +0.01868 (+0.01485, +0.02247) | +0.00217 (+0.00127, +0.00304) | Strongest tested contribution. |
| Schedule inputs | +0.00076 (+0.00016, +0.00136) | +0.00206 (+0.00099, +0.00312) | +0.00178 (+0.00057, +0.00296) | +0.00079 (+0.00053, +0.00104) | Small but consistent held-out contribution. |
| Opponent adjustments | +0.00200 (+0.00092, +0.00308) | +0.00352 (+0.00166, +0.00537) | +0.00400 (+0.00187, +0.00613) | +0.00108 (+0.00035, +0.00178) | Strong, stable contribution. |

Parentheses show paired 95% intervals from 10,000 game-resamples stratified by target season. The direct message is that **recent goals, opponent adjustments, shots, and schedule carry clearer incremental value than current xG inputs**. That does not mean xG should be deleted: 5v5 xG improves team-goal MAE slightly, but it has not earned a reliable improvement in the full score or win-probability distribution. Power-play inputs also need narrower feature tests before we conclude they are useless.

This is a family-level ablation; correlated features can substitute for one another, so it cannot identify a single coefficient as causal. The next useful xG test is to compare its calibrated conversion and persistence over different windows, then test 5v5 and power-play xG separately against goals/shot inputs in nested folds. The schedule family should be tested next for 3-in-4, 4-in-6, B2B interaction, and a strictly pregame getaway definition.


## Third test result: rolling-window sensitivity

Retrained the full process feature set with 20-, 40-, 60-, and 100-game windows. For every window and target season, regularization and score-family choice were made using only the previous season. These are paired comparisons over the same 6,560 held-out games. This changes the rolling window for **all** team-form features together; xG-specific window behavior is still to be isolated.

| Window | Team-goal log loss | Team-goal MAE | Final-total MAE | Final-score log loss | Home-win Brier |
|---:|---:|---:|---:|---:|---:|
| 20 | 1.92422 | 1.37473 | 1.85432 | 3.71356 | 0.23670 |
| 40 | 1.92262 | 1.37376 | 1.85326 | 3.71007 | 0.23590 |
| **60** | **1.92155** | **1.37246** | **1.85170** | **3.70791** | **0.23536** |
| 100 | 1.92312 | 1.37435 | 1.85556 | 3.71228 | 0.23550 |

Compared with 60, the 20-game window worsens team-goal log loss by 0.00266 (paired 95% interval +0.00123 to +0.00409), final-score log loss by 0.00565 (+0.00276 to +0.00853), and home-win Brier by 0.00135 (+0.00064 to +0.00205). The 40-game window is also worse on those three measures, though by less. The 100-game window worsens team-goal log loss by 0.00157 (+0.00051 to +0.00263) and final-score log loss by 0.00436 (+0.00225 to +0.00645). The 60-game window has the best pooled results; season-level winners vary (60 wins 4/5 seasons on team-goal log loss and 3/5 on final-score log loss and home-win Brier). **Keep 60 as the current overall-form benchmark**, while retaining that season variation in the record.

This does not prove 60 is optimal for xG itself; correlated goal, shot, special-team, and xG features all changed windows together. Continue with an xG-specific window/conversion test before changing the feature pipeline.


## Fourth test result: xG-specific windows and conversion residuals

This test held the non-xG model inputs at their current 60-game form and changed only the rolling 5v5/power-play xG features to 20 or 100 games. A second set added strictly prior-date goals-minus-xG and opponent goals-allowed-minus-xG-against residual features at 20, 60, or 100 games. Every candidate was retrained in each rolling fold, with alpha and score-law choice made on the preceding season only.

The 20-game rolling xG variant is the best lead so far for the **average goal-rate shortfall**: regulation-goal underprediction shrank from 0.125 to 0.077 goals per game, and final-total underprediction shrank from 0.185 to 0.138. Final-score log loss improved by 0.00162 (paired 95% interval −0.00251 to −0.00073), with a very small team-count log-loss improvement. But team-goal MAE worsened by 0.00055 (interval −0.00022 to +0.00134); home-goal MAE worsened by 0.00169 ( +0.00051 to +0.00287), while away-goal MAE improved by 0.00058 with an interval that includes zero. Final-total MAE improved by only 0.00055, also uncertain. **This narrows bias but has not yet shown a reliable improvement in expected-goal accuracy.**

The 100-game xG variant was neutral to slightly worse. Adding rolling goals-minus-xG conversion residuals did not reliably improve the 20-game xG candidate. Added to the 60-game reference, conversion features barely moved team-goal MAE or score log loss; they reduced home MAE slightly but did not yield a dependable overall gain. Do not add a global scoring offset or promote these features yet.

The practical next step is to split the promising 20-game xG signal into its components: 5v5 attack xG, 5v5 xG allowed, power-play xG, and the opponent's defensive/goaltending residual. Keep goal MAE and count log loss as primary measures of expected-goal accuracy, while retaining score log loss as a secondary check. Inspect each season before selecting any candidate, because the pooled improvements are small.


## Fifth test result: xG input isolation

Removed one xG input family at a time while keeping the other xG signals, shots, recent goals, schedule, and opponent features fixed. Each version was retrained in all five rolling folds; negative ablation-minus-full values mean removing the feature improved the metric.

| Removed input | Team-goal MAE change | Home-goal MAE change | Final-score log-loss change | Interpretation |
|---|---:|---:|---:|---|
| Team 5v5 attack xG | +0.00027 (+0.00004, +0.00049) | +0.00058 (+0.00023, +0.00091) | −0.00032 (−0.00058, −0.00006) | Helps expected-goal error modestly; score-law fit slightly prefers removal. |
| Team 5v5 xG against | +0.00058 (+0.00009, +0.00109) | +0.00136 (+0.00055, +0.00215) | −0.00096 (−0.00152, −0.00039) | Helps expected-goal MAE, especially home; score-law fit again trades the other way. |
| Opponent 5v5 xG allowed | −0.00006 (−0.00015, +0.00002) | −0.00005 (−0.00017, +0.00008) | −0.00002 (−0.00012, +0.00009) | No detectable unique benefit at current window. |
| Power-play xG for/against | −0.00015 (−0.00063, +0.00032) | +0.00001 (−0.00068, +0.00069) | −0.00036 (−0.00088, +0.00016) | No detectable unique benefit in this feature form. |
| Opponent-adjusted xG | −0.00019 (−0.00074, +0.00033) | −0.00009 (−0.00086, +0.00067) | −0.00027 (−0.00088, +0.00035) | No detectable unique benefit after the other rolling inputs. |
| All rolling xG | +0.00125 (+0.00016, +0.00237) | +0.00268 (+0.00097, +0.00435) | −0.00234 (−0.00359, −0.00106) | Rolling xG helps goal-rate MAE but hurts exact score log loss. |

Values are ablation minus the full 60-game reference, with paired 95% intervals from 10,000 game-resamples stratified by target season. These are small effects and many related candidates were examined, so treat them as leads for another holdout, not final proof. The useful distinction is that **5v5 team process inputs improve the expected-goal mean**, while probability-of-exact-score calibration remains a separate issue. The next test changes the 5v5 attack and defense xG windows independently.

## Limits and reproducibility

The OOF predictions are from the recovered 2026-11-04 dataset rebuild. Its Parquet hashes differ from the frozen EXP-0027/0028 inputs, so results are snapshot reruns rather than exact source-input reproductions. The confidence intervals are paired, game-resampled intervals stratified by five target seasons; they do not include uncertainty from choosing among many feature groups. The ablation re-scores the existing OOF variants; the next component test must retrain and retune inside each rolling fold to avoid optimistic feature selection.

Code and metric outputs are stored beside this report under `research/` and `research/exports/GOALS_XG_BATTERY_20261105/`.
