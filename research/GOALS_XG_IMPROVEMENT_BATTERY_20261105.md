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
| 2 | Process-component ablation | Which of 5v5 xG for/against, shots, power-play xG/goals, penalties, recent goals, and schedule inputs adds unique value? | **Next:** remove/add one coherent component at a time; choose regularization only on S−1; test S; assess log loss, MAE, score probabilities and win Brier. Require stable gains across seasons. |
| 3 | xG calibration and conversion persistence | Do recent xG levels predict goals, and does goals-minus-xG finishing persist enough to help future rates? | Open. Test team attack and opponent defense separately, partial pooling, and no same-game data. |
| 4 | Form-window and decay sensitivity | Is 60 games too slow or noisy? | Open: 20/40/60/100 games, season-to-date, and exponentially weighted forms in identical rolling folds. Tune only on the prior season. |
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

## Limits and reproducibility

The OOF predictions are from the recovered 2026-11-04 dataset rebuild. Its Parquet hashes differ from the frozen EXP-0027/0028 inputs, so results are snapshot reruns rather than exact source-input reproductions. The confidence intervals are paired, game-resampled intervals stratified by five target seasons; they do not include uncertainty from choosing among many feature groups. The ablation re-scores the existing OOF variants; the next component test must retrain and retune inside each rolling fold to avoid optimistic feature selection.

Code and metric outputs are stored beside this report under `research/` and `research/exports/GOALS_XG_BATTERY_20261105/`.
