# Model Accuracy Test Battery 001

**Date:** 2026-11-04  
**Status:** Research protocol; no production model change  
**Scope:** Team win probabilities, goalie adjustments, and score/total-goal distributions

## Goal

Every candidate must answer one measurable question: does it improve a pregame forecast on games it did not train or tune on? Keep the win-probability test and the score-distribution test separate. A better score distribution can improve win probabilities derived from it, but it does not prove the independently fitted HWIN forecast improved.

## Current reference and evidence

The active HWIN comparison baseline is `team-strength-004+b2b-007`. Goalie version `prospective-0.3.0` adds an age- and exposure-shrunk save-percentage adjustment and is being evaluated prospectively. Historical identity tests that select the actual game starter after the game are retrospective upper bounds; only timestamped projected or confirmed starter snapshots can establish pregame value.

| Existing test | Result on available data | Decision |
|---|---|---|
| HWIN backbone plus B2B | 6,993 rolling-origin games, 2022–2026: Brier 0.234726, log loss 0.661540, accuracy 59.92% | Keep as the comparison baseline |
| Current goalie adjustment | Weighted Brier lift 0.000057 and log-loss lift 0.000084; improved in 3/5 folds, latest fold worsened | Keep under prospective review |
| Goalie metric audit | Shrunk SV%: ΔBrier −0.000141, Δlog loss −0.000280; age-adjusted GAA alone had a tiny uneven lift; SOG-GSAx and combinations with SV% worsened aggregate scores | Keep SV% as the only active goalie input |
| B2B interaction screen | The three tested interaction variants slightly worsened pooled Brier and had uncertainty intervals crossing zero | Do not add these interactions |
| Nested goal-window selection (010) | On 6,560 OOF games, final totals were underpredicted by 0.158 goals/game; over 5.5 was 53.70% vs 57.16% observed, and over 6.5 was 43.25% vs 46.17% | Prioritize rate and upper-tail calibration |

The first forward goalie snapshots are the decisive evidence still being collected. Historical goalie results above do not establish the live starter-change effect.

## Battery

### 0. Data and leakage gate

Run before scoring any candidate.

- Verify one canonical game row, two team sides, one verified winner, valid home/away orientation, and no duplicate game-side keys.
- Rebuild every rolling feature from records strictly before puck drop. Apply a same-day ordering guard; suppress features when timestamps cannot establish order.
- Record source versions, input cutoffs, run IDs, row counts, and hashes.
- Exclude current-game actual starter, postgame box statistics, market odds, and corrupted schedule score fields from pregame features.
- For live goalie joins, resolve ID first and use only a unique normalized name alias as fallback. Preserve status, timestamps, and unavailable reason.

**Fail condition:** any duplicate target, unresolved label, postgame field, or future-dated feature invalidates that run.

### 1. HWIN feature and goalie ablations

Use identical games and nested chronological folds. For target season S, train through S−2, tune only on S−1, then score S. Keep the same frozen team-only baseline in every comparison.

Predeclare these candidates:

- Team-only reference; reference plus B2B.
- Goalie SV% only; GAA only; SOG-based GSAx only; then SV%+GAA and SV%+GSAx.
- Tied-state xG at 10, 20, and 40 prior games with one fixed shrinkage option.
- The three already screened B2B interaction forms, retained as diagnostics unless a better hypothesis is preregistered.

Report Brier and log loss as primary measures. Also report accuracy, calibration intercept/slope, reliability bins, and fold-level deltas. Do not select on accuracy alone.

### 2. Goalie A-versus-B counterfactual

For each game, freeze team strength and every non-goalie input. Substitute goalie A and goalie B into the same calculation and report:

- The probability and fair-moneyline change in percentage points.
- The goalie rating difference, age, starts, shots, minutes, shrinkage reliability, and data timestamp.
- Whether the change improves the held-out forecast once the result is known.
- Results by goalie-quality gap and exposure tier.

This separates the model's mechanical response to a goalie change from evidence that the response is calibrated. Run a retrospective version using actual starters only as an upper-bound diagnostic. The primary test uses the timestamped projected starter, followed by a confirmed-starter update captured before puck drop.

### 3. Goalie aging and exposure calibration

Fit the age curve on prior seasons only. Within each fold's validation season, select the exposure shrinkage for starts, shots, and minutes from a small predeclared grid. Test SV%, GAA, and GSAx separately before testing combinations. Report cold-start goalies, low-exposure goalies, established starters, and age bands separately.

Keep the current 400-shot / 15-start / 1,800-minute settings as the control. Do not hand-code an aging curve or promote a setting because it wins on one season.

### 4. Calibration and error slices

For each frozen out-of-fold set, report calibration and proper scores by:

- Target season and month.
- Home/away and regular season/playoffs.
- Favorite probability: 50–55%, 55–60%, 60–65%, and 65%+.
- B2B state, strength gap, goalie-rating gap, goalie exposure, and starter status.
- Expected scoring environment when that pregame estimate is available.

Show sample counts beside each slice and suppress interpretations of sparse groups. If testing recalibration, fit it only on the validation season and apply it unchanged to the next target season. Keep the uncalibrated score as the paired control.

### 5. Score and total-goal distribution

Use the same nested folds, then compare team-goal rate models and score laws on the held-out season. Candidate families already represented in research include independent Poisson, negative binomial, Dixon–Coles, bivariate Poisson, and diagonal-inflated Poisson. Test only a small predeclared set of team-history windows and scoring-environment offsets.

Report home and away regulation-goal MAE/bias, joint regulation-score log loss, regulation outcome Brier/log loss, final-total log loss, over/under Brier at 4.5/5.5/6.5, and overtime calibration. Focus first on the observed high-total underprediction. Tune every window, offset, and distribution parameter on the prior validation season only.

### 6. Prospective goalie ledger

At each slate, preserve immutable snapshots for the initial projected starter, any confirmed-starter change, and the last pregame state. Store both team-only and goalie-adjusted probabilities, model/data versions, timestamps, and fair moneylines. After games finish, append verified results without rewriting forecasts.

Score the goalie-adjusted forecast against its paired team-only probability using Brier and log loss. Report calibration and results by goalie-change size and exposure. Before opening the confirmatory evaluation, use the historical paired variance to calculate the number of games needed to detect the minimum useful improvement. Keep tracking until that sample is reached.

## Uncertainty and decision rules

- Compare models only on the same game rows and use paired per-game score differences.
- Report 95% paired intervals and season-by-season sign consistency. Add team-season block resampling to reflect repeated teams; with only five target seasons, show the interval and do not treat it as the sole decision rule.
- A candidate can stay in research when results are mixed. A strong historical screen should improve both Brier and log loss overall, improve in at least four of five target seasons, and avoid material calibration harm.
- A candidate is not a production improvement until leakage checks pass and the prospective test supports the same direction. A goalie A-versus-B scenario by itself is not predictive evidence.
- Record rejected candidates too, so future runs do not repeat unhelpful searches.

## Run artifacts

Each battery run should package:

1. A manifest with input versions, cutoffs, hashes, code version, and fold design.
2. Immutable game-level out-of-fold predictions.
3. Fold, season, reliability, and subgroup metrics with counts.
4. Paired uncertainty intervals and a concise decision report.

## Next execution order

1. Add paired uncertainty and goalie-exposure slices to the existing SV%/GAA/GSAx comparison.
2. Re-run the score-distribution screen focused on total-goal bias and 5.5/6.5 tails.
3. Continue the timestamped projected-to-confirmed goalie forward ledger; do not tune against its outcomes until the predeclared sample is complete.
4. Revisit tied-state xG windows only after these runs, because its prior pooled lift was small and its paired interval crossed zero.
