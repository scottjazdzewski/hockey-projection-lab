# Goals and Score Distribution Model 003 — Selection Objective Test

**Experiment:** EXP-0021  
**Model ID:** `HOCKEY-LAB-GOALS-DISTRIBUTION-003-RESEARCH`  
**Status:** RESEARCH  
**Run date:** 2026-11-03  
**Champion protection:** Hockey Lab Game v1.0.0 remains frozen. No HWIN or betting-market inputs. No website export.

## Question

The EXP-0020 score law was selected on regulation exact-score likelihood. Does selecting the score family and parameter on full-game score likelihood or total-goal likelihood improve the out-of-sample objective without losing the requested regulation and overtime calibration?

## Design

This experiment keeps the EXP-0020 team-goal rates and five score-law families fixed. For each rolling-origin fold, it evaluates every family/parameter on the previous season and independently selects by:

1. regulation exact-score log loss;
2. official full-game exact-score log loss; or
3. official full-game total-goal log loss.

When evaluating validation full-game outcomes, the OT home-win rate is estimated from training seasons only. The selected distribution and parameter are then frozen, count rates are refit through the previous season, and the target season is scored. Source seasons 2022–2026 contribute 1,312 games each, 6,560 games total. These remain development OOF folds, not an untouched promotion test.

The score model is independent from the expected-goal rate layer. All three selectors therefore have the same expected-goal forecasts; only the score matrix and probabilities derived from it change.

## Results

| Selection rule | Regulation score LL | Full-game score LL | Full-game total LL | Regulation 3-way Brier | OT Brier | Predicted OT rate |
|---|---:|---:|---:|---:|---:|---:|
| Select on regulation score LL | **3.8375** | 3.7136 | **2.1798** | **0.6256** | **0.1732** | 21.6% |
| Select on full-game score LL | 3.8649 | **3.6997** | 2.1835 | 0.6365 | 0.1805 | 13.5% |
| Select on full-game total LL | 3.8428 | 3.7157 | 2.1810 | 0.6279 | 0.1747 | 21.8% |

Observed OT rate is 22.2%. Constant-rate OT Brier is approximately 0.1729. The final-score selector improved official exact-score log loss in all five target seasons versus the regulation-score selector, but it materially reduced tie/OT mass and worsened OT Brier in all five seasons. The total-goal selector did not beat the existing selector's total-goal likelihood and also had worse outcome metrics.

The selected full-game-score model's final exact-score log loss is better than the EXP-0019 and EXP-0020 alternatives, but that single gain comes with worse regulation outcomes, worse OT calibration, and worse total-goal log loss. It is not a production improvement across the required outputs.

## Decision

No selector is promoted. Keep the EXP-0020 regulation-score selector as the better current research candidate for regulation outcome and OT behavior, while retaining the final-score selector as a useful diagnostic alternative. Both remain **RESEARCH**. The goal-rate means still underpredict total scoring; expected-goal forecasts are not changed by this selector experiment.

Do not publish either result to the Game Cards. Hockey Lab Game v1.0.0 remains the full-game win-probability champion. Goals outputs still lack a live 2026–27 inference path and timestamped pregame goalie/lineup data.

## Next experiment

Predeclare a constrained selection rule: among score laws whose prior-validation OT Brier does not exceed the prior-validation constant-rate OT Brier, select the one with the best prior-validation full-game exact-score log loss. Evaluate that rule only on the next chronological target season. If no candidate satisfies the constraint, fall back to the validation OT-Brier minimum and record the fallback. This tests whether full-score likelihood can improve without spending the OT calibration error observed here.
