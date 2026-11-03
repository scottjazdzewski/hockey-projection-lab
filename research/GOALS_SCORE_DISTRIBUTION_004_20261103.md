# Goals and Score Distribution Model 004 — OT-Constrained Selection

**Experiment:** EXP-0022  
**Model ID:** `HOCKEY-LAB-GOALS-DISTRIBUTION-004-RESEARCH`  
**Status:** RESEARCH  
**Run date:** 2026-11-03  
**Champion protection:** Hockey Lab Game v1.0.0 remains frozen. No HWIN or betting-market inputs. No website export.

## Question and predeclared rule

EXP-0021 found that validation selection on official full-game exact-score likelihood improved that likelihood in all five test seasons, but sharply reduced predicted OT and worsened OT Brier. This experiment asked whether a validation-only OT-calibration constraint could keep the full-score objective from sacrificing OT quality.

For each target season S, candidates are score families and parameters evaluated on S−1. Eligible candidates must have validation OT Brier no worse than a constant forecast equal to the S−1 OT rate. Among eligible candidates, choose the one with the lowest validation official full-game exact-score log loss. If no candidate qualifies, select the validation OT-Brier minimum and record the fallback. Validation OT predictions are regulation tie mass, and validation OT winner probabilities use a home OT rate estimated from training seasons only.

The model was then scored on S, with score family and parameter fixed before looking at S. The same 6,560 target games across seasons 2022–2026 were used. These are development OOF folds, not an untouched promotion test.

## Results

| Metric | EXP-0020: select by regulation score LL | EXP-0022: OT-constrained full-score selection |
|---|---:|---:|
| Full-game exact-score log loss | 3.7136 | **3.7116** |
| Full-game total-goal log loss | **2.17984** | 2.17990 |
| Regulation outcome Brier | 0.62564 | **0.62551** |
| OT Brier | 0.17320 | **0.17313** |
| Predicted / observed OT rate | 21.6% / 22.2% | 20.9% / 22.2% |

The constant-rate OT Brier is about 0.1729, so even the constrained candidate remains slightly worse than that simple benchmark. Only two of the five validation seasons had any candidates satisfy the OT-Brier constraint; in the other three, the predeclared OT-Brier fallback was used. The aggregate changes from EXP-0020 are very small and do not represent a clear production improvement.

## Decision

Keep the constrained selector at **RESEARCH**. It avoided the large OT underprediction of unrestricted full-score selection, but the constraint was feasible in only two folds and the out-of-sample gains over EXP-0020 are negligible. The model still does not beat OT-rate climatology, and the total-score objective remains essentially unchanged.

This experiment changed only the score distribution. It did not change expected-goal means. The count layer still underpredicts final totals by roughly 0.186 goals per game, and no live 2026–27 inference/export path or timestamped pregame goalie/lineup history exists. Do not publish goals outputs. Hockey Lab Game v1.0.0 remains the frozen full-game win-probability champion.

## Next focus

Return to the team-goal rate layer: test opponent-adjusted attack/defense shrinkage and a training-only scoring-rate level adjustment, with a predeclared objective that scores home/away expected goals, total-goal likelihood, and outcome probabilities together. Keep the same chronological folds. After rates improve, revisit joint score dependence and add goalie/lineup only when time-stamped pregame data can be reconstructed legally.
