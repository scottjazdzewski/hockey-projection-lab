# NHL Game Model: B2B Interaction Test

**Experiment:** HWIN B2B Interactions 012  
**Status:** RESEARCH; no model promoted  
**Run date:** 2026-11-03  
**Primary test season:** 2025–26 (`source_season=2026`)  
**Chronological evaluation:** 2022–2026; 6,993 games per variant

## Question and design

Does the effect of back-to-back fatigue change when both teams are on a back-to-back, or when the Elo strength gap is different for the home or away team on short rest?

The reference is the unchanged Team Strength 004 + Schedule 007 B2B model. It already includes separate home and away B2B indicators (plus their difference), so this test focused on nonlinear additions rather than repeating those main effects. Three predeclared candidates added: (1) a both-teams-on-B2B interaction; (2) interactions between pregame Elo logit and home/away B2B status; or (3) both groups together.

Each target-season fold trained through S-2, selected regularization on S-1, and scored S. The same fold-specific Elo and opponent-adjustment settings were used as in Schedule 007. There were no goalie, lineup, market, or same-game outcome predictors. Schedule and Elo features were calculated from information available before the target game. These seasons have already been inspected in prior research, so they remain development diagnostics rather than a final untouched test.

## Results across five seasons

| Variant | Accuracy | Brier | Log loss | Correct picks vs reference | Brier change vs reference (95% paired interval) |
|---|---:|---:|---:|---:|---:|
| Team Strength 004 + B2B | **59.92%** | **0.234726** | **0.661540** | — | — |
| + both-teams B2B interaction | 59.83% | 0.234752 | 0.661597 | −6 | +0.000026 [−0.000023, +0.000077] |
| + Elo gap × home/away B2B | 59.55% | 0.234756 | 0.661608 | −26 | +0.000030 [−0.000103, +0.000165] |
| + both candidate interaction groups | 59.67% | 0.234772 | 0.661646 | −17 | +0.000046 [−0.000098, +0.000187] |

The paired intervals include zero. Each interaction set improved Brier in only **2 of 5** target seasons. The Elo-by-side variants lost accuracy across the pooled folds; no candidate improved all three primary metrics.

## 2025–26 result

The both-teams interaction had a tiny probability-score improvement: Brier moved from 0.245150 to 0.245137 and log loss from 0.683085 to 0.683060. Its paired Brier change was −0.000013, with a 95% interval of **[−0.000042, +0.000017]**. It selected two fewer winners correctly: 766 of 1,394 (54.95%) versus 768 (55.09%).

Adding Elo-gap-by-side interactions did worse: the home-win Brier rose to 0.245274 and accuracy fell to 54.30% (11 fewer correct picks). Combining both interaction groups scored 0.245251 Brier and 54.38% accuracy (10 fewer picks). These are not useful improvements.

## Context diagnostic

The 2025–26 reference accuracy varied by schedule state: 59.39% for away-only B2B games (197), 53.93% for home-only (89), 51.90% when both teams were on B2B (79), and 54.62% when neither was (1,029). The both-teams interaction changed probability scores slightly in these groups but did not add correct winner picks in the B2B-only groups. The smaller groups remain noisy.

Across the pooled predictions, calibration was already close to neutral for the reference (intercept −0.017, slope 0.998). The both-teams candidate was nearly unchanged (−0.017, 0.997); the added terms did not solve a calibration problem.

## Decision

**Do not promote the interactions.** The both-teams term is a plausible small signal worth retaining as a documented research result, but its pooled scores worsen slightly, its apparent 2025–26 Brier improvement is within noise, and it adds no correct picks in that season. The Elo-gap interactions reduce winner accuracy. Keep the current B2B reference and frozen live shadow unchanged.

Next on the research queue is the predeclared tied-state xG window comparison. The timestamped pregame goalie-starter workflow remains parked until that test is complete.
