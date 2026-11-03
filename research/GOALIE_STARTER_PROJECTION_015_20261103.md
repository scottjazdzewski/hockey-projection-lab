# Goalie Starter Projection 015 — Rotation Baseline

**Run date:** 2026-11-03  
**Status:** RESEARCH diagnostic; not production eligible  
**Experiment ID:** `HOCKEY-LAB-GOALIE-STARTER-PROJECTION-015`  
**Champion protection:** Hockey Lab Game v1.0.0 remains frozen; no public/site exports changed.

## Question

Can we estimate the likely starting goalie using only schedule and goalie-start history available before the game, without relying on a same-game postgame starter flag as an input?

## Data and leakage controls

- Canonical game spine V3, dated 2026-11-02, and goalie game features V1, dated 2026-11-02, from the canonical Hockey Lab Drive warehouse.
- Historical completed games span 2022-10-07 through 2026-04-16; the forward 2026-27 slice spans 2026-09-29 through 2026-11-01.
- Actual starter identity (`starter_bool`) was used only as the outcome label. Features were built from prior team starts and the schedule gap to the previous team game. No same-game goalie box fields, goalie decisions, scores, market data, or future starts entered a projection.
- One latest-starter history row was skipped per team where no prior starter history existed. Full dataset: 10,925 scored team-sides; current-season slice: 462 team-sides (231 games).
- This test evaluates starter identity only. It does not yet evaluate projected-goalie expected goals or game win probabilities.

## Rules compared

1. **Previous starter:** repeat the goalie who started the prior team game.
2. **Five-game rotation leader:** choose the goalie with most starts among the previous five team games; break ties by the most recent start.
3. **Ten-game rotation leader:** same rule over ten games.
4. **Back-to-back rotation rule (candidate):** if the team played the previous calendar day, choose the most-started goalie other than the previous game's starter over the previous ten team games; otherwise use the five-game rotation leader. Ties go to the most recent start.

The B2B flag is a calendar-date proxy derived from schedule dates. It should be verified against exact start times before a production implementation.

## Top-one starter accuracy

| Rule | Historical team-sides | Accuracy |
|---|---:|---:|
| Previous starter | 10,925 | 42.1% |
| Five-game rotation leader | 10,925 | 57.2% |
| Ten-game rotation leader | 10,925 | 56.4% |
| Back-to-back rotation rule | 10,925 | **64.6%** |

On the historical back-to-back subset (1,597 team-sides), the previous-starter rule was correct 7.8%, the five-game leader 29.3%, and the back-to-back rotation rule 78.5%. The rule was selected after comparing historical results, so this historical subgroup score is exploratory and selection-biased; the 2026-27 slice below is the forward check. This supports treating a goalie who started the previous day as unlikely to start again, while accounting for which alternative goalie has recently carried the workload.

## Forward 2026-27 slice

| Rule | Correct / scored sides | Accuracy |
|---|---:|---:|
| Previous starter | 186 / 462 | 40.3% |
| Five-game rotation leader | 269 / 462 | 58.2% |
| Back-to-back rotation rule | 293 / 462 | **63.4%** |

The B2B rotation rule was correct on 36 of 53 scheduled back-to-back team-sides (67.9%). The actual starter appeared among the goalies seen in the prior 20 team starts for 435/462 sides (94.2%), so the remaining 5.8% represent an important new/returning-goalie coverage problem. This early-season sample is encouraging but small.

## Interpretation and limitations

The strongest simple signal is goalie rotation conditioned on schedule density. “Repeat the last starter” is a poor NHL rule, particularly on back-to-backs. A rotation projection is feasible from the existing historical warehouse and does not require scraping a third-party starter board.

These are top-one classifications, not calibrated starter probabilities. The candidate rule also does not know who is currently healthy, rostered, or announced by the club. The 2026-27 forward slice is only 231 games. Do not convert the accuracy gain directly into goal or win-probability adjustments.

## Next experiment

1. Generate probability distributions across plausible goalies, and calibrate them on rolling historical seasons; report multiclass log loss, Brier score, top-one accuracy, and coverage, including back-to-back and new-goalie cases.
2. Use those probabilities to form a mixture of goalie-specific score distributions, rather than selecting one goalie with certainty. Compare expected-goal error, full score log loss, regulation-win Brier/log loss, and overtime calibration against the no-goalie model and the retrospective actual-starter upper bound.
3. Keep the projection output in shadow/research. Add confirmed-starter updates only from a lawful, timestamped source or a manually captured, attributable pregame record. Promote only if the pregame-valid candidate improves outcomes out of sample.

## Decision

**Keep as RESEARCH.** The rotation rule improves starter top-one accuracy in the early 2026-27 forward slice, but probability calibration and downstream forecast improvement have not yet been demonstrated. No change to Hockey Lab Game v1.0.0 or any website output.
