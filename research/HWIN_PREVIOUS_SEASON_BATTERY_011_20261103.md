# NHL Game Model: 2025–26 Previous-Season Battery

**Experiment:** HWIN Previous-Season Battery 011  
**Status:** RESEARCH; no model promoted  
**Run date:** 2026-11-03  
**Previous completed season:** 2025–26 (`source_season=2026`)  
**Data:** canonical game spine V3, canonical SportsDataverse/fastRhockey PBP features, and goalie game features, retrieved 2026-11-02/03.

## Design

The current no-goalie reference is Team Strength 004 + Schedule 007 B2B. The battery re-ran the existing chronological experiments and tested additional goalie-history encodings. For each historical fold, model settings were chosen on the preceding season, then scored on the next season. The five targets 2022–2026 contain 6,993 games; the specific 2025–26 test season contains 1,394 games (1,312 regular-season, 82 playoff).

All goalie performance, workload, and rest features use goalie games dated strictly before the target game date. The identity of the target game's starter is taken from the same-game postgame goalie box. That identity test is therefore a retrospective upper bound for a future workflow that records the announced starter before puck drop. It is included because the project can refresh a probability when the starter changes; it is not a timestamped pregame-starter backtest.

These historical seasons have been inspected during prior research. Results below are development diagnostics, not an untouched final test. The frozen 2026–27 shadow predictions were not changed or included.

## 2025–26 results

The current B2B reference selected the winner in **768 of 1,394 games (55.09%)**. Always picking the home team got 724 right (51.94%); B2B was 44 picks better. Its Brier score was 0.245150 and log loss 0.683085. Average home probability was 54.13%, while home teams won 51.94% of these games.

| Model / feature set | Correct | Accuracy | Brier | Log loss | Result |
|---|---:|---:|---:|---:|---|
| Always pick home; p=0.50 | 724 | 51.94% | 0.250000 | 0.693147 | Simple winner baseline |
| Elo only | 734 | 52.65% | 0.248843 | 0.690902 | Better probabilities than coinflip, below team model |
| Lagged event form only | 754 | 54.09% | 0.245630 | 0.684099 | Useful signal, but below B2B reference |
| Team Strength 004 | 756 | 54.23% | 0.245779 | 0.684447 | Reference without B2B |
| Team Strength 004 + B2B | **768** | **55.09%** | 0.245150 | 0.683085 | Best tested winner accuracy among the core no-goalie candidates |
| + tied-state 5v5 xG | 763 | 54.73% | 0.245022 | 0.682798 | Better probability loss, fewer correct picks; one season only |
| + tied and close-state xG | 765 | 54.88% | **0.245004** | **0.682762** | Best Brier/log loss in this single season, but weaker cross-season than tied-only |
| + shrunk prior goalie save percentage | 767 | 55.02% | 0.245125 | 0.683054 | One fewer correct pick; negligible score change |
| + prior goalie GSAx per 30 shots, 20-start window | 770 | 55.24% | 0.245166 | 0.683114 | Highest raw accuracy in the goalie screen, only 2 picks above reference; Brier/log loss worse |
| + PP xG quality | 758 | 54.38% | 0.245285 | 0.683372 | Worse than B2B reference |
| + PP xG, exposure, and finishing | 756 | 54.23% | 0.245523 | 0.683881 | Worse than B2B reference |

The full table, including all goalie encodings and team-strength variants, is in `2026_model_battery.csv`.

## Results across five chronological seasons

| Model | Games | Accuracy | Brier | Log loss | Interpretation |
|---|---:|---:|---:|---:|---|
| Team Strength 004 | 6,993 | 59.47% | 0.235382 | 0.662976 | No-goalie foundation |
| Team Strength 004 + B2B | 6,993 | 59.92% | 0.234726 | 0.661540 | Best stable candidate; Brier improved in 5/5 folds |
| + tied-state xG | 6,993 | 59.96% | 0.234630 | 0.661329 | Small probability gain; Brier improved in 4/5 folds, paired 95% interval crosses zero |
| + career-shrunk save percentage | 6,993 | 59.90% | 0.234579 | 0.661230 | Small pooled gain; Brier improved in 4/5 folds, paired 95% interval crosses zero |

For tied-state xG, the paired game-level 95% interval for Brier change was **[-0.000223, +0.000029]**. For career-shrunk save percentage it was **[-0.000300, +0.000010]**. Neither clears a reliable improvement threshold. The intervals are exploratory and not clustered by team or season.

The B2B result is the most repeatable feature gain so far. Tied-state xG remains worth a narrow follow-up. Broad goalie quality/workload bundles, rolling GSAx variants, and special-teams proxies did not beat B2B consistently.

## Where 2025–26 errors clustered

These partitions are exploratory and cannot justify a same-season adjustment.

- **Middle favorites were overconfident:** among 397 games where the favorite was projected at 55–60%, the average favorite probability was 57.3%, but favorites won 49.6%. In the 60–65% band (250 games), the average was 62.3% and favorites won 55.6%. The stronger 65–75% bands were much closer to forecast, but had smaller samples.
- **Team residuals:** Montreal's average team win probability was 47.9% against a 56.4% win rate (101 games); Minnesota was 48.7% against 54.8% (93 games). Florida was forecast at 55.6% against a 48.8% win rate (82 games). These are one-season residuals, not evidence for team-specific adjustments.
- **Back-to-back context:** away-only B2B games (197) scored 59.4% accuracy; both teams on B2B (79) scored 51.9%; home-only B2B (89) scored 53.9%. This supports a predeclared home/away B2B interaction test, with uncertainty due to subgroup size.
- **Postgame-only diagnostic partitions:** the model scored worse in the highest actual-game-xG quartile (Brier 0.2503, n=349) than the lowest (0.2434, n=349). On games that ended in a shootout, accuracy was 39.5% (n=119). These are not pregame features; they suggest testing a pregame scoring-environment/uncertainty signal and separating overtime/shootout probabilities later.

## Goalie decision

The battery included current actual-starter identity with lagged save percentage, GSAx, 5v5 GSAx, recent-versus-established form, and workload/rest. The career-shrunk save-percentage feature is the only goalie encoding with a modest multi-season Brier/log-loss improvement, but its uncertainty interval crosses zero and its 2025–26 lift is effectively nil. The GSAx/20-start version gained only two winner picks in 2025–26 while worsening both Brier and log loss; it improved Brier in only 1/5 historical folds.

**Decision: keep goalie versions in RESEARCH; promote none from this battery.** The user's authorization allows retrospective starter identity in research. A future live version can accept a confirmed pregame starter and recalculate when the starter changes. Its forward test must record the goalie and timestamp before puck drop so we can measure that actual workflow.

## Next experiments

1. Test a small, predeclared home/away B2B interaction set against the same B2B reference.
2. Test tied-state xG at 10/20/40 prior-game windows with one fixed shrinkage choice, selected only inside each fold's validation season.
3. Test whether goalie ability adds value when the starter identity is captured pregame and frozen, including probability updates when the starter changes.
4. Diagnose the 55–65% favorite overstatement and high-scoring/shootout residuals with model inputs and uncertainty analysis; do not apply a post-hoc probability correction.

No public JSON, website output, or promoted model was changed by this experiment.
