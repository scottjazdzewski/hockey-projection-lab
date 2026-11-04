# HWIN Research Ledger

Last updated: 2026-11-04

## Status

**Current production:** the exact team-only baseline `team-strength-004+b2b-007` plus goalie model `prospective-0.3.0`, promoted for prospective evaluation. The scorer keeps the team-only probabilities and falls back to them when a goalie identity or rating is unavailable.

Current research backbone:
- dynamic Elo-style team strength
- rolling 20-game 5v5 shots-on-goal share
- rolling 20-game 5v5 expected-goal share
- rolling recent non-empty-net scoring
- rest-days difference

This backbone is the benchmark that new layers must beat incrementally.

## Data authority

See `research/DATA_AUTHORITY.md`.

Winner target:
- goalie box-score `W` decision
- 22,139 verified regular-season/playoff results in canonical spine V3
- zero winner-team identity mismatches

Canonical normalized spine:
- 24,014 scheduled games
- 22,139 verified results
- 20,611 regular season
- 1,528 playoffs

Canonical full-PBP HWIN event population:
- 20,819 games
- 41,638 team-game rows
- exactly two sides per game
- 0 duplicate game-side keys
- 0 unresolved winner labels
- 2014-15 full PBP unavailable in audited archive

## Signal Audit 002

Development validation: source season 2025.
Development test: source season 2026.
Current source season 2027 remains prospective shadow.

Best simple event carry-forward signal in 2026 development test:
- rolling 20-game 5v5 SOG share difference
- Brier 0.245176
- log loss 0.683296
- accuracy 55.02%

Event multivariate benchmark:
- 2025 Brier 0.241244
- 2026 Brier 0.245639

## Elo Baseline 001

Best initial training-grid parameters:
- K = 10
- home advantage = 20 Elo points
- offseason retention = 0.70

2025:
- Brier 0.238835

2026:
- Brier 0.247660

Interpretation: Elo and event form carry different information.

## Fusion 001

Elo + broader event form.

2025:
- Brier 0.237479

2026:
- Brier 0.246175

Promising, not promoted.

## Rolling Backtest 001

Nested rolling-origin evaluation for target seasons 2022-2026.
Each target season tunes only on the preceding season and trains on earlier history.

Total games across folds: 6,993.

Weighted results:

| Model | Brier | Log loss | Accuracy |
|---|---:|---:|---:|
| Constant home | 0.248723 | 0.690592 | 53.65% |
| Elo | 0.237896 | 0.668352 | 58.49% |
| Event backbone | 0.239300 | 0.671283 | 58.26% |
| Fusion backbone | **0.236064** | **0.664431** | **59.32%** |

Fusion Brier by target season:
- 2022: 0.228961
- 2023: 0.232873
- 2024: 0.234206
- 2025: 0.237392
- 2026: 0.246943

Event backbone beat fusion in 2026 Brier, but fusion beat event backbone in 2022-2025 and is best overall across all five folds.

Conclusion:
- dynamic team strength is durable
- 5v5 territorial form adds independent value
- fusion is the current research benchmark
- no single 2026 result should determine feature promotion

## Goalie Layer

Goalie feature ETL 001:
- 20,820 games
- 83,283 goalie-game rows
- 41,640 starter rows
- exactly two starters per game
- starter PBP shot coverage: 99.983%
- zero duplicate goalie-game keys
- cross-era comparable xG metric: SOG-based GSAx
- richer unblocked-shot attribution available in modern seasons

Confirmed-starter HWIN Layer 001:
- broad goalie bundle did **not** improve the 2026 development test
- backbone Brier 0.246230
- backbone + broad goalie bundle Brier 0.246556
- broad goalie bundle is not promoted

Goalie Signal Audit 002:
- best single incremental signal so far: 20-game starter save percentage
  - 2026 Brier 0.245949
  - delta vs same backbone: -0.000281
- second: 20-game SOG-based GSAx
  - 2026 Brier 0.246009
  - delta vs backbone: -0.000221
- effects are small and require rolling-origin confirmation before inclusion
- short/medium/long goalie features were highly correlated and unstable when bundled

Rolling-origin and prospective-ledger review are required before promoting a new goalie feature or version. Version `prospective-0.3.0` is already promoted for prospective evaluation.

## Promotion rules

A candidate layer must:
1. be pregame-only and leakage-safe
2. improve Brier/log loss, not only accuracy
3. survive multiple rolling-origin seasons
4. add value after existing backbone features
5. have a live-data path that can reproduce the same feature before puck drop
6. be versioned and archived before promotion

Market odds remain comparison data, not an input to the independent HWIN baseline.


## Goalie Rolling Backtest 003 (historical)

Five nested rolling-origin target seasons (2022-2026), 6,993 games.

Backbone:
- Brier 0.236064
- log loss 0.664431
- accuracy 59.32%

Best goalie candidate: **shrunk prior starter save percentage**
- Brier 0.235900
- log loss 0.664074
- accuracy 59.57%
- delta Brier vs backbone: -0.000165
- improved 4 of 5 target seasons; 2023 was very slightly worse

20-game raw starter save percentage:
- Brier 0.236004
- delta -0.000060

20-game SOG-based GSAx:
- Brier 0.236018
- delta -0.000047

Interpretation:
- broad goalie bundles overfit and are rejected
- goalie skill adds only a **small** incremental signal after team strength/territorial play
- heavily shrunk long-run starter quality is more stable than stacking several recent goalie metrics
- at that stage, shrunk prior save percentage remained a research candidate; it was later recalibrated against the active production baseline and promoted as `prospective-0.3.0` for monitored evaluation
- current 2026-27 shadow results are not used for feature selection


## Team Strength 004 — Recency / Opponent Adjustment

Nested rolling-origin targets 2022-2026, 6,993 games.

Flat-20 backbone:
- Brier 0.236064
- log loss 0.664431
- accuracy 59.32%

Exponentially weighted recent form:
- Brier 0.235665
- log loss 0.663555
- accuracy 59.62%
- delta Brier vs flat-20: -0.000399

Pure opponent-adjusted SOG/xG ratings:
- Brier 0.235509
- log loss 0.663242
- accuracy 59.57%
- delta Brier: -0.000556

**Flat-20 + opponent-adjusted SOG/xG hybrid:**
- Brier **0.235382**
- log loss **0.662976**
- accuracy 59.47%
- delta Brier: **-0.000683**
- Brier improved vs flat-20 in 2023, 2024, 2025 and 2026; 2022 was slightly worse

Opponent-adjusted offense/defense components separately:
- Brier 0.235411
- log loss 0.663031
- accuracy 59.53%
- slightly worse Brier than the simpler aggregate opponent-adjusted hybrid

Interpretation:
- schedule/opponent quality materially improves the team-strength representation
- recency decay also helps, but opponent adjustment adds more
- the simpler aggregate opponent-adjusted SOG/xG signal is preferred over four separate offense/defense component terms
- **new research backbone candidate = Elo + flat-20 5v5 form + opponent-adjusted 5v5 SOG/xG + recent scoring + rest**
- still research-only; not production HWIN


## Goalie Metric Audit 005 — 2026-11-04

Compared prior-only, age-adjusted goalie metrics against the active `team-strength-004+b2b-007` backbone on five rolling-origin target seasons (2022–2026), 6,993 games. The current promoted exposure calibration was held fixed at 400 shots, 15 starts, and 1,800 minutes; candidate weights were fit on each validation season. Market odds were not used.

| Candidate | Delta Brier | Delta log loss | Folds improved (Brier / log loss) |
|---|---:|---:|---:|
| Save percentage | -0.000141 | -0.000280 | 3/5, 3/5 |
| Age-adjusted GAA alone | -0.000037 | -0.000060 | 3/5, 2/5 |
| SOG-based GSAx alone | +0.000075 | +0.000137 | 2/5, 2/5 |
| Save percentage + GAA | +0.000129 | +0.000320 | 3/5, 3/5 |
| Save percentage + SOG-GSAx | +0.000152 | +0.000322 | 2/5, 2/5 |

Interpretation:
- GAA by itself has only a tiny, uneven lift and is not additive to the current SV% signal in this comparison.
- The cross-era SOG-based GSAx rate did not improve the current goalie adjustment. Keep it in research until its shot-quality calibration or metric construction improves.
- The active model remains based on age-adjusted, exposure-shrunk SV%. Do not add GAA or GSAx to production from this audit.
- The scorer's historical identity join was also corrected to resolve unique first-initial/surname aliases from goalie history; ambiguous names still fall back to the team-only forecast.
- Repeat the comparison with `research/hwin_goalie_metric_audit_005.py` when the method or source data changes. The CI workflow `.github/workflows/hwin-goalie-metric-audit-005.yml` packages the fold-level results.