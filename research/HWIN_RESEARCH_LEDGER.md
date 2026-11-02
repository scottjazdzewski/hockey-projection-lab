# HWIN Research Ledger

Last updated: 2026-11-02

## Status

**No production HWIN model has been promoted yet.**

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

Confirmed-starter HWIN layer is under evaluation against the fusion backbone.

## Promotion rules

A candidate layer must:
1. be pregame-only and leakage-safe
2. improve Brier/log loss, not only accuracy
3. survive multiple rolling-origin seasons
4. add value after existing backbone features
5. have a live-data path that can reproduce the same feature before puck drop
6. be versioned and archived before promotion

Market odds remain comparison data, not an input to the independent HWIN baseline.
