# Closing-market benchmark

This is a research-only lane. Sportsbook lines are external comparison data and
must never be passed into Hockey Lab scoring, training, tuning, or probability
calibration. No market field is added to `public/data/game-projections.json`.

## Source decision

PuckAPI is the preferred historical source. Its NHL odds archive starts in
2019-20, carries NHL game IDs, and `get_game_detail` returns sportsbook odds
snapshots with captured timestamps. This supports a direct game-ID join and
per-book latest-timestamp selection. The collector uses `get_game_detail` to
request all sportsbook snapshots held for each game; it immediately excludes
any row without a valid timestamp or timestamped at/after puck drop. The
benchmark independently repeats the strict timing check.

Published historical coverage, last checked 2026-11-03:

| Season | Completed games with odds | Published coverage | Published book coverage |
| --- | ---: | ---: | --- |
| 2019-20 | 1,212 | 100% | About 6 books/game |
| 2020-21 | 952 | 100% | About 16–18 books/game through 2023-24 |
| 2021-22 | 1,401 | 100% | Same archive period |
| 2022-23 | 1,400 | 100% | Same archive period |
| 2023-24 | 1,400 | 100% | Same archive period; ESPN BET begins |
| 2024-25 | 1,398 | 100% | 8 books/game |
| 2025-26 | 1,394 | 100% | 8 books/game; 60+ books for about 300 games |
| 2026-27 | 235 | 100% of completed games | Current partial season |

The provider says its sportsbook archive includes DraftKings, FanDuel,
BetMGM, Pinnacle, ESPN BET, Bet365, Caesars and additional books, with
availability varying by season and game. These are provider-published counts,
not a successful query or a verified Hockey Lab join. The benchmark report
will replace these planning figures with observed per-game and per-book
coverage after an authenticated backfill.

The Odds API is the fallback/validation source. It documents NHL historical
odds from June 6, 2020, with snapshots every ten minutes through September
2022 and every five minutes afterward. Querying at scheduled start minus one
second returns the latest available snapshot at or before that time. Its
events do not carry NHL game IDs, so matching must use exact date, both teams,
and scheduled start; ambiguous rows remain unmatched.

The published source pages were reviewed on 2026-11-03. No paid endpoint was
queried for this implementation. Exact fetched sportsbook/game coverage and
exact join counts therefore remain unverified until an account key is
available. PuckAPI lists 10 credits per game detail call and a 500-credit
monthly free allowance; its published 2025-26 season example is 13,940
credits. The historical fetcher is dry-run by default and requires a
caller-supplied `--max-credits` before it will make any request. Do not add a
subscription, buy credits, or run a paid backfill without explicit approval.

PuckAPI allows account holders to store data for internal research/backtesting
while the account is open, but prohibits redistributing raw odds as a bulk
dataset/feed/export. Keep these artifacts private and outside `public/data/`.

## Coverage and identity

The canonical research spine uses NHL game IDs. PuckAPI documents the same
league-assigned key, so that source joins by `game_id` without a team-name
crosswalk. This is a structurally exact join, but the actual number of matched
source rows is not known until the archive is queried. The source's documented
2019-20 start provides seven completed season labels through 2025-26, plus the
current 2026-27 season as it accumulates. It cannot benchmark earlier
canonical games.

For providers without NHL IDs, `closing_market_benchmark.py` only accepts an
exact match on normalized game date, UTC scheduled start, home team and away
team. It writes unmatched and ambiguous rows to `unmatched-audit.json`; it
never resolves a tie by choosing the closest game.

The canonical game result must be the verified HWIN winner target (the goalie
box-score decision documented in `research/DATA_AUTHORITY.md`). A home-win
label can be derived from the canonical spine's `home_win` field.

## Timing, line selection, and no-vig math

For each game and sportsbook, source observations must include both sides of
the full-game moneyline and a timestamp. The latest timestamp strictly earlier
than scheduled start is selected. A `closing` label is not accepted as proof
that a line predates puck drop: the source documents that some closing
snapshots are captured at or just after the scheduled start. Rows at/after
scheduled start, with one-sided prices, invalid prices, or conflicting prices
at the same timestamp are rejected and listed in `line-rejections.json`. No
later line is carried backward to fill a missing close.

American odds are converted to raw implied probabilities per side. Negative
price `-x` uses `x/(x+100)`; positive price `+x` uses `100/(x+100)`. Each
book's two raw probabilities are normalized by their sum to remove that
book's vig. The consensus is the median of per-book no-vig home probabilities;
away probability is one minus that median. The output retains the sportsbook
rows, book count, min/max, population standard deviation, and earliest/latest
timestamps represented.

Quality labels are fixed: 3+ books `CONSENSUS`, 2 `LIMITED_CONSENSUS`, 1
`SINGLE_BOOK`, and 0 `UNAVAILABLE`. Primary comparison metrics use only
`CONSENSUS`; limited and single-book rows remain visible as secondary samples.
The primary comparison allow-list is Pinnacle, Circa, DraftKings, FanDuel,
BetMGM, Caesars, ESPN BET and Bet365. Other authenticated sportsbook rows are
retained in the book-line artifact and coverage audit but do not count toward
the consensus threshold. Common display-name aliases are normalized before
deduplication; the source display name remains in `sportsbook_source_name`.

## Forecast eligibility and model isolation

The comparison accepts only:

- a forward record frozen before scheduled start (`forecast_origin` is
  `FROZEN_PREGAME`); or
- a properly chronological out-of-sample row explicitly marked
  `CHRONOLOGICAL_OOS`.

Forecast timestamps at/after puck drop, invalid probabilities, and probability
pairs that do not sum to one are excluded. No forecasts are recreated from a
model trained on future games. The repository currently has a forward-ledger
path but no retained historical pregame forecast file; therefore the historical
market-vs-model evaluation cannot be populated from the checked-in history
alone. Do not label retrospective full-sample scores as pregame forecasts.

The current canonical historic result spine covers 22,139 verified NHL
regular-season/playoff games, while the odds archive begins in 2019-20. Only
actual frozen or chronological out-of-sample forecast rows can contribute to
the three-way comparison. Goalie confirmation and rest/load breakdowns are
reported only when present in the eligible forecast record; postgame goalie
starts are not substituted.

## Build commands

Install `pandas`, `numpy`, and `pyarrow` in the research environment.

Dry-run a historical PuckAPI request estimate (no network call):

```bash
python research/market/fetch_puckapi_closing_odds.py \
  --games /path/to/spine_v3/games.parquet \
  --start-season 2019 --end-season 2026
```

After data-use approval and with `PUCKAPI_API_KEY` set, pass a reviewed credit
cap to fetch. The script stores only timestamped rows strictly before the
canonical scheduled start; the benchmark repeats this guard and writes the
final canonical artifacts.

```bash
python research/market/fetch_puckapi_closing_odds.py \
  --games /path/to/spine_v3/games.parquet \
  --start-season 2019 --end-season 2026 \
  --max-credits <approved-credit-cap>

python research/market/closing_market_benchmark.py \
  --games /path/to/spine_v3/games.parquet \
  --forecasts /path/to/frozen-or-oos-forecasts.parquet \
  --lines artifacts/market/staging/puckapi-closing-lines.parquet \
  --outdir artifacts/market
```

Outputs:

- `closing-book-lines.parquet` — one strictly pre-start row per game/book.
- `closing-consensus.parquet` — one row per canonical game, including
  unavailable and limited-quality states.
- `hockey-lab-vs-market.parquet` — forecast, market, actual result, picks,
  probability differences, and row-level scores.
- `market-benchmark-report.json` — overall, calibration, season, probability,
  disagreement, favorite-strength, version, rest/load and goalie summaries.
- `unmatched-audit.json` and `line-rejections.json` — visible join/timing
  failures; no silent substitutions.

## Forward snapshot

`capture_forward_market.py` is separate from the scorer. It updates only the
reserved `forecast.market_benchmark` block of an already frozen forward-ledger
record; model probabilities, features, and scorer dispatch are untouched. Each
book row must be timestamped no later than the capture request and strictly
before puck drop. It has no network behavior unless called with an explicit
credit cap and a `PUCKAPI_API_KEY`. The scheduled workflow remains disabled for
market capture unless the repository variable
`HOCKEY_LAB_MARKET_CAPTURE_ENABLED` is set to `true` and a per-run credit cap is
configured. This ensures a market update cannot trigger or alter a model
rescore.
