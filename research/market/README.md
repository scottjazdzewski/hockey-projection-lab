# Closing-market benchmark

This is a research-only lane. Sportsbook lines are external comparison data and
must never be passed into Hockey Lab scoring, training, tuning, or probability
calibration. No market field is added to `public/data/game-projections.json`.

## Hockey Lab evaluation window

The model-performance export reports 6,993 rolling-origin games for source-season
codes 2022–2026. In the canonical spine these correspond to NHL seasons
2021-22 through 2025-26. The historical benchmark window is therefore those
five seasons; odds from earlier seasons are outside the Hockey Lab evaluation
sample. Source-season code 2027 is 2026-27 and is handled only through the
forward snapshot.

The forward ledger's first retained production forecasts are dated November 3,
2026. Earlier per-game probabilities are not archived, but the model's aggregate
performance export covers those five rolling-origin folds. The new
`export_chronological_oos_forecasts.py` replays the production model-selection
and fitting policy with prior-season training/validation and emits explicitly
labeled `CHRONOLOGICAL_OOS` rows. These are historical research replays, not
forecasts actually frozen before those games. The replay covers 6,993 games;
its accuracy/Brier/log-loss are 0.599600/0.234708/0.661498, compared with the
stored model-performance export's 0.599171/0.234726/0.661540. Keep that small
replay-to-export difference visible in provenance and do not describe the
reconstructed rows as the original frozen forecasts.

## Source decision

PuckAPI is the leading documented candidate for a historical pull,
but its odds coverage and pre-puck timestamps have not been verified against
the canonical spine. Its [NHL historical odds archive](https://puckapi.com/data/historical-odds)
starts in 2019-20, carries NHL game IDs, and [`get_game_detail`](https://puckapi.com/docs/get-game-detail)
returns sportsbook odds snapshots with captured timestamps. The collector
uses those timestamps for a direct game-ID join and keeps only the latest
snapshot strictly before puck drop; it excludes missing timestamps and any
capture at or after scheduled start. A provider's `closing` label alone is
not accepted as proof of a valid pregame quote.

Published odds coverage, checked 2026-11-03:

- The provider describes odds records from the 2019 playoffs and games from
  2019-20 through the current season, with 191,000+ moneyline, puck-line and
  total records. Its coverage status is Partial; availability varies by
  sportsbook, game and market.
- It reports about 16–18 books per game from 2020-21 through 2023-24, about
  eight books per game since, and 60+ books on roughly 300 games in 2025-26
  when requesting all books. These are provider-wide descriptions, not
  verified per-game coverage on the canonical Hockey Lab spine.
- The docs price get_game_detail and get_odds at 10 credits per game returned.
  Free accounts receive 500 credits monthly, enough for at most 50 game
  responses. A local dry-run against the canonical spine finds 6,993 completed
  games in the model's 2021-22 through 2025-26 OOS window: 69,930 credits at
  the documented price. This is an estimate only; no paid request was made.
  The pricing page lists 13,710 credits for 2025-26 historical odds, so use
  estimate_only and verify actual provider coverage before any pull.
- Odds rows carry capturedAt. The provider says its hockey closing snapshot is
  recorded once the game starts, so the closing label alone cannot prove a
  strictly pre-start timestamp. The collector must retain only actual rows
  with capturedAt earlier than the canonical scheduled start. This can leave
  an earlier pregame price or no eligible price; it must not fill from a later
  line.

The main source audit records the free alternatives and why they are not
eligible for canonical benchmark metrics. No paid endpoint has been queried.
Exact source coverage and game-ID joins remain unverified until a free key is
available or a paid backfill is explicitly approved.

The Odds API is the fallback/validation source. Its [historical odds docs](https://the-odds-api.com/historical-odds-data/)
document NHL historical odds from June 6, 2020, with snapshots every ten
minutes through September 2022 and every five minutes afterward. Querying at scheduled start minus one
second returns the latest available snapshot at or before that time. Its
events do not carry NHL game IDs, so matching must use exact date, both teams,
and scheduled start; ambiguous rows remain unmatched.

The published source pages and [timing and rights terms](https://puckapi.com/docs/timing-and-rights)
were reviewed on 2026-11-03. No paid endpoint was
queried for this implementation. Exact fetched sportsbook/game coverage and
exact join counts therefore remain unverified until an account key is
available. PuckAPI lists 10 credits per game detail call and a 500-credit
monthly free allowance; its published 2025-26 season example is 13,940
credits. The historical fetcher is dry-run by default and requires a
caller-supplied `--max-credits` before it will make any request. Do not add a
subscription, buy credits, or run a paid backfill without explicit approval.

PuckAPI's [published terms](https://puckapi.com/docs/timing-and-rights) allow
data pulled under an account to remain in use for internal research/backtesting,
including after cancellation, but prohibit redistributing raw odds as a bulk
dataset/feed/export. Keep these artifacts private and outside `public/data/`.

## Coverage and identity

The canonical research spine uses NHL game IDs. PuckAPI documents the same
league-assigned key, so that source joins by `game_id` without a team-name
crosswalk. This is a structurally exact join, but the actual number of matched
source rows is not known until the archive is queried. The source's 2019-20 archive extends earlier than Hockey Lab's useful
historical window. The fetcher is scoped to source-season codes 2022–2026
(2021-22 through 2025-26), matching the 6,993-game rolling-origin sample.
Do not request earlier seasons just because the provider has them. Source-season
2027 is the current 2026-27 season and belongs only to the forward snapshot.

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
BetMGM, Caesars, ESPN BET, Bet365, Unibet and PointsBet. Other authenticated
rows (including exchanges) are retained in the book-line artifact and coverage
audit but do not count toward the sportsbook consensus threshold. Common
display-name aliases are normalized before deduplication; the source display
name remains in `sportsbook_source_name`.

## Forecast eligibility and model isolation

The comparison accepts only:

- a forward record frozen before scheduled start (`forecast_origin` is
  `FROZEN_PREGAME`); or
- a properly chronological out-of-sample row explicitly marked
  `CHRONOLOGICAL_OOS`.

Forecast timestamps at/after puck drop, invalid probabilities, and probability
pairs that do not sum to one are excluded. No forecasts are recreated from a
model trained on future games. The checked-in performance file has aggregate
metrics only; `export_chronological_oos_forecasts.py` creates row-level
historical replay forecasts with training cutoffs recorded and no market input.
The forward ledger supplies actual frozen production forecasts beginning
November 3, 2026. Keep the replay and frozen-forward rows distinguishable.

The current canonical historic result spine covers 22,139 verified NHL
regular-season/playoff games, while the odds archive begins in 2019-20. Only
actual frozen or chronological out-of-sample forecast rows can contribute to
the three-way comparison. Goalie confirmation and rest/load breakdowns are
reported only when present in the eligible forecast record; postgame goalie
starts are not substituted.

## Build commands

Install `pandas`, `numpy`, and `pyarrow` in the research environment.

First create the historical model-only forecast replay. It uses the production
model code and has no market input:

```bash
python research/market/export_chronological_oos_forecasts.py \
  --pbp /path/to/game_team_event_features.parquet \
  --games /path/to/spine_v3/games.parquet \
  --out artifacts/market/staging/chronological-oos-forecasts.parquet
```

Dry-run a historical PuckAPI request estimate (no network call):

```bash
python research/market/fetch_puckapi_closing_odds.py \
  --games /path/to/spine_v3/games.parquet \
  --start-season 2022 --end-season 2026
```

After data-use approval and with `PUCKAPI_API_KEY` set, pass a reviewed credit
cap to fetch. The script stores only timestamped rows strictly before the
canonical scheduled start; the benchmark repeats this guard and writes the
final canonical artifacts.

```bash
python research/market/fetch_puckapi_closing_odds.py \
  --games /path/to/spine_v3/games.parquet \
  --start-season 2022 --end-season 2026 \
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
