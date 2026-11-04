# Free historical odds source audit

Reviewed 2026-11-04 for the closing-market benchmark. This audit records free
candidate data without admitting unauditable prices into the strict benchmark.
No paid source was queried and no odds were added to the Hockey Lab scorer.

## Hockey Lab historical window

The production model-performance export reports 6,993 rolling-origin games,
with source-season codes 2022–2026. These are NHL seasons 2021-22 through
2025-26. The immutable forward ledger's first retained production forecasts
are from November 3, 2026 (source-season 2027). The benchmark's historic odds
scope is therefore 2021-22 through 2025-26, with forward capture beginning in
2026-27; pre-2021-22 odds are out of scope.

A row-level chronological out-of-sample replay was generated from the current
production selection/training policy for those 6,993 games. It carries explicit
training cutoffs and `CHRONOLOGICAL_OOS` provenance, and uses no market data.
It is not the original frozen forecast ledger. Replay accuracy/Brier/log-loss
are 0.599600/0.234708/0.661498; the existing aggregate production export
reports 0.599171/0.234726/0.661540 over the same game count. Preserve this small
difference and keep replay rows labeled separately from actual frozen rows.

## Public one-season consensus dataset

[Min Maxxxer's NHL 2025-26 Closing Odds dataset](https://huggingface.co/datasets/minmaxxxer/nhl-closing-odds-2025-26)
is downloadable without signup and is marked CC BY 4.0. Its CSV contains 1,394
games (1,312 regular season and 82 postseason), one row per game, a precomputed
no-vig home close probability, and a bookmaker count from 16 to 23 (median 20).
The dataset describes removing each bookmaker's margin proportionally and
taking the median across bookmaker brands.

A local matching audit joined **1,394/1,394** rows to the canonical
2025-26 NHL game spine with exact UTC scheduled start and home/away franchise
crosswalks. Forty-seven source date_et labels differ from the canonical
game-date label because one is Eastern time and the other follows the NHL
schedule date; the join used the exact UTC start and teams, not an inferred
date correction.

This is useful as a coverage and market-probability cross-check, but it is not
eligible for the strict benchmark: the published file has no per-book prices
or odds capture timestamps. Its README calls the closing value the “latest”
price “once the game started,” so the required invariant
odds_timestamp < scheduled_start cannot be verified. Do not promote its
probabilities to canonical closing consensus, calculate official model-vs-market
metrics from them, or treat its bookmaker count as independently auditable
book rows. If reused for a separately labelled exploratory analysis, preserve
attribution to Min Maxxxer and the dataset link.

## ESPN close-only archive lead

The public [Momentum NHL project README](https://github.com/renenunezg/momentumnhl#closing-lines)
reports a one-time ESPN odds API cache with one preferred book's two-way
closing moneyline for 5,134 of 5,225 games from 2022-23 onward. Its parser
([backend/odds/espn.py](https://github.com/renenunezg/momentumnhl/blob/main/backend/odds/espn.py))
materializes date, teams, close prices, and source, but not a per-quote capture
timestamp proving the close was observed before scheduled start. It also does
not cover the model's first 2021-22 fold. This is a useful lead for a separately
labeled single-book reference, but it cannot satisfy the required timestamp,
coverage, or multi-book consensus rules. No odds were copied from that project.

## Timestamped line-level archive

[The Odds Gap historical data page](https://theoddsgap.com/data) documents
per-book scan timestamps and each book's last price before game start, with
archive history beginning June 4, 2026. Its free sample is limited to
September 15–21, 2026; broader history requires a paid plan. That sample is
2026-27 preseason, outside the 2021-22 through 2025-26 model window. The public
API policy also limits its endpoints to individual/assistant display rather
than use as a data source for other products and prohibits bulk scraping. It
is not an appropriate source for this repository's canonical data. No paid
subscription or extraction was made.

## PuckAPI free allowance and coverage

PuckAPI offers 500 credits per month after free-key signup, with no card
required. Its current docs price get_game_detail and get_odds at 10 credits per
game returned, so the free allowance covers at most 50 game responses before
other API use. The pricing page lists 13,710 credits for 2025-26 historical
odds; that implies about 1,371 returned games, fewer than the canonical
season's 1,394 completed games. A local dry-run found 6,993 completed canonical
games in the model's 2021-22 through 2025-26 window, or 69,930 credits at the
documented rate. Do not treat completed-game counts as odds coverage. The API
has estimate_only pricing, but no key is present in the research environment,
so no account-only data was requested.

The provider describes historical odds from 2019-20 onward (plus partial 2019
playoffs), with about 16–18 books per game in 2020-21 through 2023-24 and about
eight since; more than 60 books are available for roughly 300 games in
2025-26 when requesting all books. It labels odds coverage Partial and says
availability varies by book and game. These are provider-level coverage
descriptions, not verified pregame line counts for Hockey Lab's canonical
spine.

Its timing guide says the strict rule is to use captured_at before scheduled
start; the source's closing label alone is not enough. The collector therefore
rejects missing or at/after-start timestamps and never substitutes a later
price. Historical requests remain too large for the free monthly allowance.

## Decision

No free public dataset located in this audit supplies the 2021-22 through
2025-26 model sample with both (a) individual sportsbook moneylines and
(b) auditable last-before-start timestamps. The one-season public consensus
file matches the 2025-26 game spine but fails the timestamp and per-book
requirements. The Odds Gap no-key sample is preseason-only, and its public
API policy excludes use as a data source for another product. The ESPN close-
only archive lead lacks observed timestamps and misses the first target
season. PuckAPI remains the best documented source for a strict backfill, but
its 500-credit free allowance covers at most 50 of the required 6,993 game
responses. Until an approved historical data source is available, report
historical market coverage as unavailable rather than substituting weaker
prices.
