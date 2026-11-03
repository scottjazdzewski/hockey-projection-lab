# Free historical odds source audit

Reviewed 2026-11-03 for the closing-market benchmark. This audit records free
candidate data without admitting unauditable prices into the strict benchmark.
No paid source was queried and no odds were added to the Hockey Lab scorer.

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

## Timestamped line-level archive

[The Odds Gap NHL archive page](https://theoddsgap.com/data/nhl) reports
per-book lines and a close defined as each book's last price before the game
starts, with coverage from June 4, 2026. It reports 239 completed games across
20 books through November 2, 2026. Its no-key free data is only the sample
week September 15–21, 2026; the wider history is subscription-only. That
sample overlaps the 2026-27 preseason (which began September 19) and precedes
the regular-season opener on September 29, so it cannot provide a regular-
season historical backfill. No subscription was started.

## PuckAPI free allowance

PuckAPI publishes a 500-credit monthly free allowance and charges 10 credits
per get_game_detail game, or at most 50 game details per month before any
other use. The broad historical pull planned in the main source audit is
approximately 93,920 credits. No API key is present in the research
environment, so no account-only data was requested. The existing collector
remains dry-run by default and strict about timestamp validation.

## Decision

No free public dataset located in this audit supplies a broad historical NHL
sample with both (a) individual sportsbook moneylines and (b) auditable
last-before-start timestamps. The one-season public consensus file matches the
game spine perfectly but fails the timestamp and per-book requirements. Keep
it excluded from canonical artifacts. The Odds Gap free sample is too short
and preseason-only for the requested historical regular-season comparison.
PuckAPI remains the best documented source for a strict backfill, but its free
quota is not a practical way to obtain the full history. Until a free key,
free bulk archive, or approved paid source is available, report historical
coverage as unavailable rather than substituting these weaker data.
