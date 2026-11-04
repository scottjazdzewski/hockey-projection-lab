# Hockey Lab Live Pregame Context Feeds

## Purpose

This layer supplies time-stamped pregame context to Hockey Lab without changing model probabilities unless a promoted model explicitly uses that context.

Current normalized surface:

- `public/data/live-context.json`
- schema: `artifacts/schemas/live-context.schema.json`

## Current provider

**Daily Faceoff public pages**

Used pages:

- `https://www.dailyfaceoff.com/starting-goalies`
- `https://www.dailyfaceoff.com/teams/{slug}/line-combinations`

Access policy:

- public web pages only
- reads the page's embedded `__NEXT_DATA__` JSON
- does **not** call Daily Faceoff `/api/` routes
- provider robots policy currently allows public pages and disallows `/api/`
- requests carry an identifiable Hockey Projection Lab user agent
- lineup requests are limited to teams playing that day and are paced

This is a development/research integration. Before relying on this source for a public commercial product, confirm any licensing/terms requirements with the provider.

## Goalie normalization

Per team:

- `UNKNOWN`: no named goalie
- `PROJECTED`: a goalie is named but Daily Faceoff has not marked the report Confirmed
- `CONFIRMED`: Daily Faceoff's goalie news strength is Confirmed

Confirmed observations retain:

- goalie name/id
- provider confirmation timestamp
- source/reporter name
- source URL
- Hockey Lab observation timestamp

## Lineup normalization

Each playing team's current projected lineup includes:

- `f1` through `f4`: forward lines
- `d1` through `d3`: defense pairs
- `pp1`, `pp2`: power-play units
- `pk1`, `pk2`: penalty-kill units
- `g`: goalie depth group
- injury status
- game-time-decision flag
- provider source
- provider update timestamp

The feed is marked `PROJECTED`, not `CONFIRMED`. NHL line combinations can change up to and during warmups.

## Update cadence

The production workflow refreshes at the established daily checkpoints:

- 5:15 AM Eastern (after the 5:00 AM slate build)
- 10:30 AM Eastern
- 2:30 PM Eastern
- 5:30 PM Eastern

It also performs **two game-specific pregame pulls for every scheduled start time**:

- approximately **30 minutes before puck drop**
- approximately **15 minutes before puck drop**

The scheduler runs on a five-minute heartbeat and gates against each game's actual `start_time_et` in `public/data/today-games.json`. This means staggered starts receive their own 30-minute and 15-minute checks.

## Freeze policy and goalie-score handoff

The live-context updater may update:

- `goalie_state`
- `lineup_state`
- context source flags

It must **not** directly change:

- win probabilities
- fair prices
- model ID/version
- model generation timestamp
- model data cutoff

The workflow compares the selected goalie identity/status with each new live-context snapshot. It requests a scorer rerun only when the active feature version includes the explicit `goalie-adjustment-` marker; the current team-only feature version keeps this trigger dormant. Changes that require a fresh pregame score include:

- a PROJECTED goalie name/ID changing;
- a PROJECTED goalie becoming CONFIRMED, including confirmation of the same goalie; and
- a CONFIRMED starter being replaced by a different confirmed goalie.

The workflow then invokes the production scorer with the newest live-context snapshot. The scorer recalculates both teams' probabilities and fair moneylines from the fixed team baseline plus the selected goalies' calibrated ratings. The browser renders the newest validated export; it never calculates the goalie effect. If a goalie is UNKNOWN or cannot be matched to a validated rating, use the documented team-only fallback and label the goalie adjustment unavailable.

Every re-score must preserve the goalie name/ID, PROJECTED or CONFIRMED status, source timestamp, Hockey Lab observation timestamp, and scorer/model version. Capture a new immutable forward-ledger snapshot before puck drop. Never rewrite a published projection after puck drop.

**Current implementation state:** the live feed and card display goalie status, but the active model feature version is team-only and its projections carry a `no_goalie_layer` flag. A goalie update therefore changes context display only today. Do not describe current percentages as goalie-adjusted. The future layer's engineering contract and calibration gate are in `docs/GOALIE_CONDITIONAL_WIN_PROJECTIONS.md`.

## Licensed-provider upgrade path

SportsDataIO currently documents:

- Starting Goaltenders by Date, recommended call interval 3 minutes
- Line Combinations by Season, recommended call interval 15 minutes
- injury/player feeds
- projected goaltenders the preceding night and confirmations after official announcements

A SportsDataIO adapter can replace the current provider by writing the same normalized `live-context.json` contract. No frontend or model-interface rewrite should be necessary.

Do not commit API keys. Any licensed provider credential must be stored as a GitHub/Cloudflare secret.


## Refresh cadence

Production live context uses a five-minute scheduler with a gate. Full provider pulls occur at the established daily checkpoints (5:15 AM, 10:30 AM, 2:30 PM, and 5:30 PM America/New_York), plus **two pregame checks for every game: about 30 minutes and 15 minutes before its scheduled start**. The gate reads the actual `start_time_et` values from `public/data/today-games.json`, so staggered start times receive independent checks. The scheduler does not perform a provider pull on quarter-hour ticks that match neither a fixed checkpoint nor one of the pregame windows.


### Scheduler reliability guard

The workflow wakes every five minutes, but it does **not** pull on every wake-up. It computes exact target times for each game's T-30 and T-15 checks and compares them with the last successful `live-context.json generated_at`. A target is pulled once, with a short grace window for normal GitHub Actions scheduling delay. This prevents duplicate provider requests while making the pregame checks much less sensitive to a delayed cron delivery.
