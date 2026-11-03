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

The production workflow refreshes around:

- 5:15 AM Eastern (after the 5:00 AM slate build)
- 10:30 AM Eastern
- 2:30 PM Eastern
- 5:30 PM Eastern

The UTC cron is duplicated around DST boundaries and the job gates on America/New_York local hour.

## Freeze policy

The live-context updater may update:

- `goalie_state`
- `lineup_state`
- context source flags

It must **not** change:

- win probabilities
- fair prices
- model ID/version
- model generation timestamp
- model data cutoff

If future research promotes a goalie or lineup feature, the daily scorer may generate a new pregame model snapshot before puck drop under a new/final snapshot policy. Historical published snapshots must remain archived.

## Licensed-provider upgrade path

SportsDataIO currently documents:

- Starting Goaltenders by Date, recommended call interval 3 minutes
- Line Combinations by Season, recommended call interval 15 minutes
- injury/player feeds
- projected goaltenders the preceding night and confirmations after official announcements

A SportsDataIO adapter can replace the current provider by writing the same normalized `live-context.json` contract. No frontend or model-interface rewrite should be necessary.

Do not commit API keys. Any licensed provider credential must be stored as a GitHub/Cloudflare secret.


## Refresh cadence

Production live context uses a quarter-hour scheduler with a gate. Full provider pulls occur at the established daily checkpoints (5:15 AM, 10:30 AM, 2:30 PM, and 5:30 PM America/New_York) and once more for each game when it enters the approximately 15-minute pregame window. The pregame gate reads the actual `start_time_et` values from `public/data/today-games.json`, so staggered start times receive their own final pregame check. The scheduler does not perform a provider pull on quarter-hour ticks that match neither a fixed checkpoint nor a pregame window.
