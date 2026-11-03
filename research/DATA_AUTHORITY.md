# Hockey Projection Lab — Data Authority Contract

Last updated: 2026-11-03

This file defines which source fields are allowed to become canonical inputs/targets. It exists to prevent silent leakage, historical key errors, and accidental reuse of provider fields already proven unreliable.

## Game identity and schedule context

**Authority:** SportsDataverse individual season schedule files `nhl_schedule_YYYY.parquet`.

Use for:
- game_id
- game_date
- home/away team identity
- game type (regular season / playoffs / preseason)
- future schedule

Do **not** use `nhl_schedule_master.parquet` as the canonical key. Audit found duplicate game IDs and it did not contain the active 2026-27 schedule.

Do **not** use historical schedule `home_score` / `away_score` as HWIN outcomes. Multiple season files contain corrupted/constant score fields.

## HWIN winner target

**Authority:** goalie box-score decision.

For every completed regular-season/playoff game in the normalized goalie population:
- exactly one goalie has decision `W`
- there are zero games with multiple winning goalies
- winner team identity matches scheduled home/away identity in the audited population

Audited result population: 22,139 games.
- regular season: 20,611
- playoffs: 1,528
- regular-season home-win rate: 54.2526%
- playoff home-win rate: 53.0105%

The HWIN label must come from this decision field, not schedule scores.

## Goal-production / margin targets

**Primary event source:** full PBP scoreboard and goal events.

Schedule score fields are excluded.

Known current issue:
- PBP score can remain tied for shootout games, which is expected because shootout scoring is not ordinary goal production.
- A small anomaly bucket exists where PBP score is tied but shootout flag is absent. Those games may remain in HWIN winner research because winner truth comes from goalie decision, but they must be excluded/investigated before goal-margin model promotion.

## Full PBP historical mapping

SportsDataverse release naming has a historical discontinuity.

Canonical mapping:
- schedule 2010 <- PBP 2010
- 2011 <- 2011
- 2012 <- 2012
- 2013 <- 2013
- 2014 <- 2014
- schedule 2015 (2014-15): full PBP not present in the audited archive
- schedule 2016 <- PBP 2015
- schedule 2017 <- PBP 2016
- schedule 2018 <- PBP 2017
- schedule 2019 <- PBP 2018
- schedule 2020 onward <- same-numbered PBP release

PBP 2018 and PBP 2019 both contain the 2018-19 season. PBP 2018 is more complete by one valid game (2018020266), so PBP 2018 is authoritative for that season.

Never infer the mapping from filename alone.

## PBP feature ledger integrity gate

Canonical feature artifact: `CANONICAL_HWIN_EVENT_FEATURES_GOALIE_WIN_TARGET_2010_2027_20261102.zip` (20,819 games; 41,638 team-game rows). It passes the two-side and duplicate-key checks. Use this artifact for model research.

A superseded Drive copy named `SUPERSEDED_UNCANONICAL_HWIN_GAME_TEAM_EVENT_FEATURES_2010_2027_20261102.zip` has 44,352 rows, 2,714 duplicate game-side keys, and lacks `game_type`; it is archived and must not be used.

Canonical event features must satisfy:
- one home row and one away row per game
- zero duplicate (game_id, home_away) keys
- exactly one HWIN winner per game from goalie decision
- no unresolved winner labels

Same-game event statistics are descriptive observations only. Any HWIN feature derived from them must be lagged so only information known before puck drop is used.

## Team box-score caveat

Many historical team box-score releases do not include `game_id`.

Do not attach these rows to games by row order or guessed sequence.

Keyed modern team-box data can be used where game_id exists. Historical team-strength work should rely on keyed PBP/player/goalie/roster sources instead.

## Player / goalie / roster authority

Normalized keyed coverage is available across the historical result population:
- goalie box scores
- game rosters
- scratches
- skater/player box scores

Player information may feed HWIN for lineup strength, role, availability and goalie state. Player-prop modeling is a later project stage.

## Shift data

Season shift files contain keyed game IDs plus player on/off changes and can support workload, deployment, line/pair continuity and lineup-state research. Shift-derived features are not yet promoted.

## Validation contract

- No future leakage.
- Market odds are not model inputs for the independent HWIN baseline.
- Same-game results are never fed backward into that game's prediction.
- Train/validation/test splits are chronological.
- 2024-25 (source_season 2025) is the primary development-validation season.
- 2025-26 (source_season 2026) was inspected during early signal/fusion research and is therefore a **development test**, not an untouched final holdout.
- Current 2026-27 (source_season 2027) remains prospective shadow/live and is not training data.
- Promotion research must also use rolling-origin season backtests so no single season determines a feature's value.
- Evaluate Brier score, log loss and calibration in addition to winner accuracy.
- No candidate is promoted from a single favorable test.
