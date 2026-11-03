# Goalie Starter and Relief Audit 014

**Run date:** 2026-11-03  
**Status:** RESEARCH diagnostic  
**Scope:** Verify the goalie identity used by the goals-model goalie candidate and quantify how often a relief goalie entered. No production model, public export, or site output was changed.

## Data and method

The audit joined the 6,560 regular-season out-of-fold games from the 2022–2026 goals-model evaluation to the canonical goalie-game feature table and canonical spine goalie box. The box table has an explicit starter field, goalie identity, team side, time on ice, and goalie decision. For each team side, a relief appearance was recorded when a second goalie had positive time on ice.

The goalie candidate's feature builder selects rows marked `starter_bool=True` to identify the target game's starter. It does not infer the starter from the `decision` (W/L/O) column. Goalie ability features are computed using only earlier game dates. Same-game starter identity is still retrospective because the warehouse does not preserve an announcement timestamp.

## Coverage and relief appearances

- The 6,560 games contain 13,120 team-sides.
- Exactly one starter row was marked on every team-side: 13,120/13,120.
- A second goalie recorded positive time on ice on 811 team-sides (6.18%).
- At least one side used a relief goalie in 788 games (12.01%).
- Both sides used a relief goalie in 22 games.

A relief appearance does not prove that the starter was pulled for poor play; injury and other reasons are not distinguished by this flag.

## Sample game records

Times below come from the goalie box feature table. `starter_bool` identifies the starter even when another goalie later received the win decision.

| Game ID | Date | Team | Starter (starter flag, TOI) | Relief goalie (TOI) | Starter shots / goals against |
|---:|---|---|---|---|---|
| 2021020008 | 2021-10-14 | TOR | P. Mrazek (true, 40:00) | J. Campbell (19:27) | 26 / 3 |
| 2021020011 | 2021-10-14 | DAL | B. Holtby (true, 48:07) | A. Khudobin (13:31; received W) | 27 / 2 |
| 2021020016 | 2021-10-14 | VGK | R. Lehner (true, 40:00) | L. Brossoit (20:00) | 31 / 4 |
| 2021020026 | 2021-10-16 | CHI | M. Fleury (true, 11:25) | K. Lankinen (48:22) | 10 / 4 |
| 2021020036 | 2021-10-18 | SEA | P. Grubauer (true, 28:49) | C. Driedger (31:11) | 20 / 5 |

The listed relief time is the recorded box time. The starter flag remains on the opening goalie; a win decision may instead be credited to a relief goalie. This confirms the candidate's starter lookup is not accidentally treating the credited winner as the starter.

## Pull-status diagnostic

A postgame-only subgroup check compared the no-goalie reference with the career-shrunk starter-quality candidate:

| Games | Count | Reference full-game home-win Brier | Goalie candidate | Difference |
|---|---:|---:|---:|---:|
| No relief goalie recorded | 5,772 | 0.236454 | 0.236073 | -0.000381 |
| At least one relief goalie recorded | 788 | 0.227325 | 0.226926 | -0.000400 |

This descriptive split does not show a larger goalie-feature gain in games with a relief appearance. It cannot establish that relief status is unimportant: the partition is defined after the game, may include injuries, and is not available as a pregame feature.

## Interpretation and next step

The historical starter identities are present and consistently marked, including when a starter was replaced. The current goalie model uses the starting goalie's lagged skill; the game's realized goals include both starter and relief-goalie performance. Relief-goalie identity and realized pull status are not predictors. This can add irreducible noise to the starter-only estimate, but this audit does not explain away the small pooled goalie lift.

For a pregame-valid expansion, capture expected/confirmed starter identity and timestamp, plus the available backup's pregame quality/workload. Test the starter and backup features separately and preserve each frozen pregame prediction. Do not use the target game's pull flag or relief-goalie performance as a pregame input.
