# Hockey Lab engine performance and production audit — 2026-11-16

**Audit timestamp:** November 16, 2026, approximately 07:00 Eastern.  
**Repository:** `scottjazdzewski/hockey-projection-lab` (`main`).  
**Authority:** committed production JSON, individually frozen forward-game records, GitHub Actions histories/logs, promoted model artifact, and saved market snapshots.  
**Scope:** HWIN game winner probabilities, goalie layer, forward forecast capture, market benchmark, xG output completeness, and publishing reliability. **No model parameters or historical forecasts were changed in this audit.**

## Executive findings

1. GitHub is receiving writes: Nov 16 at 03:39 ET forward ledger was updated by `hockey-lab-bot`. Market and live-context bots also pushed on Nov 15. The Nov 16 slate recovery run was queued at audit time.
2. Historical **rolling-origin** evaluation: 6,993 games, accuracy **60.0458%**, Brier **0.234651**, log loss **0.661414**, calibration summary **0.037643** for promoted goalie model `prospective-0.3.0`. These metrics are read from the fixed promoted artifact, **not recomputed from current-season outcomes on each daily publishing run**.
3. Verified **prospective frozen** evaluation: **18 games / 8 wins correctly picked = 44.44%**, Brier **0.257360**, log loss **0.714190**. All 18 frozen records individually verified to have forecast, source context, and freeze timestamps before scheduled start. This is a very small, nonrepresentative sample, not a reliable population estimate.
4. Within forward records, **10 team-only** forecasts: 4/10 correctly picked, Brier 0.254175; **8 goalie-adjusted**: 4/8 correctly picked, Brier 0.261341. These groups are different games and must not be read as an A/B experiment.
5. **Paired goalie layer on the same eight games:** mean goalie-adjusted minus team-only Brier **+0.005089** (worse), log-loss **+0.011837** (worse). Too few games to decide promotion or rejection.
6. **Expected goals:** no forward xG observations: `xg_games_scored=0`, and every frozen game has `expected_goals: null`. No honest forward xG accuracy claim is currently possible.
7. **Forward coverage is incomplete:** the 18 records cover only Nov 3, 7, 8, 13, 15, rather than a continuous daily freeze. This is the immediate measurement blocker.
8. **Data freshness:** Nov 15 published win projections carry a `2026-11-02T23:59:59-05:00` source-data cutoff, indicating at least thirteen calendar days of historical-feature lag.
9. **Scoring reliability defect:** among the 100 most recent workflow runs inspected, `Score Hockey Lab Game V1` had 8 successes and 4 failures. Latest failed run 43787447143 rejected games 2026020326–2026020328 as 'generated after puck drop'. **The validation guard is correct and must not be removed.**
10. **Goalie recalculation discrepancy:** on Nov 15 games 2026020329 and 2026020330, the saved final pregame context records both goalie starters as CONFIRMED, but the applied goalie-adjustment evidence records both as PROJECTED, and projection `generated_at` predates confirmed context by hours. Verify actual goalie identities and rescore-on-confirmation behavior; status mismatches alone do not prove different predicted starters.

## Win probability evaluation

| Dataset | n | Winning picks | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| Historical rolling-origin promoted artifact | 6,993 | not included as raw per-game rows | 60.05% | 0.234651 | 0.661414 |
| All immutable forward forecasts | 18 | 8 | 44.44% | 0.257360 | 0.714190 |
| Forward team-only model 1.0.0 | 10 | 4 | 40.00% | 0.254175 | 0.705264 |
| Forward goalie model 0.3.0 | 8 | 4 | 50.00% | 0.261341 | 0.725348 |

Trivial 50/50 prediction benchmark on the forward sample: Brier **0.25**, log loss **0.693147**. Hockey Lab's first 18 forward forecasts score slightly worse than this benchmark, with substantial sampling noise.

The historical goalie layer's **incremental gain** is very small: rolling-origin Brier improves from **0.234708452** to **0.234651026**, change **−0.000057426**; log loss improves **−0.000084024**. It improved only **3 of 5** rolling-origin seasons, and the latest fold worsened. See `artifacts/models/goalie-adjustment-model.json`.

### Forward pick-probability calibration (descriptive only)

| Pick probability interval | Games | Picks correct | Mean pick probability |
|---|---:|---:|---:|
| 50–55% | 8 | 1 | 52.0% |
| 55–60% | 3 | 3 | 58.1% |
| 60–65% | 4 | 3 | 64.2% |
| 65–70% | 1 | 1 | 65.3% |
| 70–75% | 1 | 0 | 72.2% |
| 75–80% | 1 | 0 | 77.0% |

**These bins are not actionable calibration evidence at n=18**. No reliable one-percentage-point bucket analysis (e.g., 54%, 55%, 56%) can be inferred from the forward ledger yet. Preserve bins and accumulate hundreds/thousands of prospective observations. For historical bucket calibration, recompute from per-game rolling-origin predictions instead of fabricating bin counts from aggregate model metrics.

### Market-only benchmark (strictly separated)

Eight Nov 7/8/13/15 forward games have matching nine-book consensus no-vig probabilities saved **before their starts**. On that exact matched subset:

| Model | n | Pick accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| Hockey Lab `prospective-0.3.0` | 8 | 50% | 0.261341 | 0.725348 |
| Saved market no-vig snapshot | 8 | 50% | 0.269884 | 0.743934 |

**Important:** saved pregame market snapshots are not all verified final closing lines. This extremely small comparison cannot establish sustainable outperformance. Market prices remain benchmark-only, never training/model inputs.

## Reliability audit

100 most recent GitHub Actions runs sampled:
- `Update Forward Prediction Ledger`: 22/22 successful.
- `Refresh Independent NHL Market Benchmark`: 27/27 successful.
- `Update Live NHL Context`: 20/20 successful.
- `Update Today's NHL Games`: 12 successes, 1 queued.
- `Score Hockey Lab Game V1`: 8 successes, **4 failures**.

One scheduled job can be reported successful without creating a missing pregame freeze: end-to-end coverage requires **one immutable valid forecast per every eligible NHL game**, not merely successful workflow status.

The latest scorer failure is GitHub Actions run [43787447143](https://github.com/scottjazdzewski/hockey-projection-lab/actions/runs/43787447143), at the `Validate production output` step. The current publisher gives a *single* `game-projections.json.generated_at` for the entire day's slate; rescore attempts after earlier games start overwrite that shared timestamp, then validator correctly rejects the published rows for those started games. Need safe per-game provenance/freeze isolation, not a relaxed timestamp check.

## Highest-priority next work (ordered)

1. **Production integrity / per-game snapshot boundaries.** Design and test a rescore/publish scheme that allows pre-puck goalie changes to modify only not-yet-started games, while preserving the original pre-puck evidence and timestamps of started games. Update scorer, validator, export schema, and ledger together. Retain all pre-puck and no-leakage guards. Test mixed early/late puck times, delays, replay/retry, race conditions, and an interrupted workflow. Acceptance: zero after-start projection publication, zero false validation failures, and no historical forecast rewrites.
2. **Forward coverage recovery.** Audit every eligible date/game against schedule; alert on missing freeze 20 minutes before puck and unresolved at puck. Preserve source snapshots, outcome settlement, model/version, and a transparent 'unscored' classification for missed games. **Never backfill missed predictions with hindsight.** Acceptance: near-100% future eligible-game capture with explicit exceptions.
3. **Source-feature freshness.** Daily current-season results and pregame-only team feature states must update from authoritative feeds instead of remaining on the Nov 2 artifact. Add dashboard/manifest freshness gate and date-stamped source cutoff; prohibit future leakage. Acceptance: postgame results appear in the prior-state feature sequence for the *next* game.
4. **Goalie conditional integrity.** Compare projected-to-confirmed goalie identities and per-game probability version/timestamps; re-score before start when starter changes, and verify each side's adjustment status matches the forecast actually used. Match against frozen team-only baseline on paired games.
5. **True granular calibration tests.** Export per-game rolling-origin evaluation data (model p, winner, season, start time, rest/B2B, goalie state, home/away, model version). Calculate per-percentage-point reliability with n and uncertainty intervals; compare to logistic/isotonic adjustments out of sample. No post hoc manual bucket corrections.
6. **xG model publication / audit.** Supply both teams' predicted goals from a promoted pregame model before freezing. Then compute home/away/total-goals MAE, bias, distributional scoring, and improvement by pace, goalie, rest state. Do not conflate HWIN accuracy with xG accuracy.
7. **Market comparator.** Save final available strictly pre-puck multi-book snapshots and score against identical frozen game outcomes only after each game's result; never feed market odds to the prediction model.

## Source locations

- `public/data/model-performance.json`
- `artifacts/models/goalie-adjustment-model.json`
- `public/data/forward-ledger-summary.json`
- `artifacts/forward/games/*.json` (all 18 checked)
- `public/data/game-projections.json`
- `research/market_benchmarks/live/2026-11-{07,08,13,15}.json`
- `.github/workflows/score-hockey-lab-game-v1.yml`
- `src/production/hockey_lab_game_v1.py`
- `src/production/validate-public-data.mjs`
- `docs/FORWARD_PREDICTION_LEDGER.md`
- `research/HWIN_RESEARCH_LEDGER.md`

This document is an evidence snapshot, not a claim that any of the listed production defects has already been repaired.
