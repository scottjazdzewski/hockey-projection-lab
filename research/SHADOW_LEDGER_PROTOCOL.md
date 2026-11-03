# Pregame Shadow Ledger Protocol

## Purpose

Record genuinely pregame probabilities for a forward NHL test. The ledger is research-only until the model clears promotion review. It is not a website export.

## Prediction snapshot

For each slate, `research/hwin_shadow_score_010.py`:

1. Reads the canonical game spine and the versioned historical PBP feature bundle.
2. Selects the same-day games whose canonical start time is still in the future at the prediction timestamp.
3. Fits the Team Strength 004 + B2B research candidate: parameter selection uses completed 2025–26 games; the final fit uses completed seasons through 2025–26.
4. Uses only prior PBP games, current schedule context, and pregame data. No bookmaker odds or current-game outcomes enter.
5. Writes a new per-slate Parquet prediction file and summary. It refuses to overwrite a nonempty output directory.

Every row stores game ID, teams, start time, prediction timestamp, input cutoff, latest PBP game date, model/feature/source versions, home/away probabilities, goalie/lineup state, and data-quality flags. Probabilities must be finite, within [0,1], sum to 1, and be saved before puck drop.

If an expected starter or lineup snapshot later changes, save a new timestamped pregame snapshot as a separate immutable file. Never replace an earlier prediction file. The primary evaluation snapshot is the last valid model snapshot before the scheduled start.

## Outcomes and scoring

After a game is final, write its verified winner and final result to a separate outcome artifact with the outcome source and result-available timestamp. Join outcomes to frozen predictions by game ID; do not edit prediction Parquet files or their manifests.

Score the forward test by Brier score, log loss, accuracy, calibration slope/intercept, and reliability bins. Report results by season, probability band, home/away, B2B/rest, goalie state, and data freshness when sample size permits. Do not fit a calibration adjustment to the same forward games used to claim improvement.

## Source freshness and withholding

Record both source artifact timestamps and the most recent game date represented in PBP. If a source is stale, preserve that fact in the row flags and manifest. If an input cutoff is after prediction time, or prediction time is at/after puck drop, refuse to score the game. Do not fill missing games with guessed stats, market prices, or postgame fields.

The first shadow batch is dated 2026-11-03. It uses canonical PBP through 2026-11-01, so each prediction carries a two-day PBP lag flag. The site’s `today-games.json` was dated 2026-11-02 and was not used for the model slate. No public export was created.

## First shadow batch

- Model ID: `HWIN-SHADOW-010-SCHEDULE007-B2B-RESEARCH`
- Run ID: `HWIN_SHADOW_2026-11-03`
- Slate: 11 canonical regular-season games scheduled for 2026-11-03 ET
- Status: frozen research-only pregame predictions; outcomes pending
- Data cutoff: 2026-11-02T22:29:26.434Z
- Latest PBP game date: 2026-11-01
- Data sources and hashes: `HWIN_SHADOW_2026-11-03_MANIFEST.json` in Drive `04_VALIDATION`

This batch is a forward test, not a production promotion. The historical folds used to choose the candidate were previously inspected; only later games provide prospective evaluation.
