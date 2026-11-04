# Goals Model Research Audit and Reproduction Blocker

**Review date:** 2026-11-04  
**Scope:** Goals / score-distribution research only  
**Production status:** No production or Game Card changes  
**Win-probability status:** No HWIN model changes

## Repository and history review

The default branch does not contain the goals research package. The work is preserved on the unmerged branches `codex/hockey-goals-distribution-001` through `codex/hockey-goals-nested-window-selection-010`. These branches contain the scripts and reports for EXP-0019 through EXP-0028. The 010 branch also has the EXP-0028 manifest. Its manifest reports exact byte parity on a second full run, but the run outputs and input Parquet files are not committed on that branch.

A separate Drive handoff, EXP-0029, was reviewed because it is newer than the 010 branch. It is a retrospective actual-starter upper-bound study. Its same-game starter identity has no authenticated pregame timestamp, so its goalie layer is not eligible as a live input. The HWIN win-probability model and public cards remain outside this work.

## Best recoverable goals benchmark

The current strongest goals-only benchmark in the saved research is EXP-0027's fixed 60-prior-date team-history model. It improves modestly over the 20-date benchmark on both expected-goal error and score/outcome likelihood. EXP-0028's nested window selector chose 60 dates in four folds and 40 in one, but trailed fixed 60 dates on regulation score log loss and three-way Brier; it is not the preferred benchmark.

EXP-0029's retrospective goalie feature improves score likelihood further by a small amount, but cannot be treated as a pregame forecast. For any live-eligible comparison, fixed 60 dates remains the reference until timestamped starter identities can be joined to historical prediction cutoffs.

The reports record this broad progression:

| Experiment | Main result | Current interpretation |
| --- | --- | --- |
| EXP-0019 | Full process plus Dixon–Coles: final-total mean error −0.240 goals/game; predicted OT 17.0% vs 22.2% observed | Original research benchmark; clear total and tie-mass underprediction |
| EXP-0020 | Diagonal-inflated Poisson: mean total error −0.186; predicted OT 21.6%; OT Brier 0.1732 vs about 0.1729 constant-rate | Better distributional fit, but still not calibrated enough |
| EXP-0021 | Selecting by final-score likelihood reduced predicted OT to 13.5% and worsened OT Brier | Reject that selection objective |
| EXP-0022 | OT-constrained selection predicted OT 20.9%; OT Brier 0.17313 | Small changes; still slightly worse than climatology |
| EXP-0023/24 | Prior-season rate offset shrank pooled bias to −0.045; goal MAE worsened slightly and regulation/OT Brier did not improve | Supports a rate-level component, not a blanket correction |
| EXP-0026/27 | Longer team-history windows modestly improved several metrics; EXP-0027 fixed 60 dates was strongest | Candidate research benchmark; not a promotion test |
| EXP-0028 | Nested 20/40/60 selector was slightly worse than fixed 60 dates on key regulation metrics | Retain as a useful selector diagnostic, not champion |
| EXP-0029 | Career-shrunk goalie quality improved score likelihood slightly using retrospective same-game starter identity | Pre-game eligibility unproven; exclude from live conclusions |

These are reported historical results, not a fresh reproduction in this workspace.

## Reproduction status

A new run could not be reproduced from the available workspace. The EXP-0028 manifest points to external inputs:

- `/tmp/hockey_events/data/pbp_feature_output/game_team_event_features.parquet`
- `/tmp/hockey_spine/data/spine_v3/games.parquet`

The manifest records their checksums, but neither file nor the OOF prediction/PMF exports are present in this workspace or on the reviewed 010 repository branch. The manifest also references a source manifest for each dataset. The attached task contained instructions only, not the data. The checked-out workspace has no historical Parquet, CSV, or run files.

Therefore the saved historical numbers can be audited against the reports and manifest, but they cannot be independently recalculated here. No claim of a new run or reproduction is made. Recover the exact, checksum-matching Parquet inputs and prior OOF artifacts before fitting any new candidate; do not substitute reconstructed or differently versioned data silently.

## Bias diagnosis from existing evidence

The total-goal bias has at least two separable components:

1. **Rate-level underprediction.** EXP-0019's reported regulation mean errors were −0.068 home and −0.119 away goals per team-game. EXP-0023 used side-specific prior-season residual offsets. It reduced final-total mean error substantially, but worsened home/away goal MAE slightly and did not improve regulation or OT Brier. This is evidence of pooled level drift, but not evidence that a fixed correction is safe.
2. **Score-law / tie-mass sensitivity.** With the same EXP-0020 rates, the score-selection objective changed predicted OT from 21.6% under regulation-score selection to 13.5% under full-score selection. EXP-0022's constraint brought it to 20.9%, but OT Brier remained slightly worse than a constant-rate baseline. EXP-0020 also tested bivariate Poisson without a useful shared-component gain.

The summaries do **not** isolate season drift, high-score compression, home/away asymmetry in final scoring, special-teams conversion, rest effects, opponent-strength interactions, finishing calibration, or the fraction of OT error explained by rate error versus dependence. Those require game-level OOF rows and the score PMFs. Do not claim the residual cause has been resolved.

## Next diagnostic and candidate protocol

Once the exact input snapshots and prior OOF artifacts are restored:

1. Re-run the EXP-0027 fixed-60 benchmark and verify checksums, target counts, fold boundaries, regulation labels, probability sums, and key pooled metrics before changing the model.
2. Freeze its per-game OOF forecasts and report residual mean and MAE separately by target season, home/away, predicted-goal decile, predicted-total decile, strength-difference band, rest/B2B, and lagged power-play feature bands. Include uncertainty from paired, season-clustered resampling. Use these to distinguish broad level drift from rate compression or subgroup-specific misses.
3. For total-rate calibration, compare a no-correction baseline to validation-fitted shared intercept, side-specific intercepts, shared slope, and side-specific slopes on the log-rate scale. Fit each only on the preceding validation season, refit through that season, and score the next season. Include the EXP-0023 offset as a comparator. Do not add a fixed +0.24 or choose terms from target-season residuals.
4. Diagnose OT separately by predicted regulation-tie probability band, exact tied-score diagonal (0–0, 1–1, 2–2, etc.), predicted total, and expected-goal differential. On each preceding validation season, compare the existing diagonal-inflated law with independent Poisson and dependence alternatives already in the research code. Keep a constant-rate OT forecast as a benchmark. Score each frozen choice on the next season for OT Brier, calibration, regulation score likelihood, total-goal likelihood, and goal MAE.
5. Keep all development folds labeled as previously inspected. Require a new prospective shadow period for promotion. Do not add goalie identity unless each historical starter observation has an authenticated pregame timestamp.

## Decision

The model remains **RESEARCH**. The fixed 60-date goals-only model is the best recoverable pregame-eligible reference, not a promoted model. Existing evidence points to both rate-level bias and score-law sensitivity, while the detailed cause of the remaining total-goal and OT errors is unresolved. The next valid step is to restore the exact research data and OOF package, reproduce EXP-0027, then run the predeclared decompositions above. No production, win-probability, or Game Card files were changed.

## Data recovery route found

The repository already contains two read-only manual GitHub Actions workflows that rebuild the versioned sources from SportsDataverse:

- `.github/workflows/hockey-pbp-features.yml` builds `game_team_event_features.parquet`, its source manifest, and summary, then uploads `hockey-pbp-features-2010-2027-20261102.zip`.
- `.github/workflows/hockey-game-spine-v3.yml` builds `games.parquet`, its source manifest, and summary, then uploads `hockey-canonical-spine-v3-20261102.zip`.

Both are configured with `workflow_dispatch`, have read-only repository permissions, and retain artifacts for 14 days. Their pinned source mapping matches the research data authority contract, making these the right recovery route to test against the EXP-0028 checksums before retraining.

I could not start the Actions runs through the available GitHub connector: it can read and edit repository files but does not expose workflow dispatch or run-listing. The browser route requires GitHub sign-in. The selected account reported that password sign-in is unsupported; the secure Google account selection did not complete, and no account was chosen or authenticated. The browser is left at the GitHub sign-in handoff. No Actions workflow was dispatched and no data was modified.

After sign-in, run both workflows, download their artifacts, and compare the resulting Parquet checksums with the EXP-0028 manifest. Only an exact match closes this reproduction blocker. If source releases have changed, preserve the new source manifests and treat the rebuilt set as a new data snapshot rather than an exact reproduction.
