# Expected-Goals Improvement Implementation Status — 2026-11-06

## What is implemented on this research branch

- A nested rolling-origin runner to compare the fixed-60 xG features against 20/40/60/100-game exponentially weighted xG inputs.
- Per-run SHA-256 recording for the PBP and canonical game-spine files. Optional expected hashes stop execution if a supplied frozen snapshot does not match.
- An append-only pregame goalie snapshot schema and JSONL validator. The contract rejects outcome/market fields, out-of-order captures, post-puck-drop snapshots, and invalid projected/confirmed state transitions.
- CI checks for Python syntax, schema validity, ledger transitions, and the strict-prior-date EWMA feature guard.

## Validation completed

GitHub Actions run [42236854903](https://github.com/scottjazdzewski/hockey-projection-lab/actions/runs/42236854903) passed on this branch. This covers schema and validator behavior plus a synthetic EWMA leakage check. It does **not** run the five-season goal backtest or demonstrate an accuracy gain.

## Current status

- Production predictions, public cards, and model weights are unchanged.
- The EWMA model experiment is ready to run when the research data runtime and canonical PBP/spine artifacts are available.
- Prior xG research used a rebuilt snapshot whose hashes differ from frozen EXP-0027/0028. The new runner records exact hashes and can enforce expected hashes, but the original frozen inputs must be restored to reproduce those results exactly.
- The repository's current public site is static and the files available here contain no durable goalie-snapshot store or historical pregame goalie feed. The contract is ready; live capture cannot be enabled until the real pregame source and persistent storage path are connected.

## Decision gate

Do not change the 60-game benchmark or publish a goalie adjustment based on scaffold CI. First execute the EWMA backtest on a hash-pinned dataset, compare paired per-season goal MAE and team-count log loss, and require stable improvement without material home/away or score-log-loss regression. Separately, collect timestamped expected and confirmed goalie states prospectively, then evaluate only information available before each forecast cutoff. Markets remain benchmark-only.
