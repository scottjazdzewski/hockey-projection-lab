"""Chronological bridge test from exact-starter goal forecasts to game wins.

This is a research-only comparison. The exact-starter goal variant uses the
same-game goalie identity from the postgame goalie box, so its results are an
upper-bound diagnostic until a timestamped pregame starter feed is integrated.
The script compares that score-derived probability with the frozen HWIN OOF
probability and fits one convex blend weight on source season 2025 before the
untouched source-season 2026 evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_ID = "EXP-0034"
MODEL_ID = "HOCKEY-LAB-EXACT-STARTER-GOAL-WIN-BRIDGE-020-RESEARCH"
GOALS_ID = "HOCKEY-LAB-GOALS-PREDICTED-GOALIE-018"
HWIN_ID = "HOCKEY-LAB-GAME-1.0.0"
VALIDATION_SEASON = 2025
TEST_SEASON = 2026
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261103


def normalize_game_id(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"\.0$", "", regex=True)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    if len(y) == 0:
        raise ValueError("Cannot score an empty fold")
    logloss = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    brier = (p - y) ** 2
    return {
        "games": int(len(y)),
        "log_loss": float(logloss.mean()),
        "brier": float(brier.mean()),
        "accuracy": float(((p >= 0.5) == y.astype(bool)).mean()),
        "mean_home_probability": float(p.mean()),
        "observed_home_win_rate": float(y.mean()),
        "mean_absolute_probability": float(np.abs(p - 0.5).mean()),
    }


def fit_calibration(y: np.ndarray, p: np.ndarray) -> dict:
    """Fit logistic calibration intercept and slope on the supplied fold."""
    y = np.asarray(y, dtype=float)
    z = np.log(np.clip(p, 1e-10, 1.0 - 1e-10) / np.clip(1.0 - p, 1e-10, 1.0))
    x = np.column_stack([np.ones(len(z)), z])
    beta = np.zeros(2, dtype=float)
    for _ in range(60):
        eta = np.clip(x @ beta, -30.0, 30.0)
        fitted = 1.0 / (1.0 + np.exp(-eta))
        weights = np.maximum(fitted * (1.0 - fitted), 1e-8)
        gradient = x.T @ (fitted - y)
        hessian = (x.T * weights) @ x
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


def paired_block_ci(
    rows: pd.DataFrame,
    candidate_col: str,
    baseline_col: str,
    loss_name: str,
) -> dict:
    """Paired 14-day time-block bootstrap for candidate-minus-baseline loss."""
    y = rows["home_win"].to_numpy(dtype=float)
    p1 = np.clip(rows[candidate_col].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
    p0 = np.clip(rows[baseline_col].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
    if loss_name == "brier":
        delta = (p1 - y) ** 2 - (p0 - y) ** 2
    elif loss_name == "log_loss":
        delta = -(y * np.log(p1) + (1.0 - y) * np.log1p(-p1))
        delta -= -(y * np.log(p0) + (1.0 - y) * np.log1p(-p0))
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    dates = pd.to_datetime(rows["game_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Date-block bootstrap requires a valid game_date for every game")
    block_id = ((dates - dates.min()).dt.days // 14).to_numpy()
    blocks = np.unique(block_id)
    block_indices = [np.flatnonzero(block_id == block) for block in blocks]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.empty(BOOTSTRAP_REPS, dtype=float)
    for rep in range(BOOTSTRAP_REPS):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([block_indices[index] for index in selected])
        boot[rep] = delta[indices].mean()
    return {
        "candidate_minus_baseline_mean_loss": float(delta.mean()),
        "paired_14_day_block_bootstrap_95pct_ci": [
            float(x) for x in np.quantile(boot, [0.025, 0.975])
        ],
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "calendar_blocks": int(len(blocks)),
        "block_days": 14,
    }


def load_inputs(goals_path: str, hwin_path: str, direct_hwin_path: str):
    goal = pd.read_parquet(goals_path)
    goal = goal.loc[
        goal.model_variant.isin(
            ["no_goalie", "retrospective_actual_starter_diagnostic"]
        ),
        [
            "game_id", "source_season", "model_variant", "p_full_home",
            "home_full_win",
        ],
    ].copy()
    goal["game_id"] = normalize_game_id(goal.game_id)

    hwin_all = pd.read_parquet(hwin_path)
    hwin = hwin_all.loc[
        hwin_all.model_variant.eq("team_strength_004_plus_b2b")
        & hwin_all.game_type.eq("R"),
        [
            "game_id", "game_date", "source_season", "home_win", "p_home",
            "game_type",
        ],
    ].copy()
    hwin["game_id"] = normalize_game_id(hwin.game_id)
    hwin = hwin.drop_duplicates("game_id")
    if hwin.game_id.duplicated().any():
        raise ValueError("Frozen HWIN predictions are not unique by game ID")

    common = goal.merge(
        hwin,
        on="game_id",
        how="inner",
        suffixes=("_goals", "_hwin"),
        validate="many_to_one",
    )
    if (common.source_season_goals != common.source_season_hwin).any():
        raise ValueError("Goal and HWIN season labels disagree after game-ID join")
    if (common.home_full_win.astype(int) != common.home_win.astype(int)).any():
        raise ValueError("Goal and HWIN outcome labels disagree")
    common = common.rename(
        columns={"source_season_goals": "source_season", "p_home": "p_hwin"}
    )
    common["home_win"] = common.home_win.astype(int)
    common["p_hwin"] = common.p_hwin.astype(float)
    common["p_full_home"] = common.p_full_home.astype(float)
    return common, hwin_all, direct_hwin_path


def choose_weight(rows: pd.DataFrame, goalie_column: str) -> dict:
    y = rows.home_win.to_numpy(dtype=float)
    p_hwin = rows.p_hwin.to_numpy(dtype=float)
    p_goals = rows[goalie_column].to_numpy(dtype=float)
    grid = np.linspace(0.0, 1.0, 101)
    losses = []
    for weight in grid:
        p = (1.0 - weight) * p_hwin + weight * p_goals
        losses.append(binary_metrics(y, p)["log_loss"])
    best = int(np.argmin(losses))
    return {
        "goal_model_weight": float(grid[best]),
        "objective": "validation-season full-game home-win log loss",
        "validation_log_loss": float(losses[best]),
        "validation_games": int(len(rows)),
        "grid_step": 0.01,
        "boundary_selected": bool(best in (0, len(grid) - 1)),
    }


def main(goals_path: str, hwin_path: str, direct_hwin_path: str, outdir: str) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    joined, hwin_all, _ = load_inputs(goals_path, hwin_path, direct_hwin_path)
    wide = joined.pivot(
        index=["game_id", "source_season", "game_date", "home_win", "p_hwin"],
        columns="model_variant",
        values="p_full_home",
    ).reset_index()
    wide = wide.rename(
        columns={
            "no_goalie": "p_goals_no_goalie",
            "retrospective_actual_starter_diagnostic": "p_goals_exact_starter",
        }
    )
    expected = {"p_goals_no_goalie", "p_goals_exact_starter"}
    if not expected.issubset(wide.columns):
        raise ValueError("Both goal-probability variants are required for every game")
    wide["p_blend_no_goalie"] = np.nan
    wide["p_blend_exact_starter"] = np.nan

    val = wide.loc[wide.source_season.eq(VALIDATION_SEASON)].copy()
    test = wide.loc[wide.source_season.eq(TEST_SEASON)].copy()
    if len(val) != 1312 or len(test) != 1312:
        raise ValueError(
            f"Expected 1,312 regular-season games in each fold; got "
            f"{VALIDATION_SEASON}={len(val)}, {TEST_SEASON}={len(test)}"
        )
    weight_no = choose_weight(val, "p_goals_no_goalie")
    weight_exact = choose_weight(val, "p_goals_exact_starter")
    for weight, source_column, target_column in [
        (weight_no["goal_model_weight"], "p_goals_no_goalie", "p_blend_no_goalie"),
        (weight_exact["goal_model_weight"], "p_goals_exact_starter", "p_blend_exact_starter"),
    ]:
        wide[target_column] = (
            (1.0 - weight) * wide.p_hwin + weight * wide[source_column]
        )

    test = wide.loc[wide.source_season.eq(TEST_SEASON)].copy()
    per_season = {}
    for season, group in wide.groupby("source_season", sort=True):
        y = group.home_win.to_numpy(dtype=float)
        variants = {
            "frozen_hwin_v1": "p_hwin",
            "goals_score_prob_no_goalie": "p_goals_no_goalie",
            "goals_score_prob_exact_starter": "p_goals_exact_starter",
        }
        if group["p_blend_no_goalie"].notna().all():
            variants["hwin_plus_goals_no_goalie_blend"] = "p_blend_no_goalie"
            variants["hwin_plus_goals_exact_starter_blend"] = "p_blend_exact_starter"
        per_season[str(int(season))] = {
            name: {
                **binary_metrics(y, group[column].to_numpy()),
                "calibration": fit_calibration(y, group[column].to_numpy()),
            }
            for name, column in variants.items()
        }

    # Recompute the HWIN exact-starter layer's 2026 regular-season score on the
    # same IDs, excluding playoff games present in the original HWIN battery.
    exact_hwin = pd.read_parquet(direct_hwin_path)
    exact_hwin["game_id"] = normalize_game_id(exact_hwin.game_id)
    regular_ids = set(
        hwin_all.loc[
            hwin_all.model_variant.eq("team_strength_004_plus_b2b")
            & hwin_all.game_type.eq("R"),
            "game_id",
        ].astype("string")
    )
    exact_hwin = exact_hwin.loc[
        exact_hwin.game_id.isin(regular_ids)
        & exact_hwin.source_season.eq(TEST_SEASON)
    ].copy()
    frozen_regular = hwin_all.loc[
        hwin_all.model_variant.eq("team_strength_004_plus_b2b")
        & hwin_all.game_type.eq("R")
        & hwin_all.source_season.eq(TEST_SEASON),
        ["game_id", "home_win", "p_home"],
    ].copy()
    frozen_regular["game_id"] = normalize_game_id(frozen_regular.game_id)
    exact_hwin = exact_hwin.merge(
        frozen_regular,
        on="game_id",
        how="inner",
        suffixes=("_challenger", "_frozen"),
        validate="one_to_one",
    )
    if len(exact_hwin) != 1312:
        raise ValueError(f"Expected 1,312 matched direct-HWIN games; got {len(exact_hwin)}")
    if (exact_hwin.home_win_challenger.astype(int) != exact_hwin.home_win_frozen.astype(int)).any():
        raise ValueError("Direct HWIN challenger labels disagree with frozen HWIN outcomes")
    if not np.allclose(exact_hwin.p_base, exact_hwin.p_home, atol=1e-10):
        raise ValueError("Direct HWIN base probabilities do not match frozen HWIN")
    direct_hwin_test = {}
    for label, column in [
        ("frozen_hwin", "p_base"),
        ("exact_starter_hwin_challenger", "p_exact_starter_candidate"),
    ]:
        direct_hwin_test[label] = binary_metrics(
            exact_hwin.home_win_challenger.to_numpy(dtype=float),
            exact_hwin[column].to_numpy(dtype=float),
        )
    direct_hwin_test["games"] = int(len(exact_hwin))
    direct_hwin_test["selected_family"] = str(
        exact_hwin.selected_family.iloc[0]
    ) if len(exact_hwin) else None

    main_test = test.copy()
    paired = {
        "exact_starter_goals_derived_vs_frozen_hwin": {
            metric: paired_block_ci(main_test, "p_goals_exact_starter", "p_hwin", metric)
            for metric in ("log_loss", "brier")
        },
        "exact_starter_blend_vs_frozen_hwin": {
            metric: paired_block_ci(main_test, "p_blend_exact_starter", "p_hwin", metric)
            for metric in ("log_loss", "brier")
        },
        "exact_starter_goals_derived_vs_no_goalie_goals_derived": {
            metric: paired_block_ci(
                main_test, "p_goals_exact_starter", "p_goals_no_goalie", metric
            )
            for metric in ("log_loss", "brier")
        },
    }
    output_cols = [
        "game_id", "source_season", "game_date", "home_win", "p_hwin",
        "p_goals_no_goalie", "p_goals_exact_starter", "p_blend_no_goalie",
        "p_blend_exact_starter",
    ]
    wide[output_cols].to_parquet(out / "paired_predictions.parquet", index=False)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "status": "RESEARCH; no promotion, production export, or website update",
        "champion": HWIN_ID,
        "goal_model": GOALS_ID,
        "run_date": "2026-11-03",
        "validation_design": (
            "Use source season 2025 only to select a 0.01-grid convex-blend weight by "
            "log loss; evaluate unchanged on 2026 regular season. Score probabilities "
            "against HWIN OOF predictions on identical game IDs."
        ),
        "goalie_identity_caveat": (
            "Exact goalie variant uses same-game actual starter identity from the "
            "postgame goalie box. Goalie quality/workload features are prior-date only, "
            "but the identity itself has no pregame timestamp. Treat all results as an "
            "upper-bound retrospective diagnostic, not live pregame evidence."
        ),
        "feature_selection_caveat": (
            "The underlying goal-model goalie feature family and blend weight both use "
            "the 2025 validation season. The 2026 fold remains untouched, but a larger "
            "rolling-origin sample is required before promotion."
        ),
        "game_counts": {
            str(int(season)): int(len(group))
            for season, group in wide.groupby("source_season", sort=True)
        },
        "weights_selected_on_2025": {
            "no_goalie_score_model": weight_no,
            "exact_starter_score_model": weight_exact,
        },
        "metrics_by_season": per_season,
        "exact_starter_hwin_challenger_2026_regular_season": direct_hwin_test,
        "paired_2026_test_loss_deltas": paired,
        "production_decision": (
            "Do not change Hockey Lab Game v1.0.0. Exact-starter goals improve goal-score "
            "likelihood in the prior EXP-0029 five-season test, but 2026 win-probability "
            "results here are mixed and the paired block intervals include zero. The "
            "separate exact-starter HWIN challenger must also clear further rolling tests."
        ),
        "artifacts": {
            "paired_predictions": "paired_predictions.parquet",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--goals",
        default="research/.tmp_goalie_probability_016/out/goals_018_final/oof_predictions.parquet",
    )
    parser.add_argument(
        "--hwin",
        default="research/exports/HWIN_PREVIOUS_SEASON_BATTERY_011_20261103/results/goalie_battery_011/oof_predictions.parquet",
    )
    parser.add_argument(
        "--direct-hwin",
        default="research/.tmp_goalie_probability_016/out/hwin_exact_starter_019/oof_predictions.parquet",
    )
    parser.add_argument(
        "--out",
        default="research/.tmp_goalie_probability_016/out/exact_starter_goal_win_bridge_020",
    )
    args = parser.parse_args()
    main(args.goals, args.hwin, args.direct_hwin, args.out)
