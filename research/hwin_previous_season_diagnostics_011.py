"""Context diagnostics for the frozen 2025-26 (source_season 2026) OOF ledger."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hwin_schedule_007 as sched


def normalize_game_id(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric.astype("Int64").astype("string")
    return values.astype("string")


def metrics(g: pd.DataFrame) -> dict:
    if not len(g):
        return {"games": 0}
    y = g.home_win.to_numpy(dtype=int)
    p = np.clip(g.p_home.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    result = {
        "games": int(len(g)),
        "mean_home_probability": float(p.mean()),
        "home_win_rate": float(y.mean()),
        "brier": float(np.mean((y - p) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(np.mean((p >= 0.5) == y)),
    }
    if "favorite_probability" in g:
        result["mean_favorite_probability"] = float(g.favorite_probability.mean())
        result["favorite_win_rate"] = float(g.favorite_won.mean())
    return result


def grouped_metrics(df: pd.DataFrame, group_type: str, group_col: str) -> list[dict]:
    out = []
    for value, g in df.groupby(group_col, dropna=False, sort=True):
        out.append({"group_type": group_type, "group_value": str(value), **metrics(g)})
    return out


def main(prediction_path: str, pbp_path: str, spine_path: str, outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(prediction_path)
    pred = pred.loc[
        pred.target_season.eq(2026) & pred.model_variant.eq("plus_b2b")
    ].copy()
    pred["game_id"] = normalize_game_id(pred.game_id)
    if pred.game_id.duplicated().any():
        raise ValueError("Baseline OOF ledger has duplicate target-season games")

    spine = pd.read_parquet(spine_path)
    spine = spine.loc[
        spine.result_available.fillna(False)
        & spine.game_type.astype("string").isin(["R", "P"])
    ].copy()
    spine["game_id"] = normalize_game_id(spine.game_id)
    spine["game_date"] = pd.to_datetime(spine.game_date, errors="coerce").dt.normalize()
    schedule = sched.build_schedule_features(spine)
    schedule["game_id"] = normalize_game_id(schedule.game_id)
    frame = pred.merge(schedule, on="game_id", how="left", validate="one_to_one")
    frame["month"] = pd.to_datetime(frame.game_date).dt.month
    frame["favorite_probability"] = np.maximum(frame.p_home, frame.p_away)
    frame["favorite_won"] = np.where(
        frame.p_home.ge(0.5), frame.home_win.eq(1), frame.home_win.eq(0)
    )
    frame["favorite_band"] = pd.cut(
        frame.favorite_probability,
        bins=[0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 1.001],
        labels=["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"],
        right=False,
        include_lowest=True,
    )
    frame["b2b_state"] = np.select(
        [
            frame.h_sched_b2b.eq(1) & frame.a_sched_b2b.eq(0),
            frame.h_sched_b2b.eq(0) & frame.a_sched_b2b.eq(1),
            frame.h_sched_b2b.eq(1) & frame.a_sched_b2b.eq(1),
        ],
        ["home_only", "away_only", "both"],
        default="neither_or_unknown",
    )
    frame["rest_difference"] = frame.h_sched_rest_days - frame.a_sched_rest_days
    rest = frame.rest_difference
    frame["rest_state"] = np.select(
        [rest.ge(2), rest.ge(1), rest.gt(-1), rest.le(-2), rest.le(-1)],
        ["home_2plus_days", "home_1_day", "equal_or_within_half_day", "away_2plus_days", "away_1_day"],
        default="unknown",
    )

    pbp = pd.read_parquet(pbp_path)
    pbp["game_id"] = normalize_game_id(pbp.game_id)
    pbp2026 = pbp.loc[pbp.source_season.eq(2026)].copy()
    event_diag = pbp2026.groupby("game_id", as_index=False).agg(
        actual_game_xg=("xg", "sum"),
        went_ot=("went_ot", "max"),
        went_shootout=("went_shootout", "max"),
    )
    frame = frame.merge(event_diag, on="game_id", how="left", validate="one_to_one")
    frame["actual_xg_environment"] = pd.qcut(
        frame.actual_game_xg, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop"
    ).astype("string").fillna("unknown")
    frame["overtime_state"] = np.select(
        [frame.went_shootout.fillna(False).astype(bool), frame.went_ot.fillna(False).astype(bool)],
        ["shootout", "overtime_no_shootout"], default="regulation",
    )

    groups = []
    # Probability bands show pick accuracy and probability reliability together.
    for band, g in frame.groupby("favorite_band", observed=False):
        groups.append({
            "group_type": "favorite_probability_band", "group_value": str(band),
            **metrics(g), "favorite_win_rate": float(g.favorite_won.mean()) if len(g) else None,
        })
    for group_type, col in [
        ("game_type", "game_type"), ("month", "month"), ("b2b_state", "b2b_state"),
        ("rest_state", "rest_state"), ("actual_xg_environment_diagnostic_only", "actual_xg_environment"),
        ("overtime_state_diagnostic_only", "overtime_state"),
    ]:
        groups.extend(grouped_metrics(frame, group_type, col))
    frame["strength_confidence_band"] = pd.qcut(
        frame.favorite_probability, q=5, labels=["Q1_closest", "Q2", "Q3", "Q4", "Q5_strongest"], duplicates="drop"
    ).astype("string")
    groups.extend(grouped_metrics(frame, "favorite_strength_quantile", "strength_confidence_band"))

    # Per-team records pool home and away appearances; score the model from
    # that team's perspective and retain support counts.
    home_col = "home_team_abbr" if "home_team_abbr" in frame else "home_team"
    away_col = "away_team_abbr" if "away_team_abbr" in frame else "away_team"
    team_games = pd.concat([
        pd.DataFrame({
            "team": frame[home_col], "team_win": frame.home_win.astype(int),
            "team_probability": frame.p_home, "game_id": frame.game_id,
        }),
        pd.DataFrame({
            "team": frame[away_col], "team_win": 1 - frame.home_win.astype(int),
            "team_probability": frame.p_away, "game_id": frame.game_id,
        }),
    ], ignore_index=True)
    team_rows = []
    for team, g in team_games.groupby("team", dropna=False):
        p = np.clip(g.team_probability.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        y = g.team_win.to_numpy(dtype=int)
        team_rows.append({
            "team": str(team), "games": int(len(g)),
            "team_probability_mean": float(p.mean()), "team_win_rate": float(y.mean()),
            "brier": float(np.mean((y - p) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "accuracy": float(np.mean((p >= 0.5) == y)),
            "model_favored_team_rate": float(np.mean(p >= 0.5)),
            "team_win_rate_when_favored": float(y[p >= 0.5].mean()) if np.any(p >= 0.5) else None,
        })

    pd.DataFrame(groups).to_parquet(out / "subgroup_metrics.parquet", index=False)
    pd.DataFrame(team_rows).to_parquet(out / "team_metrics.parquet", index=False)
    frame.to_parquet(out / "2026_baseline_with_diagnostics.parquet", index=False)
    summary = {
        "season_label": "2025-26 (source_season 2026)",
        "model_variant": "Team Strength 004 + B2B",
        "games": int(len(frame)),
        "overall": metrics(frame),
        "diagnostic_only_groups": ["actual_xg_environment_diagnostic_only", "overtime_state_diagnostic_only"],
        "interpretation_note": "Subgroups are exploratory descriptions of a previously inspected test season; they are not causal evidence and must not be used for same-season probability rescue.",
        "outputs": ["subgroup_metrics.parquet", "team_metrics.parquet", "2026_baseline_with_diagnostics.parquet"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--pbp", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.predictions, args.pbp, args.spine, args.out)
