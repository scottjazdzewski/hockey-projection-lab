"""Residual-map diagnostics for rolling-origin expected-goal forecasts."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "run_009_new_snapshot/goalie_team_form/oof_predictions.parquet"
OUT = ROOT / "research/exports/GOALS_XG_BATTERY_20261105"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT)
df = df[df.model_variant.eq("full_reference_60_independent_poisson")].copy()
parts = []
for side in ("home", "away"):
    x = df[["game_id", "game_date", "source_season", f"exp_reg_{side}_goals", f"reg_{side}_goals"]].copy()
    x.columns = ["game_id", "game_date", "season", "predicted_goals", "actual_goals"]
    x["side"] = side
    parts.append(x)
teams = pd.concat(parts, ignore_index=True)
teams["residual_actual_minus_predicted"] = teams.actual_goals - teams.predicted_goals

# Equal-count forecast deciles, pooled then separated by home/away.
decile_tables = []
for group_name, group in [("all", teams), ("home", teams[teams.side.eq("home")]), ("away", teams[teams.side.eq("away")])]:
    group = group.copy()
    group["forecast_decile"] = pd.qcut(group.predicted_goals, 10, labels=False, duplicates="drop") + 1
    out = group.groupby("forecast_decile", observed=True).agg(
        n=("actual_goals", "size"),
        mean_predicted=("predicted_goals", "mean"),
        mean_actual=("actual_goals", "mean"),
        mean_residual=("residual_actual_minus_predicted", "mean"),
        mae=("residual_actual_minus_predicted", lambda s: s.abs().mean()),
        rmse=("residual_actual_minus_predicted", lambda s: np.sqrt(np.mean(s*s))),
        min_predicted=("predicted_goals", "min"),
        max_predicted=("predicted_goals", "max"),
    ).reset_index()
    out.insert(0, "group", group_name)
    decile_tables.append(out)
pd.concat(decile_tables, ignore_index=True).to_csv(OUT / "expected_goal_residual_by_forecast_decile.csv", index=False, float_format="%.8f")

# Per-season calibration slope/intercept, both sides pooled and separately.
calibration = []
for season, season_df in teams.groupby("season"):
    for side in ("all", "home", "away"):
        group = season_df if side == "all" else season_df[season_df.side.eq(side)]
        slope, intercept = np.polyfit(group.predicted_goals, group.actual_goals, 1)
        calibration.append({
            "season": int(season), "group": side, "team_games": len(group),
            "mean_predicted": group.predicted_goals.mean(), "mean_actual": group.actual_goals.mean(),
            "mean_residual_actual_minus_predicted": group.residual_actual_minus_predicted.mean(),
            "mae": group.residual_actual_minus_predicted.abs().mean(),
            "calibration_intercept": intercept, "calibration_slope": slope,
        })
pd.DataFrame(calibration).to_csv(OUT / "expected_goal_calibration_by_season.csv", index=False, float_format="%.8f")

# Compact overall/season summaries for the report.
summary = []
for season, group in [("pooled", teams), *[(str(s), g) for s, g in teams.groupby("season")]]:
    slope, intercept = np.polyfit(group.predicted_goals, group.actual_goals, 1)
    summary.append({"season": season, "team_games": len(group), "mean_predicted": group.predicted_goals.mean(),
                    "mean_actual": group.actual_goals.mean(),
                    "mean_residual_actual_minus_predicted": group.residual_actual_minus_predicted.mean(),
                    "mae": group.residual_actual_minus_predicted.abs().mean(),
                    "calibration_intercept": intercept, "calibration_slope": slope})
pd.DataFrame(summary).to_csv(OUT / "expected_goal_calibration_summary.csv", index=False, float_format="%.8f")
print(pd.DataFrame(summary).to_string(index=False))
print("Wrote residual deciles and seasonal calibration tables to", OUT)
