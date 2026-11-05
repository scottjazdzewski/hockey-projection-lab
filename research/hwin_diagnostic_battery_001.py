import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def score(group):
    if len(group) == 0:
        return {"n": 0, "mean_pred": None, "observed": None,
                "calibration_error": None, "brier": None, "log_loss": None,
                "accuracy": None}
    y = group.home_win.to_numpy(dtype=int)
    p = np.clip(group.p_home.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(group)),
        "mean_pred": float(p.mean()),
        "observed": float(y.mean()),
        "calibration_error": float(y.mean() - p.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(np.mean((p >= 0.5) == y)),
    }


def summarize_groups(df, column, label):
    rows = []
    for value, group in df.groupby(column, dropna=False, observed=False):
        rows.append({"slice": label, "group": str(value), **score(group)})
    return rows


def clustered_gap_bootstrap(df, conditions, date_col, away_team_col, reps=1000, seed=20261105):
    """Pigeonhole bootstrap calibration-residual gaps over date and both team roles."""
    if date_col is None or away_team_col is None:
        return pd.DataFrame(), {
            "status": "skipped",
            "reason": "Missing identifiable game-date or away-team column",
            "date_column": date_col,
            "away_team_column": away_team_col,
        }

    work = df.copy()
    work["_residual"] = work.home_win - work.p_home
    work["_date_block"] = work.source_season.astype(str) + ":" + work[date_col].astype(str)
    dates = work._date_block.drop_duplicates().to_numpy()
    teams = pd.unique(pd.concat([
        work.home_team_abbr.astype(str), work[away_team_col].astype(str)
    ], ignore_index=True))
    team_values = pd.concat([
        work.home_team_abbr.astype(str), work[away_team_col].astype(str)
    ], ignore_index=True)
    team_counts = team_values.value_counts().to_dict()
    date_counts = work._date_block.value_counts().to_dict()
    date_index = work._date_block.to_numpy()
    home_index = work.home_team_abbr.astype(str).to_numpy()
    away_index = work[away_team_col].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    rows = []

    for name, flag in conditions.items():
        mask = np.asarray(flag, dtype=bool)
        residual = work._residual.to_numpy(dtype=float)
        point = float(residual[mask].mean() - residual[~mask].mean())
        draws = []
        for _ in range(reps):
            sampled_dates = rng.choice(dates, size=len(dates), replace=True)
            sampled_teams_home = rng.choice(teams, size=len(teams), replace=True)
            sampled_teams_away = rng.choice(teams, size=len(teams), replace=True)
            dcount = pd.Series(sampled_dates).value_counts().to_dict()
            hcount = pd.Series(sampled_teams_home).value_counts().to_dict()
            acount = pd.Series(sampled_teams_away).value_counts().to_dict()
            weights = np.fromiter(
                (dcount.get(d, 0) * hcount.get(h, 0) * acount.get(a, 0)
                 for d, h, a in zip(date_index, home_index, away_index)),
                dtype=float,
                count=len(work),
            )
            w_flag = weights[mask].sum()
            w_other = weights[~mask].sum()
            if w_flag > 0 and w_other > 0:
                gap = (
                    np.dot(weights[mask], residual[mask]) / w_flag
                    - np.dot(weights[~mask], residual[~mask]) / w_other
                )
                draws.append(float(gap))
        if len(draws) >= reps * 0.8:
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append({
                "slice": name,
                "flagged_vs_unflagged_residual_gap": point,
                "ci_2_5": float(low),
                "ci_97_5": float(high),
                "bootstrap_reps_used": int(len(draws)),
                "n_flagged": int(mask.sum()),
                "n_unflagged": int((~mask).sum()),
                "interval_excludes_zero": bool(low > 0 or high < 0),
            })
    return pd.DataFrame(rows), {
        "status": "complete",
        "method": "season-date stratified pigeonhole resampling of date, home-team, away-team clusters",
        "date_column": date_col,
        "away_team_column": away_team_col,
        "requested_reps": reps,
        "seed": seed,
        "caution": "Exploratory multiway cluster approximation; interval is not a causal estimate.",
    }


def main(input_path, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_predictions = pd.read_parquet(input_path)
    d = all_predictions[all_predictions.model.eq("baseline_b2b")].copy()
    if d.empty:
        raise RuntimeError("No baseline_b2b rows found in saved predictions.")
    if d.game_id.duplicated().any():
        raise RuntimeError("Baseline predictions contain duplicate game IDs.")
    d["p_home"] = np.clip(pd.to_numeric(d.p_home, errors="coerce"), 1e-6, 1 - 1e-6)
    d["home_win"] = pd.to_numeric(d.home_win, errors="coerce").astype(int)
    d["home_favorite"] = d.p_home >= 0.5
    d["favorite_probability"] = np.where(d.home_favorite, d.p_home, 1 - d.p_home)
    d["favorite_won"] = np.where(d.home_favorite, d.home_win, 1 - d.home_win)
    edges = [0, .35, .40, .45, .50, .55, .60, .65, .70, 1.000001]
    d["home_probability_band"] = pd.cut(d.p_home, edges, right=False, include_lowest=True)
    d["favorite_confidence_band"] = pd.cut(
        d.favorite_probability, [.5, .55, .60, .65, .70, .75, .80, 1.000001],
        right=False, include_lowest=True,
    )

    rows = []
    rows.append({"slice": "overall", "group": "all", **score(d)})
    rows.extend(summarize_groups(d, "source_season", "target_season"))
    rows.extend(summarize_groups(d, "home_probability_band", "home_probability_band"))
    rows.extend(summarize_groups(d, "home_favorite", "favorite_side_home"))
    for band, group in d.groupby("favorite_confidence_band", observed=False):
        favorite = group.copy()
        favorite["home_win"] = favorite.favorite_won.astype(int)
        favorite["p_home"] = favorite.favorite_probability.astype(float)
        rows.append({
            "slice": "favorite_confidence",
            "group": str(band),
            **score(favorite),
        })

    conditions = {
        "either_team_3in4": d.either_team_3in4,
        "either_team_4in6": d.either_team_4in6,
        "either_team_5in8": d.either_team_5in8,
        "away_road_trip_finale_proxy": d.away_road_trip_finale_proxy,
    }
    for name, flag in conditions.items():
        for value, subset in [("flagged", d[flag.astype(bool)]),
                              ("unflagged", d[~flag.astype(bool)])]:
            rows.append({"slice": name, "group": value, **score(subset)})

    for team, group in d.groupby("home_team_abbr"):
        if len(group) >= 100:
            rows.append({"slice": "home_team", "group": team, **score(group)})
    for (season, team), group in d.groupby(["source_season", "home_team_abbr"]):
        if len(group) >= 50:
            rows.append({
                "slice": "home_team_by_season",
                "group": f"{season}:{team}",
                **score(group),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "diagnostic_slices.csv", index=False)

    date_candidates = ["game_date", "date", "game_date_local", "game_date_utc",
                       "game_datetime", "start_time", "game_start"]
    date_col = next((c for c in date_candidates if c in d.columns), None)
    away_candidates = ["away_team_abbr", "away_team"]
    away_team_col = next((c for c in away_candidates if c in d.columns), None)
    bootstrap, bootstrap_meta = clustered_gap_bootstrap(
        d, conditions, date_col, away_team_col
    )
    bootstrap.to_csv(out / "schedule_cluster_bootstrap.csv", index=False)

    y = d.home_win.to_numpy(dtype=int)
    p = np.clip(d.p_home.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    calibration = LogisticRegression(C=1e6, max_iter=2000).fit(logits, y)
    global_summary = {
        "games": int(len(d)),
        "overall": score(d),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
        "probability_underestimation_positive": True,
        "exploratory_only": True,
        "minimum_team_slice_n": 100,
        "probability_band_edges": edges,
        "available_columns": list(all_predictions.columns),
        "schedule_cluster_bootstrap": bootstrap_meta,
        "conditions": {
            key: {
                "flagged": score(d[flag.astype(bool)]),
                "unflagged": score(d[~flag.astype(bool)]),
            }
            for key, flag in conditions.items()
        },
    }
    with open(out / "summary.json", "w") as f:
        json.dump(global_summary, f, indent=2)

    focus = metrics[
        (metrics.n >= 100)
        & metrics.calibration_error.notna()
        & ~metrics.slice.eq("overall")
    ].copy()
    focus["abs_calibration_error"] = focus.calibration_error.abs()
    focus = focus.sort_values(["abs_calibration_error", "n"], ascending=[False, False])
    focus.head(20).to_csv(out / "largest_stable_calibration_slices.csv", index=False)

    print(json.dumps(global_summary, indent=2))
    print("\nDIAGNOSTIC SLICES")
    print(metrics.to_string(index=False))
    print("\nSCHEDULE CLUSTER BOOTSTRAP")
    print(bootstrap.to_string(index=False))
    print("\nLARGEST SLICES (n>=100)")
    print(focus.head(20).to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.input, args.out)
