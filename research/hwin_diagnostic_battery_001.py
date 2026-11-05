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
    print("\nLARGEST SLICES (n>=100)")
    print(focus.head(20).to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.input, args.out)
