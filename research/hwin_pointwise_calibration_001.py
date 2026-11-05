"""Point-by-point calibration audit for NHL team win probabilities."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_team_forecasts(predictions):
    home = predictions[
        ["game_id", "source_season", "game_date", "home_team_abbr", "away_team_abbr", "p_home", "home_win"]
    ].copy()
    home = home.rename(columns={"home_team_abbr": "team", "away_team_abbr": "opponent"})
    home["p_win"] = home.p_home
    home["won"] = home.home_win.astype(int)

    away = predictions[
        ["game_id", "source_season", "game_date", "home_team_abbr", "away_team_abbr", "p_home", "home_win"]
    ].copy()
    away = away.rename(columns={"away_team_abbr": "team", "home_team_abbr": "opponent"})
    away["p_win"] = 1.0 - away.p_home
    away["won"] = 1 - away.home_win.astype(int)

    forecasts = pd.concat([home, away], ignore_index=True)
    forecasts["probability_pct"] = np.floor(forecasts.p_win * 100 + 0.5).astype(int).clip(0, 100)
    return forecasts


def multiway_bootstrap(forecasts, predictions, reps=1000, seed=20261105):
    """Season-stratified date and shared team-cluster bootstrap for calibration gaps."""
    d = predictions.copy()
    d["_date_block"] = d.source_season.astype(str) + ":" + d.game_date.astype(str)
    date_frame = d[["_date_block", "source_season"]].drop_duplicates().reset_index(drop=True)
    date_frame["_date_code"] = np.arange(len(date_frame))
    date_lookup = dict(zip(date_frame._date_block, date_frame._date_code))
    date_codes = d["_date_block"].map(date_lookup).to_numpy(dtype=int)
    date_codes_by_season = [
        group._date_code.to_numpy(dtype=int)
        for _, group in date_frame.groupby("source_season", sort=False)
    ]

    teams = pd.unique(pd.concat([
        d.home_team_abbr.astype(str), d.away_team_abbr.astype(str)
    ], ignore_index=True))
    team_lookup = {team: i for i, team in enumerate(teams)}
    home_codes = d.home_team_abbr.astype(str).map(team_lookup).to_numpy(dtype=int)
    away_codes = d.away_team_abbr.astype(str).map(team_lookup).to_numpy(dtype=int)

    home_p = d.p_home.to_numpy(dtype=float)
    away_p = 1.0 - home_p
    home_y = d.home_win.to_numpy(dtype=float)
    away_y = 1.0 - home_y
    bin_codes = np.column_stack([
        np.floor(home_p * 100 + 0.5).astype(int).clip(0, 100),
        np.floor(away_p * 100 + 0.5).astype(int).clip(0, 100),
    ]).reshape(-1)
    row_p = np.column_stack([home_p, away_p]).reshape(-1)
    row_y = np.column_stack([home_y, away_y]).reshape(-1)

    rng = np.random.default_rng(seed)
    gap_draws = np.full((reps, 101), np.nan, dtype=float)
    for rep in range(reps):
        date_counts = np.zeros(len(date_frame), dtype=float)
        for season_dates in date_codes_by_season:
            sampled = rng.choice(season_dates, size=len(season_dates), replace=True)
            date_counts += np.bincount(sampled, minlength=len(date_frame))
        sampled_teams = rng.choice(teams, size=len(teams), replace=True)
        team_counts = np.bincount(
            np.fromiter((team_lookup[t] for t in sampled_teams), dtype=int),
            minlength=len(teams),
        ).astype(float)
        game_weights = (
            date_counts[date_codes]
            * team_counts[home_codes]
            * team_counts[away_codes]
        )
        row_weights = np.repeat(game_weights, 2)
        n = np.bincount(bin_codes, weights=row_weights, minlength=101)
        pred_sum = np.bincount(bin_codes, weights=row_weights * row_p, minlength=101)
        win_sum = np.bincount(bin_codes, weights=row_weights * row_y, minlength=101)
        valid = n > 0
        gap_draws[rep, valid] = win_sum[valid] / n[valid] - pred_sum[valid] / n[valid]

    return gap_draws


def main(input_path, output_path, reps=1000):
    all_predictions = pd.read_parquet(input_path)
    d = all_predictions[all_predictions.model.eq("baseline_b2b")].copy()
    if d.empty:
        raise RuntimeError("No baseline_b2b predictions found.")
    if d.game_id.duplicated().any():
        raise RuntimeError("Expected one baseline prediction per game.")
    required = {
        "game_id", "source_season", "game_date", "home_team_abbr",
        "away_team_abbr", "p_home", "home_win",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    d["p_home"] = pd.to_numeric(d.p_home, errors="raise").clip(1e-6, 1 - 1e-6)
    d["home_win"] = pd.to_numeric(d.home_win, errors="raise").astype(int)
    forecasts = build_team_forecasts(d)

    raw = forecasts.groupby("probability_pct").agg(
        forecasts=("game_id", "size"),
        games=("game_id", "nunique"),
        mean_predicted_pct=("p_win", "mean"),
        observed_win_pct=("won", "mean"),
    )
    raw = raw.reindex(range(101))
    raw.index.name = "probability_pct"
    raw["calibration_gap_pp"] = raw.observed_win_pct * 100 - raw.mean_predicted_pct * 100
    gap_draws = multiway_bootstrap(forecasts, d, reps=reps)
    low = np.full(101, np.nan)
    high = np.full(101, np.nan)
    reps_used = np.zeros(101, dtype=int)
    for pct in range(101):
        draws = gap_draws[:, pct]
        draws = draws[np.isfinite(draws)]
        reps_used[pct] = len(draws)
        # A rounded 50% bin contains both sides of the same near-pick'em
        # games; its 50% observed rate is mechanically guaranteed.
        if pct != 50 and len(draws) >= reps * 0.8:
            low[pct], high[pct] = np.quantile(draws, [0.025, 0.975]) * 100
    raw["ci_2_5_pp"] = low
    raw["ci_97_5_pp"] = high
    raw["bootstrap_reps_used"] = reps_used
    raw["low_sample_lt_100"] = raw.games.fillna(0).lt(100)
    raw["structural_complementary_bin"] = raw.index == 50
    raw["interval_excludes_zero"] = (
        (raw.ci_2_5_pp > 0) | (raw.ci_97_5_pp < 0)
    )
    raw = raw.reset_index()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out, index=False)

    print(f"games={len(d)} team_forecasts={len(forecasts)} bins=101 reps={reps}")
    print(raw.loc[raw.probability_pct.between(45, 75)].to_string(index=False))
    print("all_bins_with_interval_excluding_zero:")
    print(raw[raw.interval_excludes_zero.fillna(False)].to_string(index=False))
    print(f"saved={out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reps", type=int, default=1000)
    args = parser.parse_args()
    main(args.input, args.output, args.reps)
