"""Previous-season research battery for the NHL game model.

Runs the existing Team Strength 004 + B2B reference against a small,
predeclared goalie feature battery. The current-game starter identity comes
from the retrospective goalie box and is therefore an upper-bound diagnostic;
all goalie skill/workload measurements are strictly prior-date features.

Historical scoring is rolling-origin: train through target-2, tune C on
target-1, test on target. This is research and does not create a production
export or alter any frozen live predictions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hwin_schedule_007 as sched
import hwin_score_state_009 as state
import hwin_team_strength_004 as ts


TARGET_SEASONS = [2022, 2023, 2024, 2025, 2026]
MODEL_VERSION = "HWIN-PREVIOUS-SEASON-BATTERY-011-RESEARCH"
FEATURE_VERSION = "team-strength-004-plus-b2b-plus-lagged-goalie-001"

GOALIE_VARIANTS = {
    "goalie_save_pct_r20": ["d_goalie_save_pct_r20"],
    "goalie_save_pct_career_shrunk": ["d_goalie_save_pct_shrunk"],
    "goalie_gsax_per30_r20": ["d_goalie_gsax_per30_r20"],
    "goalie_gsax_per30_r40": ["d_goalie_gsax_per30_r40"],
    "goalie_gsax_per30_career_shrunk": ["d_goalie_gsax_per30_shrunk"],
    "goalie_5v5_gsax_per30_r20": ["d_goalie_fivev5_gsax_per30_r20"],
    "goalie_recent_vs_established": ["d_goalie_gsax_per30_recent10_vs_r40"],
    "goalie_workload_rest": [
        "d_goalie_days_since_start", "d_goalie_starts_7d", "d_goalie_starts_14d",
    ],
    "goalie_quality_r20_plus_workload": [
        "d_goalie_save_pct_r20", "d_goalie_gsax_per30_r20",
        "d_goalie_fivev5_gsax_per30_r20", "d_goalie_days_since_start",
        "d_goalie_starts_7d", "d_goalie_starts_14d",
    ],
    "goalie_quality_r40_plus_workload": [
        "d_goalie_save_pct_r40", "d_goalie_gsax_per30_r40",
        "d_goalie_fivev5_gsax_per30_r40", "d_goalie_days_since_start",
        "d_goalie_starts_7d", "d_goalie_starts_14d",
    ],
}


def normalize_game_id(values: pd.Series) -> pd.Series:
    """Normalize integer-like PBP float IDs and spine integer IDs consistently."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric.astype("Int64").astype("string")
    return values.astype("string")


def build_goalie_pregame_features(goalies: pd.DataFrame, spine: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join actual starters, then attach only goalie history from earlier dates."""
    required = {
        "game_id", "goalie_id_key", "home_away", "starter_bool", "shots_against_pbp",
        "saves_pbp", "gsax", "fivev5_gsax", "fivev5_shots_against", "source_season",
    }
    missing = sorted(required - set(goalies.columns))
    if missing:
        raise ValueError(f"Goalie source missing required columns: {missing}")

    game_meta = spine[["game_id", "game_date", "source_season", "game_type"]].copy()
    game_meta["game_date"] = pd.to_datetime(game_meta.game_date, errors="coerce").dt.normalize()
    if game_meta.game_date.isna().any():
        raise ValueError("Canonical game dates are required for goalie history cutoffs")

    starters = goalies.loc[goalies.starter_bool.fillna(False).astype(bool)].copy()
    starters["game_id"] = normalize_game_id(starters.game_id)
    starters = starters.merge(game_meta, on="game_id", how="inner", validate="many_to_one", suffixes=("", "_spine"))
    if starters.duplicated(["game_id", "home_away"]).any():
        raise ValueError("More than one retrospective starter is present for a game-side")

    for col in ["shots_against_pbp", "saves_pbp", "gsax", "fivev5_gsax", "fivev5_shots_against"]:
        starters[col] = pd.to_numeric(starters[col], errors="coerce")
    starters["shots_against_pbp"] = starters.shots_against_pbp.clip(lower=0)
    starters["saves_pbp"] = starters.saves_pbp.clip(lower=0)
    starters["gsax_per30"] = np.where(
        starters.shots_against_pbp.gt(0),
        30 * starters.gsax / starters.shots_against_pbp,
        np.nan,
    )
    starters["fivev5_gsax_per30"] = np.where(
        starters.fivev5_shots_against.gt(0),
        30 * starters.fivev5_gsax / starters.fivev5_shots_against,
        np.nan,
    )
    starters["save_pct_start"] = np.where(
        starters.shots_against_pbp.gt(0),
        starters.saves_pbp / starters.shots_against_pbp,
        np.nan,
    )

    # Collapse goalie/date before shifting, so every game on a date gets state
    # from dates strictly earlier than that date.
    daily = starters.groupby(["goalie_id_key", "game_date"], as_index=False, sort=True).agg(
        shots=("shots_against_pbp", "sum"),
        saves=("saves_pbp", "sum"),
        gsax=("gsax", "sum"),
        fivev5_shots=("fivev5_shots_against", "sum"),
        fivev5_gsax=("fivev5_gsax", "sum"),
        start_rows=("game_id", "count"),
    )
    daily["goalie_id_key"] = daily.goalie_id_key.astype(str)
    daily["gsax_per30"] = np.where(daily.shots.gt(0), 30 * daily.gsax / daily.shots, np.nan)
    daily["fivev5_gsax_per30"] = np.where(
        daily.fivev5_shots.gt(0), 30 * daily.fivev5_gsax / daily.fivev5_shots, np.nan
    )
    daily["save_pct"] = np.where(daily.shots.gt(0), daily.saves / daily.shots, np.nan)

    history_parts = []
    for goalie_id, g in daily.groupby("goalie_id_key", sort=False):
        g = g.sort_values("game_date").copy()
        dates = g.game_date.to_numpy(dtype="datetime64[ns]")
        ix = np.arange(len(g))
        g["prior_starts"] = ix
        prev = g.game_date.shift(1)
        g["days_since_start"] = (g.game_date - prev).dt.days
        g["starts_7d"] = [i - np.searchsorted(dates, d - np.timedelta64(7, "D"), side="left") for i, d in zip(ix, dates)]
        g["starts_14d"] = [i - np.searchsorted(dates, d - np.timedelta64(14, "D"), side="left") for i, d in zip(ix, dates)]

        for metric in ["gsax_per30", "fivev5_gsax_per30", "save_pct"]:
            shifted = g[metric].shift(1)
            g[f"{metric}_r20"] = shifted.rolling(20, min_periods=5).mean()
            g[f"{metric}_r40"] = shifted.rolling(40, min_periods=8).mean()
            g[f"{metric}_recent10"] = shifted.rolling(10, min_periods=5).mean()
        g["gsax_per30_recent10_vs_r40"] = g.gsax_per30_recent10 - g.gsax_per30_r40

        # Career-to-date rates use only prior shots and shrink toward neutral:
        # league-average save percentage 0.900 and zero GSAx per 30 shots.
        prior_shots = g.shots.cumsum() - g.shots
        prior_saves = g.saves.cumsum() - g.saves
        prior_gsax = g.gsax.cumsum() - g.gsax
        g["save_pct_shrunk"] = (prior_saves + 90.0) / (prior_shots + 100.0)
        g["gsax_per30_shrunk"] = (30.0 * prior_gsax) / (prior_shots + 100.0)
        g["prior_starts_count"] = g.prior_starts.astype(float)
        history_parts.append(g[[
            "goalie_id_key", "game_date", "prior_starts_count", "days_since_start",
            "starts_7d", "starts_14d", "gsax_per30_r20", "gsax_per30_r40",
            "fivev5_gsax_per30_r20", "fivev5_gsax_per30_r40", "save_pct_r20",
            "save_pct_r40", "gsax_per30_recent10_vs_r40", "save_pct_shrunk",
            "gsax_per30_shrunk",
        ]])
    history = pd.concat(history_parts, ignore_index=True)

    joined = starters.merge(
        history,
        on=["goalie_id_key", "game_date"],
        how="left",
        validate="many_to_one",
    )
    if joined.prior_starts_count.notna().any() and (joined.prior_starts_count > 0).any():
        # For a game on the current date, matching daily history must be the
        # pre-game rolling row; it must never incorporate that date's result.
        pass

    features = [
        "prior_starts_count", "days_since_start", "starts_7d", "starts_14d",
        "gsax_per30_r20", "gsax_per30_r40", "fivev5_gsax_per30_r20",
        "fivev5_gsax_per30_r40", "save_pct_r20", "save_pct_r40",
        "gsax_per30_recent10_vs_r40", "save_pct_shrunk", "gsax_per30_shrunk",
    ]
    sides = []
    for side, label in [("home", "h"), ("away", "a")]:
        z = joined.loc[joined.home_away.eq(side), ["game_id"] + features].copy()
        z = z.rename(columns={c: f"{label}_goalie_{c}" for c in features})
        sides.append(z)
    wide = sides[0].merge(sides[1], on="game_id", how="outer", validate="one_to_one")
    for c in features:
        wide[f"d_goalie_{c}"] = wide[f"h_goalie_{c}"] - wide[f"a_goalie_{c}"]

    coverage = {
        "games_with_both_actual_starters": int(
            (wide.h_goalie_prior_starts_count.notna() & wide.a_goalie_prior_starts_count.notna()).sum()
        ),
        "game_sides_with_actual_starter": int(len(joined)),
        "game_sides_total": int(spine.game_id.nunique() * 2),
        "pregame_identity_source": "retrospective starter_bool from same-game goalie box",
        "goalie_performance_cutoff": "strictly earlier game_date; all games on current date excluded",
        "stats_shrunk": "save percentage toward .900; GSAx per 30 shots toward 0",
    }
    return wide, coverage


def score_with_goalie_features(pbp: pd.DataFrame, spine: pd.DataFrame, goalie: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    result_spine = spine.loc[
        spine.result_available.fillna(False)
        & spine.game_type.astype("string").isin(["R", "P"])
    ].copy()
    result_spine["game_id"] = normalize_game_id(result_spine.game_id)
    result_spine["game_date"] = pd.to_datetime(result_spine.game_date, errors="coerce").dt.normalize()
    result_spine["source_season"] = pd.to_numeric(result_spine.source_season, errors="coerce")
    if result_spine.game_id.duplicated().any():
        raise ValueError("Duplicate canonical result game_id")

    games = ts.add_flat_and_ewm_features(ts.game_event_table(pbp))
    games["game_id"] = normalize_game_id(games.game_id)
    meta_cols = ["game_id", "game_type", "source_season", "home_win", "game_date"]
    metadata = result_spine[meta_cols].copy()
    # Canonical spine supplies the game target; verify the PBP-derived target agrees.
    games = games.merge(metadata, on="game_id", how="inner", validate="one_to_one", suffixes=("_pbp", "_spine"))
    target_pbp = games["home_win_pbp"] if "home_win_pbp" in games else games["home_win"]
    target_spine = games["home_win_spine"] if "home_win_spine" in games else games["home_win"]
    if not np.array_equal(target_pbp.astype(int).to_numpy(), target_spine.astype(int).to_numpy()):
        raise ValueError("PBP home-win labels disagree with canonical spine")
    if "home_win_pbp" in games:
        games["home_win"] = games.pop("home_win_spine").astype(int)
        games = games.drop(columns=["home_win_pbp"])
    else:
        games["home_win"] = games.home_win.astype(int)
    if "source_season_spine" in games:
        games["source_season"] = games.source_season_spine
        games = games.drop(columns=["source_season_spine"])
    if "game_date_spine" in games:
        games["game_date"] = games.game_date_spine
        games = games.drop(columns=["game_date_spine"])

    schedule = sched.build_schedule_features(result_spine)
    games = games.merge(schedule, on="game_id", how="left", validate="one_to_one")
    goalie_wide, coverage = build_goalie_pregame_features(goalie, result_spine)
    games = games.merge(goalie_wide, on="game_id", how="left", validate="one_to_one")

    choices = sched.FOLD_CHOICES
    elo_choices = sorted(set(choice[0] for choice in choices.values()))
    elo = {f"k{k}_h{ha}_r{retain}": ts.elo_all(result_spine, k, ha, retain) for k, ha, retain in elo_choices}
    oa_choices = sorted(set(choice[1] for choice in choices.values()))
    for alpha, retain in oa_choices:
        z, _ = ts.opponent_adjusted_features(games, alpha, retain)
        games = games.merge(z, on="game_id", how="left", validate="one_to_one")

    prediction_rows, fold_rows = [], []
    for target in TARGET_SEASONS:
        val_season = target - 1
        (k, ha, retain), (alpha, oa_retain) = choices[target]
        elo_key = f"k{k}_h{ha}_r{retain}"
        oa_tag = f"a{alpha:.2f}_r{oa_retain:.2f}".replace(".", "p")
        frame = games.merge(
            elo[elo_key][["game_id", "elo_p"]], on="game_id", how="left", validate="one_to_one"
        )
        frame["elo_logit"] = np.log(
            np.clip(frame.elo_p, 1e-6, 1 - 1e-6)
            / np.clip(1 - frame.elo_p, 1e-6, 1 - 1e-6)
        )
        base = [
            "elo_logit", "d_sog_share_r20", "d_xg_share_r20", "d_goals_r20",
            "d_rest_days", f"oa_sog_diff_{oa_tag}", f"oa_xg_diff_{oa_tag}",
            "h_sched_b2b", "a_sched_b2b", "d_sched_b2b",
        ]
        train = frame.loc[frame.source_season.le(target - 2)].copy()
        val = frame.loc[frame.source_season.eq(val_season)].copy()
        test = frame.loc[frame.source_season.eq(target)].copy()
        if len(test) == 0 or len(val) < 100:
            raise RuntimeError(f"Insufficient data for target season {target}")

        specifications = {"team_strength_004_plus_b2b": base}
        for name, additions in GOALIE_VARIANTS.items():
            specifications[name] = base + additions

        for name, features in specifications.items():
            C, val_ll = sched.tune_C(train, val, features)
            fit = pd.concat([train, val], ignore_index=True)
            model = sched.fit_pipeline(fit, features, C)
            p = np.clip(model.predict_proba(test[features])[:, 1], 1e-6, 1 - 1e-6)
            m = sched.score_predictions(test.home_win, p)
            if name == "team_strength_004_plus_b2b":
                ref = state.EXPECTED_B2B_BRIER.get(target)
                if ref is not None and abs(m["brier"] - ref) > 1e-5:
                    raise RuntimeError(f"B2B baseline failed to reproduce in {target}: {m['brier']} vs {ref}")
            fold_rows.append({
                "model_variant": name, "target_season": target,
                "training_through_season": target - 2, "validation_season": val_season,
                "elo_choice": elo_key, "opponent_adjusted_choice": oa_tag,
                "regularization_C": C, "validation_log_loss": val_ll,
                **m,
            })
            for game, probability in zip(test.itertuples(index=False), p):
                prediction_rows.append({
                    "game_id": str(game.game_id), "game_date": pd.Timestamp(game.game_date).date().isoformat(),
                    "source_season": int(game.source_season), "home_team": game.home_team,
                    "away_team": game.away_team, "game_type": game.game_type,
                    "home_win": int(game.home_win), "p_home": float(probability),
                    "p_away": float(1 - probability), "model_variant": name,
                    "target_season": target, "regularization_C": C,
                    "starter_identity_state": "retrospective_actual_starter" if name != "team_strength_004_plus_b2b" else "not_used",
                    "goalie_skill_cutoff": "prior_game_date_only" if name != "team_strength_004_plus_b2b" else "not_used",
                })

    pred = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    if not np.allclose(pred.p_home + pred.p_away, 1.0, atol=1e-12):
        raise ValueError("Predicted home and away probabilities do not sum to one")
    return pred, folds, coverage


def paired_bootstrap_delta_brier(base: pd.DataFrame, candidate: pd.DataFrame, seed: int = 1103) -> dict:
    x = base[["game_id", "home_win", "p_home"]].merge(
        candidate[["game_id", "p_home"]], on="game_id", how="inner",
        validate="one_to_one", suffixes=("_base", "_candidate"),
    )
    y = x.home_win.to_numpy(dtype=float)
    delta = (y - x.p_home_candidate.to_numpy()) ** 2 - (y - x.p_home_base.to_numpy()) ** 2
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(5000, len(delta)))
    boot = delta[draws].mean(axis=1)
    return {
        "games": int(len(x)), "delta_brier": float(delta.mean()),
        "paired_game_bootstrap_95pct": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
        "resamples": 5000, "seed": seed,
        "note": "Exploratory paired-game interval; not clustered by team or season.",
    }


def main(pbp_path: str, spine_path: str, goalie_path: str, outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pbp = pd.read_parquet(pbp_path)
    spine = pd.read_parquet(spine_path)
    goalie = pd.read_parquet(goalie_path)
    pred, folds, coverage = score_with_goalie_features(pbp, spine, goalie)
    pred.to_parquet(out / "oof_predictions.parquet", index=False)
    folds.to_parquet(out / "fold_metrics.parquet", index=False)

    summary = {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "validation_design": "rolling_origin; train<=target-2; tune on target-1; test target",
        "target_seasons": TARGET_SEASONS,
        "goalie_identity_caveat": "Same-game actual starter identity is selected from the postgame goalie box. This is a retrospective upper-bound proxy for an announced pregame starter, not a timestamped pregame starter test.",
        "goalie_skill_cutoff": "All goalie performance, workload, and rest features use game dates strictly before the target date.",
        "goalie_feature_coverage": coverage,
        "models": {},
        "target_2026_paired_brier": {},
    }
    for name, g in pred.groupby("model_variant", sort=True):
        scores = sched.score_predictions(g.home_win, g.p_home)
        scores["season_metrics"] = {}
        for season, z in g.groupby("target_season"):
            scores["season_metrics"][str(int(season))] = sched.score_predictions(z.home_win, z.p_home)
        summary["models"][name] = scores

    baseline = pred.loc[pred.model_variant.eq("team_strength_004_plus_b2b")]
    target_base = baseline.loc[baseline.target_season.eq(2026)]
    for name in GOALIE_VARIANTS:
        candidate = pred.loc[(pred.model_variant.eq(name)) & pred.target_season.eq(2026)]
        summary["target_2026_paired_brier"][name] = paired_bootstrap_delta_brier(
            target_base, candidate, seed=1103 + len(summary["target_2026_paired_brier"])
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pbp", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--goalie", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.pbp, args.spine, args.goalie, args.out)
