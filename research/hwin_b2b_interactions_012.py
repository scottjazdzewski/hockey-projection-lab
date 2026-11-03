"""Chronological test of predeclared home/away B2B interactions.

The reference is the unchanged Schedule 007 Team Strength 004 + B2B model.
Candidates add (1) a both-teams-on-B2B interaction, (2) interactions between
the pregame Elo logit and each side's B2B indicator, or (3) both groups.
Every candidate uses the same rolling-origin folds and tunes regularization on
the preceding season only. No goalie, lineup, market, or same-game result fields
are used as predictors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

import hwin_schedule_007 as sched
import hwin_team_strength_004 as ts


MODEL_VERSION = "HWIN-B2B-INTERACTIONS-012-RESEARCH"
FEATURE_VERSION = "team-strength-004-plus-b2b-plus-state-and-elo-interactions-001"
SOURCE_VERSION = "sportsdataverse-pbp-features-20261102+canonical-spine-v3-20261102"
RETRIEVAL_DATE = "2026-11-03"
TARGET_SEASONS = [2022, 2023, 2024, 2025, 2026]
BASELINE_BRIER = {
    2022: 0.22825118358382493,
    2023: 0.23153785544406072,
    2024: 0.23347672774228304,
    2025: 0.23526533211413475,
    2026: 0.2451504879022673,
}

INTERACTION_SETS = {
    "b2b_interaction_both_teams": ["i_b2b_both"],
    "b2b_interaction_elo_by_side": ["i_home_b2b_elo", "i_away_b2b_elo"],
    "b2b_interaction_both_plus_elo_by_side": [
        "i_b2b_both", "i_home_b2b_elo", "i_away_b2b_elo"
    ],
}


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e8, max_iter=5000).fit(lp, y)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
    }


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "games": int(len(y)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_home_probability": float(np.mean(p)),
        "home_win_rate": float(np.mean(y)),
        **calibration_metrics(y, p),
    }


def paired_delta(base: pd.DataFrame, candidate: pd.DataFrame, seed: int) -> dict:
    x = base[["game_id", "home_win", "p_home"]].merge(
        candidate[["game_id", "p_home"]], on="game_id", how="inner",
        validate="one_to_one", suffixes=("_base", "_candidate"),
    )
    y = x.home_win.to_numpy(dtype=float)
    p0 = x.p_home_base.to_numpy(dtype=float)
    p1 = x.p_home_candidate.to_numpy(dtype=float)
    delta = (y - p1) ** 2 - (y - p0) ** 2
    correct1 = (p1 >= 0.5) == (y == 1)
    correct0 = (p0 >= 0.5) == (y == 1)
    correct_delta = correct1.astype(int) - correct0.astype(int)
    rng = np.random.default_rng(seed)
    n = len(delta)
    draws = np.empty(5000, dtype=float)
    for i in range(len(draws)):
        idx = rng.integers(0, n, n)
        draws[i] = float(np.mean(delta[idx]))
    return {
        "games": int(n),
        "delta_brier_candidate_minus_baseline": float(np.mean(delta)),
        "paired_game_bootstrap_95pct": [
            float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
        ],
        "correct_pick_change": int(np.sum(correct_delta)),
        "resamples": int(len(draws)),
        "seed": int(seed),
        "interval_note": "Exploratory paired-game interval; not clustered by team or season.",
    }


def build_model_frame(pbp_path: str, spine_path: str) -> tuple[pd.DataFrame, dict]:
    pbp = pd.read_parquet(pbp_path)
    raw_spine = pd.read_parquet(spine_path)
    spine = raw_spine.loc[
        raw_spine.result_available.fillna(False)
        & raw_spine.game_type.astype("string").isin(["R", "P"])
    ].copy()
    if spine.game_id.duplicated().any():
        raise ValueError("Canonical result spine has duplicate game_id values")
    spine["game_date"] = pd.to_datetime(spine.game_date, errors="coerce").dt.normalize()
    spine["source_season"] = pd.to_numeric(spine.source_season, errors="coerce")
    if spine.game_date.isna().any() or spine.source_season.isna().any():
        raise ValueError("Canonical result spine has missing date/season values")

    games = ts.add_flat_and_ewm_features(ts.game_event_table(pbp))
    metadata_cols = ["game_id", "game_type"]
    for col in ["home_team_id", "away_team_id"]:
        if col in spine:
            metadata_cols.append(col)
    games = games.merge(
        spine[metadata_cols], on="game_id", how="inner", validate="one_to_one"
    )
    schedule = sched.build_schedule_features(spine)
    games = games.merge(schedule, on="game_id", how="left", validate="one_to_one")

    selected_elo = sorted({choice[0] for choice in sched.FOLD_CHOICES.values()})
    elo = {}
    for k, ha, retain in selected_elo:
        key = f"k{k}_h{ha}_r{retain}"
        elo[key] = ts.elo_all(spine, k, ha, retain)
    selected_oa = sorted({choice[1] for choice in sched.FOLD_CHOICES.values()})
    for alpha, retain in selected_oa:
        oa_features, _ = ts.opponent_adjusted_features(games, alpha, retain)
        games = games.merge(oa_features, on="game_id", how="left", validate="one_to_one")

    return games, {"spine": spine, "elo": elo}


def run(pbp_path: str, spine_path: str, outdir: str) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    games, context = build_model_frame(pbp_path, spine_path)
    spine, elo = context["spine"], context["elo"]
    prediction_rows, fold_rows = [], []

    for target in TARGET_SEASONS:
        val_season = target - 1
        (k, ha, retain), (alpha, oa_retain) = sched.FOLD_CHOICES[target]
        elo_key = f"k{k}_h{ha}_r{retain}"
        oa_tag = f"a{alpha:.2f}_r{oa_retain:.2f}".replace(".", "p")
        frame = games.merge(
            elo[elo_key][["game_id", "elo_p"]], on="game_id", how="left",
            validate="one_to_one",
        )
        frame["elo_logit"] = np.log(
            np.clip(frame.elo_p, 1e-6, 1 - 1e-6)
            / np.clip(1 - frame.elo_p, 1e-6, 1 - 1e-6)
        )
        frame["i_b2b_both"] = frame.h_sched_b2b * frame.a_sched_b2b
        frame["i_home_b2b_elo"] = frame.h_sched_b2b * frame.elo_logit
        frame["i_away_b2b_elo"] = frame.a_sched_b2b * frame.elo_logit

        train = frame.loc[frame.source_season.le(target - 2)].copy()
        val = frame.loc[frame.source_season.eq(val_season)].copy()
        test = frame.loc[frame.source_season.eq(target)].copy()
        if len(test) == 0 or len(val) < 100:
            raise RuntimeError(f"Insufficient chronological data for target {target}")

        base_features = [
            "elo_logit", "d_sog_share_r20", "d_xg_share_r20", "d_goals_r20",
            "d_rest_days", f"oa_sog_diff_{oa_tag}", f"oa_xg_diff_{oa_tag}",
            "h_sched_b2b", "a_sched_b2b", "d_sched_b2b",
        ]
        variants = {"team_strength_004_plus_b2b": base_features}
        for name, additions in INTERACTION_SETS.items():
            variants[name] = base_features + additions

        for name, features in variants.items():
            C, val_ll = sched.tune_C(train, val, features)
            fit_data = pd.concat([train, val], ignore_index=True)
            fitted = sched.fit_pipeline(fit_data, features, C)
            p_home = np.clip(fitted.predict_proba(test[features])[:, 1], 1e-6, 1 - 1e-6)
            fold_metric = metrics(test.home_win.to_numpy(), p_home)
            if name == "team_strength_004_plus_b2b":
                expected = BASELINE_BRIER[target]
                delta = fold_metric["brier"] - expected
                if abs(delta) > 1e-5:
                    raise RuntimeError(
                        f"B2B reference did not reproduce for {target}: "
                        f"{fold_metric['brier']:.9f} vs {expected:.9f}"
                    )
            fold_rows.append({
                "variant": name,
                "target_season": target,
                "training_through_season": target - 2,
                "validation_season": val_season,
                "elo_choice": elo_key,
                "opponent_adjusted_choice": oa_tag,
                "regularization_C": C,
                "validation_log_loss": val_ll,
                **fold_metric,
            })
            for game, probability in zip(test.itertuples(index=False), p_home):
                prediction_rows.append({
                    "game_id": str(game.game_id),
                    "game_date": pd.Timestamp(game.game_date).date().isoformat(),
                    "source_season": int(game.source_season),
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "game_type": game.game_type,
                    "home_win": int(game.home_win),
                    "h_b2b": int(game.h_sched_b2b),
                    "a_b2b": int(game.a_sched_b2b),
                    "b2b_state": (
                        "both" if game.h_sched_b2b and game.a_sched_b2b else
                        "home_only" if game.h_sched_b2b else
                        "away_only" if game.a_sched_b2b else "neither"
                    ),
                    "p_home": float(probability),
                    "p_away": float(1 - probability),
                    "variant": name,
                    "model_version": MODEL_VERSION,
                    "feature_version": FEATURE_VERSION,
                    "source_version": SOURCE_VERSION,
                    "retrieval_date": RETRIEVAL_DATE,
                    "feature_cutoff_date": pd.Timestamp(game.game_date).date().isoformat(),
                    "data_quality_flag": "verified_result;strict_prior_date_schedule_and_ratings",
                    "target_season": target,
                    "training_through_season": target - 2,
                    "validation_season": val_season,
                    "regularization_C": C,
                })

    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    if predictions.duplicated(["game_id", "variant"]).any():
        raise ValueError("Duplicate prediction rows found")
    if not np.allclose(predictions.p_home + predictions.p_away, 1.0, atol=1e-12):
        raise ValueError("Home and away probabilities do not sum to one")

    predictions.to_parquet(out / "oof_predictions.parquet", index=False)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    summary = {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "source_version": SOURCE_VERSION,
        "retrieval_date": RETRIEVAL_DATE,
        "target_seasons": TARGET_SEASONS,
        "validation_design": "rolling origin; train<=S-2; tune on S-1; score S",
        "interaction_specs": INTERACTION_SETS,
        "starter_identity": "not used",
        "markets": "not used",
        "games_per_variant": int(predictions.game_id.nunique()),
        "models": {},
        "paired_brier_vs_baseline": {},
        "paired_brier_by_season_vs_baseline": {},
        "2026_b2b_state_diagnostics": {},
    }
    baseline = predictions.loc[predictions.variant.eq("team_strength_004_plus_b2b")]
    for name, group in predictions.groupby("variant", sort=True):
        summary["models"][name] = {
            **metrics(group.home_win.to_numpy(), group.p_home.to_numpy()),
            "season_metrics": {
                str(int(season)): metrics(g.home_win.to_numpy(), g.p_home.to_numpy())
                for season, g in group.groupby("target_season", sort=True)
            },
        }
        if name != "team_strength_004_plus_b2b":
            summary["paired_brier_vs_baseline"][name] = paired_delta(
                baseline, group, seed=1200 + len(summary["paired_brier_vs_baseline"])
            )
            summary["paired_brier_by_season_vs_baseline"][name] = {}
            for season_index, season in enumerate(TARGET_SEASONS):
                base_season = baseline.loc[baseline.target_season.eq(season)]
                candidate_season = group.loc[group.target_season.eq(season)]
                summary["paired_brier_by_season_vs_baseline"][name][str(season)] = paired_delta(
                    base_season, candidate_season,
                    seed=1300 + 10 * len(summary["paired_brier_by_season_vs_baseline"]) + season_index,
                )

    season_2026 = predictions.loc[predictions.target_season.eq(2026)]
    for state, state_group in season_2026.groupby("b2b_state", sort=True):
        state_summary = {"games": int(state_group.game_id.nunique())}
        for name, model_group in state_group.groupby("variant", sort=True):
            state_summary[name] = metrics(
                model_group.home_win.to_numpy(), model_group.p_home.to_numpy()
            )
        summary["2026_b2b_state_diagnostics"][state] = state_summary

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbp", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.pbp, args.spine, args.out)
