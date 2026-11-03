"""Test prior-game tied/close 5v5 xG shares beyond Team Strength 004 + B2B.

Feature rates use only prior team games. Current-game PBP values are never
joined into that game's pregame row. The experiment uses the same chronological
fold choices and validation protocol as Schedule 007.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

import hwin_schedule_007 as sched
import hwin_team_strength_004 as ts

MODEL_VERSION = "HWIN-SCORE-STATE-009-NOGOALIE-RESEARCH"
FEATURE_VERSION = "pregame-team-strength-004-plus-b2b-plus-state-5v5-xg-001"
SOURCE_VERSION = "sportsdataverse-canonical-pbp-features-20261102+canonical-spine-v3-20261102"
RETRIEVAL_DATE = "2026-11-03"
TARGET_SEASONS = [2022, 2023, 2024, 2025, 2026]
EXPECTED_B2B_BRIER = {
    2022: 0.22825118358382493,
    2023: 0.23153785544406072,
    2024: 0.23347672774228304,
    2025: 0.23526533211413475,
    2026: 0.24515048790226730,
}


def check_same_day_event_order(pbp: pd.DataFrame, spine: pd.DataFrame) -> None:
    """Ensure event sort by game ID does not reverse any same-day team games."""
    ids = spine[["game_id", "game_date", "game_time", "home_team_abbr", "away_team_abbr"]].copy()
    ids["game_date"] = pd.to_datetime(ids.game_date, errors="coerce").dt.normalize()
    ids["game_time"] = pd.to_datetime(ids.game_time, utc=True, errors="coerce")
    h = ids.rename(columns={"home_team_abbr": "team", "away_team_abbr": "opp"})
    a = ids.rename(columns={"away_team_abbr": "team", "home_team_abbr": "opp"})
    long = pd.concat([h, a], ignore_index=True)
    multi = long.groupby(["team", "game_date"]).filter(lambda g: g.game_id.nunique() > 1)
    if multi.empty:
        return
    for (_, _), group in multi.groupby(["team", "game_date"]):
        g = group.sort_values("game_time")
        if g.game_time.isna().any() or not g.game_id.is_monotonic_increasing:
            raise ValueError("Same-day team games lack a reliable game_id/time order")


def build_state_share_features(pbp: pd.DataFrame) -> pd.DataFrame:
    required = [
        "game_id", "game_date", "game_type", "event_team_abbr", "home_away",
        "tied_fivev5_xg", "tied_fivev5_xg_against",
        "close_fivev5_xg", "close_fivev5_xg_against",
    ]
    missing = sorted(set(required) - set(pbp.columns))
    if missing:
        raise ValueError(f"Missing score-state PBP fields: {missing}")
    d = pbp.loc[pbp.game_type.astype("string").isin(["R", "P"]), required].copy()
    d["game_date"] = pd.to_datetime(d.game_date, errors="coerce").dt.normalize()
    if d.game_date.isna().any() or d.duplicated(["game_id", "event_team_abbr"]).any():
        raise ValueError("Canonical PBP dates or game/team keys are invalid")
    for c in required[5:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(["event_team_abbr", "game_date", "game_id"]).reset_index(drop=True)
    features = []
    for state in ["tied", "close"]:
        f = f"{state}_fivev5_xg"
        a = f"{state}_fivev5_xg_against"
        gf = d.groupby("event_team_abbr", sort=False)[f].transform(
            lambda s: s.shift(1).rolling(20, min_periods=8).sum()
        )
        ga = d.groupby("event_team_abbr", sort=False)[a].transform(
            lambda s: s.shift(1).rolling(20, min_periods=8).sum()
        )
        d[f"{state}_xg_share_r20"] = np.where(gf + ga > 0, gf / (gf + ga), np.nan)
        features.append(f"{state}_xg_share_r20")
    h = d[d.home_away.eq("home")][["game_id"] + features].rename(
        columns={c: f"h_{c}" for c in features}
    )
    a = d[d.home_away.eq("away")][["game_id"] + features].rename(
        columns={c: f"a_{c}" for c in features}
    )
    out = h.merge(a, on="game_id", how="inner", validate="one_to_one")
    for c in features:
        out[f"d_{c}"] = pd.to_numeric(out[f"h_{c}"], errors="coerce") - pd.to_numeric(
            out[f"a_{c}"], errors="coerce"
        )
    return out


def main(pbp_path: str, spine_path: str, outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pbp = pd.read_parquet(pbp_path)
    spine_all = pd.read_parquet(spine_path)
    check_same_day_event_order(
        pbp,
        spine_all.loc[spine_all.game_type.astype("string").isin(["R", "P"])].copy(),
    )
    spine = spine_all.loc[
        spine_all.result_available.fillna(False)
        & spine_all.game_type.astype("string").isin(["R", "P"])
    ].copy()
    if spine.game_id.duplicated().any():
        raise ValueError("Canonical result spine contains duplicate game_id values")
    spine["game_date"] = pd.to_datetime(spine.game_date, errors="coerce").dt.normalize()
    spine["source_season"] = pd.to_numeric(spine.source_season, errors="coerce")

    games = ts.add_flat_and_ewm_features(ts.game_event_table(pbp))
    metadata_cols = ["game_id", "game_type"]
    if "game_time" in spine.columns:
        metadata_cols.append("game_time")
    for c in ["home_team_id", "away_team_id"]:
        if c in spine.columns:
            metadata_cols.append(c)
    games = games.merge(spine[metadata_cols], on="game_id", how="inner", validate="one_to_one")
    games = games.merge(sched.build_schedule_features(spine), on="game_id", how="left", validate="one_to_one")
    games = games.merge(build_state_share_features(pbp), on="game_id", how="left", validate="one_to_one")

    selected_elo = sorted(set(choice[0] for choice in sched.FOLD_CHOICES.values()))
    elo = {f"k{k}_h{ha}_r{retain}": ts.elo_all(spine, k, ha, retain) for k, ha, retain in selected_elo}
    selected_oa = sorted(set(choice[1] for choice in sched.FOLD_CHOICES.values()))
    for alpha, retain in selected_oa:
        z, _ = ts.opponent_adjusted_features(games, alpha, retain)
        games = games.merge(z, on="game_id", how="left", validate="one_to_one")

    b2b = ["h_sched_b2b", "a_sched_b2b", "d_sched_b2b"]
    common = [
        "elo_logit", "d_sog_share_r20", "d_xg_share_r20", "d_goals_r20", "d_rest_days",
    ]
    prediction_rows, fold_rows = [], []
    for target in TARGET_SEASONS:
        val_season = target - 1
        (k, ha, retain), (alpha, oa_retain) = sched.FOLD_CHOICES[target]
        elo_key = f"k{k}_h{ha}_r{retain}"
        oa_tag = f"a{alpha:.2f}_r{oa_retain:.2f}".replace(".", "p")
        frame = games.merge(elo[elo_key][["game_id", "elo_p"]], on="game_id", how="left", validate="one_to_one")
        frame["elo_logit"] = np.log(
            np.clip(frame.elo_p, 1e-6, 1 - 1e-6) / np.clip(1 - frame.elo_p, 1e-6, 1 - 1e-6)
        )
        train = frame.loc[frame.source_season.le(target - 2)].copy()
        val = frame.loc[frame.source_season.eq(val_season)].copy()
        test = frame.loc[frame.source_season.eq(target)].copy()
        if len(test) == 0 or len(val) < 100:
            raise RuntimeError(f"Insufficient chronological data for target season {target}")
        baseline_features = common + [f"oa_sog_diff_{oa_tag}", f"oa_xg_diff_{oa_tag}"]
        variants = {
            "team_strength_004_plus_b2b": baseline_features + b2b,
            "plus_tied_state_xg_share": baseline_features + b2b + ["d_tied_xg_share_r20"],
            "plus_close_state_xg_share": baseline_features + b2b + ["d_close_xg_share_r20"],
            "plus_tied_and_close_xg_share": baseline_features + b2b + [
                "d_tied_xg_share_r20", "d_close_xg_share_r20",
            ],
        }
        for name, features in variants.items():
            C, val_ll = sched.tune_C(train, val, features)
            fit = pd.concat([train, val], ignore_index=True)
            model = sched.fit_pipeline(fit, features, C)
            p = model.predict_proba(test[features])[:, 1]
            fold = sched.score_predictions(test.home_win, p)
            if name == "team_strength_004_plus_b2b":
                expected = EXPECTED_B2B_BRIER[target]
                if abs(fold["brier"] - expected) > 1e-5:
                    raise RuntimeError(f"B2B baseline did not reproduce in {target}")
            fold_rows.append({
                "model_variant": name, "target_season": target, "validation_season": val_season,
                "training_through_season": target - 2, "regularization_C": C,
                "validation_log_loss": val_ll, "elo_choice": elo_key,
                "opponent_adjusted_choice": oa_tag, **fold,
            })
            for game, probability in zip(test.itertuples(index=False), p):
                prediction_rows.append({
                    "game_id": game.game_id,
                    "game_date": pd.Timestamp(game.game_date).date().isoformat(),
                    "source_season": int(game.source_season),
                    "home_team_abbr": game.home_team,
                    "away_team_abbr": game.away_team,
                    "game_type": game.game_type,
                    "game_start_time": str(game.game_time) if hasattr(game, "game_time") else None,
                    "home_tied_xg_share_r20": getattr(game, "h_tied_xg_share_r20"),
                    "away_tied_xg_share_r20": getattr(game, "a_tied_xg_share_r20"),
                    "home_close_xg_share_r20": getattr(game, "h_close_xg_share_r20"),
                    "away_close_xg_share_r20": getattr(game, "a_close_xg_share_r20"),
                    "home_win": int(game.home_win),
                    "p_home": float(probability),
                    "p_away": float(1 - probability),
                    "model_variant": name,
                    "model_version": MODEL_VERSION,
                    "feature_version": FEATURE_VERSION,
                    "source_version": SOURCE_VERSION,
                    "retrieval_date": RETRIEVAL_DATE,
                    "feature_cutoff_date": pd.Timestamp(game.game_date).date().isoformat(),
                    "feature_cutoff_timestamp": str(game.game_time) if hasattr(game, "game_time") else None,
                    "data_quality_flag": "verified_winner;lagged_state_xg_share;prior_team_games",
                    "target_season": target,
                    "training_through_season": target - 2,
                    "validation_season": val_season,
                    "regularization_C": C,
                })

    pred = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    if not np.allclose(pred.p_home + pred.p_away, 1.0, atol=1e-12):
        raise ValueError("Home and away probabilities do not sum to one")
    pred.to_parquet(out / "oof_predictions.parquet", index=False)
    folds.to_parquet(out / "fold_metrics.parquet", index=False)
    sched.reliability_table(pred).to_parquet(out / "reliability_bins.parquet", index=False)
    baseline_name = "team_strength_004_plus_b2b"
    challenger_name = "plus_tied_state_xg_share"
    wide = pred.loc[pred.model_variant.isin([baseline_name, challenger_name])].pivot(
        index=["game_id", "game_type", "home_win", "target_season"],
        columns="model_variant", values="p_home",
    ).reset_index()
    subgroup_rows = []
    for group_col in ["game_type", "target_season"]:
        for group_value, g in wide.groupby(group_col, dropna=False):
            b = sched.score_predictions(g.home_win, g[baseline_name])
            c = sched.score_predictions(g.home_win, g[challenger_name])
            subgroup_rows.append({
                "group_type": group_col, "group_value": str(group_value), "games": len(g),
                "baseline_brier": b["brier"], "challenger_brier": c["brier"],
                "delta_brier": c["brier"] - b["brier"],
                "baseline_log_loss": b["log_loss"], "challenger_log_loss": c["log_loss"],
                "delta_log_loss": c["log_loss"] - b["log_loss"],
                "baseline_accuracy": b["accuracy"], "challenger_accuracy": c["accuracy"],
            })
    wide["baseline_probability_decile"] = pd.qcut(
        wide[baseline_name], q=10, labels=False, duplicates="drop"
    ) + 1
    for decile, g in wide.groupby("baseline_probability_decile", dropna=True):
        b = sched.score_predictions(g.home_win, g[baseline_name])
        c = sched.score_predictions(g.home_win, g[challenger_name])
        subgroup_rows.append({
            "group_type": "baseline_probability_decile", "group_value": str(int(decile)), "games": len(g),
            "baseline_brier": b["brier"], "challenger_brier": c["brier"],
            "delta_brier": c["brier"] - b["brier"],
            "baseline_log_loss": b["log_loss"], "challenger_log_loss": c["log_loss"],
            "delta_log_loss": c["log_loss"] - b["log_loss"],
            "baseline_accuracy": b["accuracy"], "challenger_accuracy": c["accuracy"],
        })
    pd.DataFrame(subgroup_rows).to_parquet(out / "subgroup_diagnostics.parquet", index=False)
    summary = {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "source_version": SOURCE_VERSION,
        "retrieval_date": RETRIEVAL_DATE,
        "validation_design": "rolling-origin; train<=target-2; tune on target-1; test target",
        "target_seasons": TARGET_SEASONS,
        "feature_note": "Rolling 20 prior team games; share is the ratio of tied/close 5v5 xG for to total state xG for and against.",
        "models": {},
    }
    for name, group in pred.groupby("model_variant", sort=True):
        score = sched.score_predictions(group.home_win, group.p_home)
        score.update(sched.calibration_metrics(group.home_win, group.p_home))
        score["season_brier"] = {
            str(int(season)): float(brier_score_loss(g.home_win, g.p_home))
            for season, g in group.groupby("target_season")
        }
        summary["models"][name] = score
    tied = pred.loc[pred.model_variant.eq("plus_tied_state_xg_share")].merge(
        pred.loc[pred.model_variant.eq("team_strength_004_plus_b2b"), ["game_id", "p_home"]].rename(
            columns={"p_home": "baseline_p_home"}
        ),
        on="game_id", how="inner", validate="one_to_one",
    )
    y = tied.home_win.to_numpy(dtype=float)
    p_candidate = tied.p_home.to_numpy(dtype=float)
    p_baseline = tied.baseline_p_home.to_numpy(dtype=float)
    paired_delta = (y - p_candidate) ** 2 - (y - p_baseline) ** 2
    rng = np.random.default_rng(9009)
    boot = np.empty(5000)
    for i in range(len(boot)):
        boot[i] = paired_delta[rng.integers(0, len(paired_delta), len(paired_delta))].mean()
    summary["paired_game_bootstrap_diagnostic"] = {
        "comparison": "plus_tied_state_xg_share minus team_strength_004_plus_b2b",
        "delta_brier": float(paired_delta.mean()),
        "resamples": len(boot),
        "seed": 9009,
        "delta_brier_95pct_interval": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "interpretation": "Exploratory paired-game bootstrap; interval is not cluster-adjusted for repeated teams or seasons.",
    }
    summary["test_feature_availability"] = {
        "home_tied_xg_share_r20": float(pred.loc[pred.model_variant.eq("plus_tied_state_xg_share"), "home_tied_xg_share_r20"].notna().mean()),
        "away_tied_xg_share_r20": float(pred.loc[pred.model_variant.eq("plus_tied_state_xg_share"), "away_tied_xg_share_r20"].notna().mean()),
        "home_close_xg_share_r20": float(pred.loc[pred.model_variant.eq("plus_tied_state_xg_share"), "home_close_xg_share_r20"].notna().mean()),
        "away_close_xg_share_r20": float(pred.loc[pred.model_variant.eq("plus_tied_state_xg_share"), "away_close_xg_share_r20"].notna().mean()),
    }
    baseline = summary["models"]["team_strength_004_plus_b2b"]
    for name, score in summary["models"].items():
        score["delta_brier_vs_b2b"] = score["brier"] - baseline["brier"]
        if name != "team_strength_004_plus_b2b":
            score["seasons_better_than_b2b"] = sum(
                score["season_brier"][s] < baseline["season_brier"][s]
                for s in baseline["season_brier"]
            )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--pbp", required=True)
    p.add_argument("--spine", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    main(a.pbp, a.spine, a.out)
