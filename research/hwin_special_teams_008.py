"""Leakage-safe test of lagged special-teams form on the HWIN game baseline.

Power-play opportunity counts are not present in the compiled event features.
Opponent penalty events are used as a noisy exposure proxy and are kept
separate from per-opportunity xG quality.  All team-date features use events
strictly before the current game date; no same-date result updates history.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

import hwin_schedule_007 as sched
import hwin_team_strength_004 as ts

MODEL_VERSION = "HWIN-SPECIAL-TEAMS-008-NOGOALIE-RESEARCH"
FEATURE_VERSION = "pregame-team-strength-004-plus-b2b-plus-lagged-special-teams-001"
SOURCE_VERSION = "sportsdataverse-canonical-pbp-features-20261102+canonical-spine-v3-20261102"
RETRIEVAL_DATE = "2026-11-03"
TARGET_SEASONS = [2022, 2023, 2024, 2025, 2026]


def build_special_teams_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """Build team PP/PK strength features from prior dates only.

    Penalty events committed by the opponent proxy PP opportunity volume;
    penalties committed by the team proxy PK exposure. These are noisy
    denominators, not official special-teams opportunity counts.
    """
    required = [
        "game_id", "game_date", "game_type", "event_team_abbr", "home_away",
        "pp_xg", "pp_goals", "pp_xg_against", "pp_goals_against", "penalties",
    ]
    missing = sorted(set(required) - set(pbp.columns))
    if missing:
        raise ValueError(f"Missing canonical special-teams fields: {missing}")
    d = pbp.loc[pbp.game_type.astype("string").isin(["R", "P"]), required].copy()
    d["game_date"] = pd.to_datetime(d.game_date, errors="coerce").dt.normalize()
    if d.game_date.isna().any():
        raise ValueError("PBP game dates are required to enforce pregame feature cutoffs")
    if d.duplicated(["game_id", "event_team_abbr"]).any():
        raise ValueError("Canonical PBP has duplicate game/team rows")
    side_count = d.groupby("game_id").size()
    if not side_count.eq(2).all():
        raise ValueError("Canonical PBP must have exactly two team rows per game")

    for c in ["pp_xg", "pp_goals", "pp_xg_against", "pp_goals_against", "penalties"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # The opponent's penalty count is the only available exposure proxy.
    opponent = d[["game_id", "home_away", "penalties"]].copy()
    opponent["home_away"] = opponent.home_away.map({"home": "away", "away": "home"})
    opponent = opponent.rename(columns={"penalties": "opp_penalties"})
    d = d.merge(opponent, on=["game_id", "home_away"], how="left", validate="one_to_one")
    d = d.rename(columns={"event_team_abbr": "team"})

    # Collapse rare same-day doubleheaders to one team-date observation. Every
    # game on that date receives the state before the date is added to history.
    raw = ["pp_xg", "pp_goals", "pp_xg_against", "pp_goals_against", "penalties", "opp_penalties"]
    daily = d.groupby(["team", "game_date"], as_index=False, sort=True)[raw].mean()
    for c in raw:
        daily[f"ewm_{c}"] = daily.groupby("team", sort=False)[c].transform(
            lambda s: s.shift(1).ewm(halflife=20, adjust=False, min_periods=8).mean()
        )

    # Stabilized rates; the half-event pseudocount damps early/low-exposure
    # teams. Exposure and finishing residual are reported as separate groups.
    daily["pp_xg_per_opp_penalty"] = daily.ewm_pp_xg / (daily.ewm_opp_penalties + 0.5)
    daily["pk_xg_allowed_per_own_penalty"] = daily.ewm_pp_xg_against / (daily.ewm_penalties + 0.5)
    daily["pp_finishing_resid"] = (
        daily.ewm_pp_goals - daily.ewm_pp_xg
    ) / (daily.ewm_opp_penalties + 0.5)
    daily["pk_finishing_allowed_resid"] = (
        daily.ewm_pp_goals_against - daily.ewm_pp_xg_against
    ) / (daily.ewm_penalties + 0.5)
    d = d.merge(
        daily[["team", "game_date", "pp_xg_per_opp_penalty", "pk_xg_allowed_per_own_penalty",
               "ewm_opp_penalties", "ewm_penalties", "pp_finishing_resid",
               "pk_finishing_allowed_resid"]],
        on=["team", "game_date"], how="left", validate="many_to_one",
    )

    feature_cols = [
        "pp_xg_per_opp_penalty", "pk_xg_allowed_per_own_penalty",
        "ewm_opp_penalties", "ewm_penalties",
        "pp_finishing_resid", "pk_finishing_allowed_resid",
    ]
    h = d[d.home_away.eq("home")][["game_id"] + feature_cols].rename(
        columns={c: f"h_{c}" for c in feature_cols}
    )
    a = d[d.home_away.eq("away")][["game_id"] + feature_cols].rename(
        columns={c: f"a_{c}" for c in feature_cols}
    )
    out = h.merge(a, on="game_id", how="inner", validate="one_to_one")
    for c in feature_cols:
        out[f"d_{c}"] = pd.to_numeric(out[f"h_{c}"], errors="coerce") - pd.to_numeric(
            out[f"a_{c}"], errors="coerce"
        )
    return out


def main(pbp_path: str, spine_path: str, outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pbp = pd.read_parquet(pbp_path)
    spine = pd.read_parquet(spine_path)
    spine = spine[
        spine.result_available.fillna(False)
        & spine.game_type.astype("string").isin(["R", "P"])
    ].copy()
    if spine.game_id.duplicated().any():
        raise ValueError("Canonical result spine contains duplicate game_id values")
    spine["game_date"] = pd.to_datetime(spine.game_date, errors="coerce").dt.normalize()
    spine["source_season"] = pd.to_numeric(spine.source_season, errors="coerce")

    games = ts.add_flat_and_ewm_features(ts.game_event_table(pbp))
    metadata_cols = ["game_id", "game_type"]
    for c in ["home_team_id", "away_team_id"]:
        if c in spine.columns:
            metadata_cols.append(c)
    games = games.merge(spine[metadata_cols], on="game_id", how="inner", validate="one_to_one")
    games = games.merge(sched.build_schedule_features(spine), on="game_id", how="left", validate="one_to_one")
    games = games.merge(build_special_teams_features(pbp), on="game_id", how="left", validate="one_to_one")

    choices = sched.FOLD_CHOICES
    selected_elo = sorted(set(choice[0] for choice in choices.values()))
    elo = {f"k{k}_h{ha}_r{retain}": ts.elo_all(spine, k, ha, retain) for k, ha, retain in selected_elo}
    selected_oa = sorted(set(choice[1] for choice in choices.values()))
    for alpha, retain in selected_oa:
        z, _ = ts.opponent_adjusted_features(games, alpha, retain)
        games = games.merge(z, on="game_id", how="left", validate="one_to_one")

    b2b = ["h_sched_b2b", "a_sched_b2b", "d_sched_b2b"]
    base_st = [
        "d_pp_xg_per_opp_penalty", "d_pk_xg_allowed_per_own_penalty",
    ]
    exposure = ["d_ewm_opp_penalties", "d_ewm_penalties"]
    finishing = ["d_pp_finishing_resid", "d_pk_finishing_allowed_resid"]

    prediction_rows, fold_rows = [], []
    for target in TARGET_SEASONS:
        val_season = target - 1
        (k, ha, retain), (alpha, oa_retain) = choices[target]
        elo_key = f"k{k}_h{ha}_r{retain}"
        oa_tag = f"a{alpha:.2f}_r{oa_retain:.2f}".replace(".", "p")
        frame = games.merge(elo[elo_key][["game_id", "elo_p"]], on="game_id", how="left", validate="one_to_one")
        frame["elo_logit"] = np.log(
            np.clip(frame.elo_p, 1e-6, 1 - 1e-6) / np.clip(1 - frame.elo_p, 1e-6, 1 - 1e-6)
        )
        frame = frame.merge(spine[["game_id", "source_season"]], on="game_id", how="left", validate="one_to_one", suffixes=("", "_spine"))
        train = frame.loc[frame.source_season.le(target - 2)].copy()
        val = frame.loc[frame.source_season.eq(val_season)].copy()
        test = frame.loc[frame.source_season.eq(target)].copy()
        if len(test) == 0 or len(val) < 100:
            raise RuntimeError(f"Insufficient chronological data for target season {target}")

        common = [
            "elo_logit", "d_sog_share_r20", "d_xg_share_r20", "d_goals_r20", "d_rest_days",
            f"oa_sog_diff_{oa_tag}", f"oa_xg_diff_{oa_tag}",
        ]
        variants = {
            "team_strength_004": common,
            "team_strength_004_plus_b2b": common + b2b,
            "plus_pp_xg_quality": common + b2b + base_st,
            "plus_pp_xg_and_exposure": common + b2b + base_st + exposure,
            "plus_pp_full_including_finishing": common + b2b + base_st + exposure + finishing,
        }
        for name, features in variants.items():
            C, val_ll = sched.tune_C(train, val, features)
            fit = pd.concat([train, val], ignore_index=True)
            model = sched.fit_pipeline(fit, features, C)
            p = model.predict_proba(test[features])[:, 1]
            fold = sched.score_predictions(test.home_win, p)
            if name == "team_strength_004":
                ref = sched.TEAM_STRENGTH_004_BRIER[target]
                fold["baseline_reproduction_delta"] = fold["brier"] - ref
                if abs(fold["baseline_reproduction_delta"]) > 1e-5:
                    raise RuntimeError(f"Team Strength 004 failed to reproduce in {target}")
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
                    "home_win": int(game.home_win),
                    "p_home": float(probability),
                    "p_away": float(1 - probability),
                    "model_variant": name,
                    "model_version": MODEL_VERSION,
                    "feature_version": FEATURE_VERSION,
                    "source_version": SOURCE_VERSION,
                    "retrieval_date": RETRIEVAL_DATE,
                    "feature_cutoff_date": pd.Timestamp(game.game_date).date().isoformat(),
                    "data_quality_flag": "verified_winner;prior-date-special-teams;penalty-events-exposure-proxy",
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
    summary = {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "source_version": SOURCE_VERSION,
        "retrieval_date": RETRIEVAL_DATE,
        "validation_design": "rolling_origin; train<=target-2; tune on target-1; test target",
        "target_seasons": TARGET_SEASONS,
        "feature_note": "PP xG per opponent penalty events; PK xG allowed per own penalty events; noisy opportunity proxy, not official power-play opportunities.",
        "models": {},
    }
    for name, group in pred.groupby("model_variant", sort=True):
        s = sched.score_predictions(group.home_win, group.p_home)
        s.update(sched.calibration_metrics(group.home_win, group.p_home))
        s["season_brier"] = {
            str(int(season)): float(brier_score_loss(g.home_win, g.p_home))
            for season, g in group.groupby("target_season")
        }
        summary["models"][name] = s
    baseline = summary["models"]["team_strength_004_plus_b2b"]["brier"]
    base_by_year = summary["models"]["team_strength_004_plus_b2b"]["season_brier"]
    for name, m in summary["models"].items():
        m["delta_brier_vs_team_strength_004_plus_b2b"] = m["brier"] - baseline
        if name not in {"team_strength_004", "team_strength_004_plus_b2b"}:
            m["seasons_better_than_team_strength_004_plus_b2b"] = sum(
                m["season_brier"][s] < base_by_year[s] for s in base_by_year
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
