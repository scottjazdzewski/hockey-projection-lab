"""Leakage-safe NHL regulation-goal and score-distribution research model.

This layer is separate from Hockey Lab Game v1.0.0. It models regulation
team-goal counts from lagged hockey evidence, then compares independent
Poisson, negative-binomial, bivariate-Poisson, and Dixon-Coles score laws.
Overtime probability is derived from regulation tie mass. A training-only
home overtime-win prior converts tied regulation scores to official final
scores. No win-model probabilities, markets, same-game goalie identity, or
same-game lineups are predictors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit, gammaln, logit
from scipy.stats import nbinom, poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import hwin_schedule_007 as sched
import hwin_team_strength_004 as ts


MODEL_VERSION = "HOCKEY-LAB-GOALS-NESTED-WINDOW-SELECTION-010-RESEARCH"
FEATURE_VERSION = "pregame-goals-process-nested-team-windows-20-40-60-010"
SOURCE_VERSION = "sportsdataverse-pbp-features-20261102+canonical-spine-v3-20261102"
RETRIEVAL_DATE = "2026-11-03"
MIN_SOURCE_SEASON = 2016
TARGET_SEASONS = [2022, 2023, 2024, 2025, 2026]
ROLLING_WINDOW = 20
ROLLING_WINDOWS = [20, 40, 60]
LEAGUE_SCORING_WINDOW = 20
ROLLING_MIN_PERIODS = 8
MAX_GOALS = 25
POISSON_ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0]
DISTROS = {
    "independent_poisson": [0.0],
    "independent_negative_binomial": [0.02, 0.05, 0.1, 0.2],
    "bivariate_poisson": [0.0, 0.03, 0.06, 0.1, 0.15],
    "dixon_coles": [-0.15, -0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05],
    "diagonal_inflated_poisson": [0.8, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8],
}
FEATURE_SETS = {
    "home_ice_only": ["is_home"],
    "team_goal_form": [
        "is_home", "team_r_reg_gf", "opp_r_reg_ga", "rest_days", "b2b",
    ],
    "goal_form_plus_xg": [
        "is_home", "team_r_reg_gf", "opp_r_reg_ga",
        "team_r_fivev5_xg", "opp_r_fivev5_xg_against",
        "team_r_pp_xg", "opp_r_pp_xg_against",
        "rest_days", "b2b", "games_prior_3_days", "games_prior_5_days",
    ],
    "fivev5_xg_pace": [
        "is_home", "team_r_fivev5_xg", "opp_r_fivev5_xg_against",
        "team_r_fivev5_xg_against", "opp_r_fivev5_xg",
        "team_r_fivev5_sog", "opp_r_fivev5_sog_against",
        "rest_days", "b2b", "games_prior_3_days", "games_prior_5_days",
    ],
    "process_without_recent_goals": [
        "is_home", "team_r_reg_ga", "opp_r_reg_gf",
        "team_r_fivev5_xg", "opp_r_fivev5_xg_against",
        "team_r_fivev5_xg_against", "opp_r_fivev5_xg",
        "team_r_fivev5_sog", "opp_r_fivev5_sog_against",
        "team_r_pp_xg", "opp_r_pp_xg_against",
        "opp_r_pp_goals_against", "team_r_penalties", "opp_r_penalties",
        "rest_days", "b2b", "games_prior_3_days", "games_prior_5_days",
        "oa_xg_attack_rel", "oa_xg_defense_rel",
        "oa_sog_attack_rel", "oa_sog_defense_rel",
    ],
    "hockey_process": [
        "is_home", "team_r_reg_gf", "opp_r_reg_ga",
        "team_r_reg_ga", "opp_r_reg_gf",
        "team_r_fivev5_xg", "opp_r_fivev5_xg_against",
        "team_r_fivev5_xg_against", "opp_r_fivev5_xg",
        "team_r_fivev5_sog", "opp_r_fivev5_sog_against",
        "team_r_pp_xg", "opp_r_pp_xg_against",
        "team_r_pp_goals", "opp_r_pp_goals_against",
        "team_r_penalties", "opp_r_penalties",
        "rest_days", "b2b", "games_prior_3_days", "games_prior_5_days",
        "oa_xg_attack_rel", "oa_xg_defense_rel",
        "oa_sog_attack_rel", "oa_sog_defense_rel",
    ],
}
FEATURE_SETS["hockey_process_plus_league_scoring"] = (
    FEATURE_SETS["hockey_process"] + ["league_r_reg_gf"]
)
CORE_FEATURE_SET = "hockey_process_plus_league_scoring"

ROLL_METRICS = [
    "reg_gf", "reg_ga", "xg", "xg_against", "fivev5_xg", "fivev5_xg_against",
    "sog", "sog_against", "fivev5_sog", "fivev5_sog_against",
    "pp_xg", "pp_xg_against", "pp_goals", "pp_goals_against", "penalties", "pim",
]


def _as_int_game_id(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("Int64")


def derive_game_targets(pbp: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    """Validate regulation labels from PBP final score + OT/SO status.

    In this feed shootout attempts are not counted as PBP goals, so a shootout
    game keeps its tied regulation score in the PBP scoreboard. A non-shootout
    overtime winner adds one goal to the PBP scoreboard; remove that OT goal to
    recover the regulation tie. Official final score for a shootout adds one
    goal for the shootout winner.
    """
    d = pbp.copy()
    d["game_id"] = _as_int_game_id(d.game_id)
    d = d.loc[d.game_type.astype("string").isin(["R", "P"])].copy()
    home = d.loc[d.home_away.eq("home"), [
        "game_id", "source_season", "game_type", "game_date", "home_abbr", "away_abbr",
        "final_home_score", "final_away_score", "went_ot", "went_shootout",
        "winner_home_away",
    ]].copy()
    away = d.loc[d.home_away.eq("away"), ["game_id"]].copy()
    if home.game_id.duplicated().any() or away.game_id.duplicated().any():
        raise ValueError("PBP source must have one home and one away feature row per game")
    if set(home.game_id.dropna()) != set(away.game_id.dropna()):
        raise ValueError("PBP home/away game coverage differs")

    home["game_id"] = _as_int_game_id(home.game_id)
    canonical = spine.copy()
    canonical["game_id"] = _as_int_game_id(canonical.game_id)
    canonical = canonical.loc[
        canonical.result_available.fillna(False)
        & canonical.game_type.astype("string").isin(["R", "P"]),
        ["game_id", "game_date", "source_season", "game_type", "home_team_abbr",
         "away_team_abbr", "home_win"],
    ].copy()
    if canonical.game_id.duplicated().any():
        raise ValueError("Canonical spine has duplicate game IDs")
    g = home.merge(canonical, on="game_id", how="inner", suffixes=("_pbp", "_spine"),
                   validate="one_to_one")
    g["source_season"] = pd.to_numeric(g.source_season_spine, errors="coerce").astype(int)
    g["game_type"] = g.game_type_spine.astype(str)
    g["game_date"] = pd.to_datetime(g.game_date_spine, errors="coerce").dt.normalize()
    g["home_team"] = g.home_team_abbr
    g["away_team"] = g.away_team_abbr
    if (g.home_team.astype(str) != g.home_abbr.astype(str)).any() or (g.away_team.astype(str) != g.away_abbr.astype(str)).any():
        raise ValueError("PBP team IDs/abbreviations disagree with the canonical game spine")
    for col in ["final_home_score", "final_away_score"]:
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g = g.dropna(subset=["final_home_score", "final_away_score", "went_ot", "went_shootout",
                         "winner_home_away", "home_win", "game_date"])
    g["final_home_score"] = g.final_home_score.astype(int)
    g["final_away_score"] = g.final_away_score.astype(int)
    g["went_ot"] = g.went_ot.astype(bool)
    g["went_shootout"] = g.went_shootout.astype(bool)
    g["home_win"] = g.home_win.astype(bool)
    g["winner_home_away"] = g.winner_home_away.astype(str)

    nonshootout_ot = g.went_ot & ~g.went_shootout
    g["reg_home_goals"] = g.final_home_score - (
        nonshootout_ot & g.winner_home_away.eq("home")
    ).astype(int)
    g["reg_away_goals"] = g.final_away_score - (
        nonshootout_ot & g.winner_home_away.eq("away")
    ).astype(int)
    g["official_final_home_goals"] = g.final_home_score + (
        g.went_shootout & g.winner_home_away.eq("home")
    ).astype(int)
    g["official_final_away_goals"] = g.final_away_score + (
        g.went_shootout & g.winner_home_away.eq("away")
    ).astype(int)
    g["reg_home_win"] = g.reg_home_goals.gt(g.reg_away_goals)
    g["reg_away_win"] = g.reg_home_goals.lt(g.reg_away_goals)
    g["reg_tie"] = g.reg_home_goals.eq(g.reg_away_goals)

    nonnegative = g.reg_home_goals.ge(0) & g.reg_away_goals.ge(0)
    reg_status_valid = g.reg_tie.eq(g.went_ot)
    final_winner_valid = g.official_final_home_goals.gt(g.official_final_away_goals).eq(g.home_win)
    g["label_valid"] = nonnegative & reg_status_valid & final_winner_valid
    scoped = g.loc[g.source_season.between(2022, 2026) & g.game_type.eq("R")]
    if not scoped.label_valid.all():
        raise ValueError("2022–2026 regular-season regulation score labels failed validation")
    if not (scoped.reg_tie == scoped.went_ot).all():
        raise ValueError("2022–2026 regular-season regulation tie labels disagree with went_ot")
    rejected = g.loc[~g.label_valid]
    excluded = int(len(rejected))
    excluded_by_season = {str(int(k)): int(v) for k, v in rejected.groupby("source_season").size().items()}
    g = g.loc[g.label_valid].copy()
    g.attrs["label_quality_excluded"] = excluded
    g.attrs["label_quality_excluded_by_season"] = excluded_by_season
    return g


def lagged_team_rolling(side: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Strictly prior-date rolling team features, excluding all games on date G."""
    # Collapse the rare same-team, same-date doubleheader into a single day
    # state, so no game on that date can enter another same-date prediction.
    daily = (
        side.groupby(["team", "game_date"], as_index=False, sort=False)[ROLL_METRICS]
        .mean()
        .sort_values(["team", "game_date"], kind="stable")
        .reset_index(drop=True)
    )
    for metric in ROLL_METRICS:
        daily[f"r_{metric}"] = daily.groupby("team", sort=False)[metric].transform(
            lambda s: s.shift(1).rolling(window, min_periods=ROLLING_MIN_PERIODS).mean()
        )
    return daily[["team", "game_date"] + [f"r_{m}" for m in ROLL_METRICS]]


def build_side_data(pbp: pd.DataFrame, spine: pd.DataFrame, labels: pd.DataFrame,
                    rolling_window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Build home/away team-score rows with strictly lagged hockey predictors."""
    d = pbp.copy()
    d["game_id"] = _as_int_game_id(d.game_id)
    d = d.loc[
        d.source_season.between(MIN_SOURCE_SEASON, 2026)
        & d.game_type.astype("string").isin(["R", "P"])
    ].copy()
    d["game_date"] = pd.to_datetime(d.game_date, errors="coerce").dt.normalize()
    d["source_season"] = pd.to_numeric(d.source_season, errors="coerce").astype(int)
    d["team"] = d.event_team_abbr.astype(str)
    d["opponent"] = d.opponent_abbr.astype(str)
    d["is_home"] = d.home_away.eq("home").astype(int)
    target_cols = ["game_id", "reg_home_goals", "reg_away_goals", "official_final_home_goals",
                   "official_final_away_goals", "went_ot", "went_shootout", "home_win",
                   "reg_tie"]
    d = d.merge(labels[target_cols], on="game_id", how="inner", validate="many_to_one")
    d["reg_gf"] = np.where(d.is_home.eq(1), d.reg_home_goals, d.reg_away_goals)
    d["reg_ga"] = np.where(d.is_home.eq(1), d.reg_away_goals, d.reg_home_goals)
    numeric_sources = {
        "xg": "xg", "xg_against": "xg_against",
        "fivev5_xg": "fivev5_xg", "fivev5_xg_against": "fivev5_xg_against",
        "sog": "sog", "sog_against": "sog_against",
        "fivev5_sog": "fivev5_sog", "fivev5_sog_against": "fivev5_sog_against",
        "pp_xg": "pp_xg", "pp_xg_against": "pp_xg_against",
        "pp_goals": "pp_goals", "pp_goals_against": "pp_goals_against",
        "penalties": "penalties", "pim": "pim",
    }
    for dst, src in numeric_sources.items():
        d[dst] = pd.to_numeric(d[src], errors="coerce")
    side = d[["game_id", "game_date", "source_season", "game_type", "team", "opponent",
              "is_home", "reg_gf", "reg_ga"] + list(numeric_sources)].copy()
    side = side.sort_values(["game_date", "game_id", "is_home"], kind="stable").reset_index(drop=True)

    roll = lagged_team_rolling(side, rolling_window)
    side = side.merge(roll, on=["team", "game_date"], how="left", validate="many_to_one")
    roll_cols = [c for c in roll.columns if c.startswith("r_")]
    opponent_roll = roll.rename(columns={"team": "opponent", **{c: f"opp_{c}" for c in roll_cols}})
    side = side.merge(opponent_roll, on=["opponent", "game_date"], how="left", validate="many_to_one")

    spine_small = spine.copy()
    spine_small["game_id"] = _as_int_game_id(spine_small.game_id)
    spine_small = spine_small.loc[
        spine_small.result_available.fillna(False)
        & spine_small.game_type.astype("string").isin(["R", "P"])
        & pd.to_numeric(spine_small.source_season, errors="coerce").between(MIN_SOURCE_SEASON, 2026)
    ].copy()
    spine_small["game_date"] = pd.to_datetime(spine_small.game_date, errors="coerce").dt.normalize()
    spine_small["source_season"] = pd.to_numeric(spine_small.source_season, errors="coerce").astype(int)
    schedule = sched.build_schedule_features(spine_small)
    side = side.merge(schedule, on="game_id", how="left", validate="many_to_one")
    side["rest_days"] = np.where(side.is_home.eq(1), side.h_sched_rest_days, side.a_sched_rest_days)
    side["b2b"] = np.where(side.is_home.eq(1), side.h_sched_b2b, side.a_sched_b2b)
    side["games_prior_3_days"] = np.where(
        side.is_home.eq(1), side.h_sched_prior3_games, side.a_sched_prior3_games
    )
    side["games_prior_5_days"] = np.where(
        side.is_home.eq(1), side.h_sched_prior5_games, side.a_sched_prior5_games
    )

    # Rolling NHL scoring environment, aggregated across all games on each
    # date and lagged by one full date. This is a pregame context feature and
    # cannot use a same-date result.
    league_daily = (
        side.groupby("game_date", as_index=False, sort=True).reg_gf.mean()
        .sort_values("game_date", kind="stable").reset_index(drop=True)
    )
    league_daily["league_r_reg_gf"] = league_daily.reg_gf.shift(1).rolling(
        LEAGUE_SCORING_WINDOW, min_periods=ROLLING_MIN_PERIODS
    ).mean()
    side = side.merge(
        league_daily[["game_date", "league_r_reg_gf"]],
        on="game_date", how="left", validate="many_to_one",
    )

    # Add opponent-adjusted hockey ratings, not win probabilities. All games on
    # the same calendar date receive the same pre-date rating state, avoiding
    # leakage between same-day games when exact historical puck-drop times are
    # unavailable.
    game_features = ts.game_event_table(pbp.loc[
        pd.to_numeric(pbp.source_season, errors="coerce").between(MIN_SOURCE_SEASON, 2026)
    ])
    if game_features.duplicated("game_id").any():
        raise ValueError("PBP event table has duplicate game IDs")
    oa = strict_date_opponent_adjusted_features(game_features, alpha=0.12, retain=0.70)
    oa["game_id"] = _as_int_game_id(oa.game_id)
    oa = oa.rename(columns={
        "oa_xg_off_diff_a0p12_r0p70": "oa_xg_off_diff",
        "oa_xg_def_adv_a0p12_r0p70": "oa_xg_def_adv",
        "oa_sog_off_diff_a0p12_r0p70": "oa_sog_off_diff",
        "oa_sog_def_adv_a0p12_r0p70": "oa_sog_def_adv",
    })
    side = side.merge(oa, on="game_id", how="left", validate="many_to_one")
    sign = np.where(side.is_home.eq(1), 1.0, -1.0)
    for src, dst in [
        ("oa_xg_off_diff", "oa_xg_attack_rel"),
        ("oa_xg_def_adv", "oa_xg_defense_rel"),
        ("oa_sog_off_diff", "oa_sog_attack_rel"),
        ("oa_sog_def_adv", "oa_sog_defense_rel"),
    ]:
        side[dst] = sign * pd.to_numeric(side[src], errors="coerce")

    rename = {}
    for c in roll_cols:
        metric = c.removeprefix("r_")
        rename[c] = f"team_{c}"
        rename[f"opp_{c}"] = f"opp_{c}"
    side = side.rename(columns=rename)
    side = side.replace([np.inf, -np.inf], np.nan)
    return side


def replace_team_history_window(base: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """Recompute only team/opponent rolling form while retaining shared inputs."""
    rolling_columns = [c for c in base.columns if c.startswith("team_r_") or c.startswith("opp_r_")]
    side = base.drop(columns=rolling_columns).copy()
    raw_columns = ["game_id", "game_date", "source_season", "game_type", "team",
                   "opponent", "is_home"] + ROLL_METRICS
    roll = lagged_team_rolling(side[raw_columns], rolling_window)
    side = side.merge(roll, on=["team", "game_date"], how="left", validate="many_to_one")
    roll_columns = [c for c in roll.columns if c.startswith("r_")]
    opponent_roll = roll.rename(columns={
        "team": "opponent", **{c: f"opp_{c}" for c in roll_columns}
    })
    side = side.merge(opponent_roll, on=["opponent", "game_date"], how="left",
                      validate="many_to_one")
    side = side.rename(columns={c: f"team_{c}" for c in roll_columns})
    return side.replace([np.inf, -np.inf], np.nan)


def strict_date_opponent_adjusted_features(games: pd.DataFrame, alpha: float,
                                           retain: float) -> pd.DataFrame:
    """Online offense/defense xG and shot ratings updated only after each date."""
    data = games.copy()
    data["game_date"] = pd.to_datetime(data.game_date, errors="coerce").dt.normalize()
    data = data.sort_values(["game_date", "game_id"], kind="stable")
    off_sog, def_sog, off_xg, def_xg = {}, {}, {}, {}
    league_sog = league_xg = None
    previous_season = None
    rows = []

    for game_date, day in data.groupby("game_date", sort=True):
        season = int(pd.to_numeric(day.source_season, errors="coerce").iloc[0])
        if previous_season is None:
            previous_season = season
        elif season != previous_season:
            off_sog = {t: v * retain for t, v in off_sog.items()}
            def_sog = {t: v * retain for t, v in def_sog.items()}
            off_xg = {t: v * retain for t, v in off_xg.items()}
            def_xg = {t: v * retain for t, v in def_xg.items()}
            previous_season = season

        # Every prediction for this date uses one frozen pre-date snapshot.
        daily_updates = []
        for r in day.itertuples(index=False):
            h, a = str(r.home_team), str(r.away_team)
            hos, aos = off_sog.get(h, 0.0), off_sog.get(a, 0.0)
            hds, ads = def_sog.get(h, 0.0), def_sog.get(a, 0.0)
            hox, aox = off_xg.get(h, 0.0), off_xg.get(a, 0.0)
            hdx, adx = def_xg.get(h, 0.0), def_xg.get(a, 0.0)
            if league_sog is None:
                league_sog = float(np.nanmean([r.h_sog, r.a_sog]))
            if league_xg is None:
                league_xg = float(np.nanmean([r.h_xg, r.a_xg]))

            pred_h_sog = max(0.1, league_sog + hos + ads)
            pred_a_sog = max(0.1, league_sog + aos + hds)
            pred_h_xg = max(0.01, league_xg + hox + adx)
            pred_a_xg = max(0.01, league_xg + aox + hdx)
            rows.append((
                r.game_id, pred_h_sog - pred_a_sog, pred_h_xg - pred_a_xg,
                hos - aos, ads - hds, hox - aox, adx - hdx,
            ))
            daily_updates.append((h, a, hos, aos, hds, ads, hox, aox, hdx, adx,
                                  r.h_sog, r.a_sog, r.h_xg, r.a_xg,
                                  pred_h_sog, pred_a_sog, pred_h_xg, pred_a_xg))

        # Apply the day's observations only after all pregame rows were emitted.
        step = alpha / 2.0
        d_off_sog, d_def_sog, d_off_xg, d_def_xg = {}, {}, {}, {}
        observed_sog, observed_xg = [], []
        for (h, a, hos, aos, hds, ads, hox, aox, hdx, adx,
             hs, aws, hx, ax, phs, pas, phx, pax) in daily_updates:
            if np.isfinite(hs) and np.isfinite(aws):
                rh, ra = hs - phs, aws - pas
                d_off_sog[h] = d_off_sog.get(h, 0.0) + step * rh
                d_def_sog[a] = d_def_sog.get(a, 0.0) + step * rh
                d_off_sog[a] = d_off_sog.get(a, 0.0) + step * ra
                d_def_sog[h] = d_def_sog.get(h, 0.0) + step * ra
                observed_sog.extend([hs, aws])
            if np.isfinite(hx) and np.isfinite(ax):
                rh, ra = hx - phx, ax - pax
                d_off_xg[h] = d_off_xg.get(h, 0.0) + step * rh
                d_def_xg[a] = d_def_xg.get(a, 0.0) + step * rh
                d_off_xg[a] = d_off_xg.get(a, 0.0) + step * ra
                d_def_xg[h] = d_def_xg.get(h, 0.0) + step * ra
                observed_xg.extend([hx, ax])
        for team, delta in d_off_sog.items(): off_sog[team] = off_sog.get(team, 0.0) + delta
        for team, delta in d_def_sog.items(): def_sog[team] = def_sog.get(team, 0.0) + delta
        for team, delta in d_off_xg.items(): off_xg[team] = off_xg.get(team, 0.0) + delta
        for team, delta in d_def_xg.items(): def_xg[team] = def_xg.get(team, 0.0) + delta
        if observed_sog:
            league_sog = (1 - alpha * 0.15) * league_sog + alpha * 0.15 * float(np.mean(observed_sog))
        if observed_xg:
            league_xg = (1 - alpha * 0.15) * league_xg + alpha * 0.15 * float(np.mean(observed_xg))

    cols = ["game_id", "oa_sog_diff_a0p12_r0p70", "oa_xg_diff_a0p12_r0p70",
            "oa_sog_off_diff_a0p12_r0p70", "oa_sog_def_adv_a0p12_r0p70",
            "oa_xg_off_diff_a0p12_r0p70", "oa_xg_def_adv_a0p12_r0p70"]
    out = pd.DataFrame(rows, columns=cols)
    if out.game_id.duplicated().any() or len(out) != len(data):
        raise ValueError("Opponent-adjusted pregame feature coverage is not one row per game")
    return out


def fit_count(train: pd.DataFrame, features: list[str], alpha: float) -> Pipeline:
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("poisson", PoissonRegressor(alpha=alpha, max_iter=2000, tol=1e-8)),
    ])
    model.fit(train[features], train.reg_gf.astype(float))
    return model


def count_log_loss(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-8, None)
    return float(np.mean(mu - y * np.log(mu) + gammaln(y + 1)))


def choose_alpha(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    best = None
    for alpha in POISSON_ALPHA_GRID:
        model = fit_count(train, features, alpha)
        mu = model.predict(val[features])
        score = count_log_loss(val.reg_gf.to_numpy(), mu)
        if best is None or score < best[0]:
            best = (score, alpha)
    return float(best[1]), float(best[0])


def poisson_vector(mu: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    return poisson.pmf(np.arange(max_goals + 1), mu)


def joint_score_pmf(mu_home: float, mu_away: float, family: str, param: float,
                    max_goals: int = MAX_GOALS) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    if family == "independent_poisson":
        matrix = np.outer(poisson_vector(mu_home, max_goals), poisson_vector(mu_away, max_goals))
    elif family == "independent_negative_binomial":
        alpha = float(param)
        shape = 1.0 / alpha
        ph = shape / (shape + mu_home)
        pa = shape / (shape + mu_away)
        matrix = np.outer(nbinom.pmf(goals, shape, ph), nbinom.pmf(goals, shape, pa))
    elif family == "bivariate_poisson":
        share = float(param)
        common = share * min(mu_home, mu_away)
        lam_home = max(mu_home - common, 1e-10)
        lam_away = max(mu_away - common, 1e-10)
        home_p = np.array([poisson.pmf(k, lam_home) if k >= 0 else 0.0 for k in goals])
        away_p = np.array([poisson.pmf(k, lam_away) if k >= 0 else 0.0 for k in goals])
        common_p = poisson.pmf(goals, common)
        matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
        for k, weight in enumerate(common_p):
            if weight == 0:
                continue
            hp = np.zeros(max_goals + 1)
            ap = np.zeros(max_goals + 1)
            hp[k:] = home_p[: max_goals + 1 - k]
            ap[k:] = away_p[: max_goals + 1 - k]
            matrix += weight * np.outer(hp, ap)
    elif family == "dixon_coles":
        matrix = np.outer(poisson_vector(mu_home, max_goals), poisson_vector(mu_away, max_goals))
        rho = float(param)
        tau = [
            [1 - mu_home * mu_away * rho, 1 + mu_home * rho],
            [1 + mu_away * rho, 1 - rho],
        ]
        low = matrix[:2, :2] * np.asarray(tau)
        if (low < 0).any():
            raise ValueError("Dixon-Coles low-score correction produced negative probability")
        matrix[:2, :2] = low
    elif family == "diagonal_inflated_poisson":
        matrix = np.outer(poisson_vector(mu_home, max_goals), poisson_vector(mu_away, max_goals))
        if float(param) <= 0:
            raise ValueError("Diagonal inflation multiplier must be positive")
        np.fill_diagonal(matrix, np.diag(matrix) * float(param))
    else:
        raise ValueError(f"Unknown score distribution {family}")
    total = float(matrix.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid score probability mass")
    matrix /= total
    return matrix


def outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    # Rows are home goals and columns are away goals.
    home = float(np.tril(matrix, -1).sum())
    away = float(np.triu(matrix, 1).sum())
    tie = float(np.trace(matrix))
    total = home + away + tie
    return home / total, away / total, tie / total


def final_score_pmf(reg_matrix: np.ndarray, p_home_ot: float) -> np.ndarray:
    n = reg_matrix.shape[0]
    final = np.zeros((n + 1, n + 1), dtype=float)
    final[:n, :n] = reg_matrix
    tie_mass = np.diag(reg_matrix).copy()
    for goals, mass in enumerate(tie_mass):
        final[goals, goals] -= mass
        final[goals + 1, goals] += mass * p_home_ot
        final[goals, goals + 1] += mass * (1 - p_home_ot)
    final /= final.sum()
    return final


def score_total_pmf(matrix: np.ndarray) -> np.ndarray:
    out = np.zeros(matrix.shape[0] + matrix.shape[1] - 1, dtype=float)
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            out[h + a] += matrix[h, a]
    out /= out.sum()
    return out


def fit_ot_home_rate(train_games: pd.DataFrame) -> float:
    tied = train_games.loc[train_games.went_ot.astype(bool)]
    # Weak, symmetric Beta(10, 10) prior keeps small historical samples near 50%.
    home_wins = int(tied.home_win.astype(bool).sum())
    return float((home_wins + 10) / (len(tied) + 20))


def validation_distribution_choice(val_games: pd.DataFrame, val_mu: pd.DataFrame) -> tuple[str, float, dict]:
    data = val_games.merge(val_mu, on="game_id", validate="one_to_one")
    scores = {}
    for family, params in DISTROS.items():
        best = None
        for param in params:
            losses = []
            for r in data.itertuples(index=False):
                mat = joint_score_pmf(r.mu_home, r.mu_away, family, param)
                p = mat[int(r.reg_home_goals), int(r.reg_away_goals)]
                losses.append(-np.log(max(p, 1e-15)))
            score = float(np.mean(losses))
            if best is None or score < best[0]:
                best = (score, float(param))
        scores[family] = {"validation_reg_score_logloss": best[0], "parameter": best[1]}
    family = min(scores, key=lambda k: scores[k]["validation_reg_score_logloss"])
    return family, float(scores[family]["parameter"]), scores


def _mean_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - observed
    design = np.column_stack([np.ones(len(predicted)), predicted])
    calibration = np.linalg.lstsq(design, observed, rcond=None)[0]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean_error_pred_minus_actual": float(np.mean(err)),
        "calibration_intercept": float(calibration[0]),
        "calibration_slope": float(calibration[1]),
    }


def binary_calibration(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    design = np.column_stack([np.ones(len(p)), logit(p)])
    if float(np.std(design[:, 1])) < 0.10:
        # A slope is not meaningfully identifiable when forecast spread is
        # nearly zero (as with these OT probabilities). Estimate only the
        # calibration-in-the-large offset with slope fixed at one.
        offset = design[:, 1]
        intercept = 0.0
        for _ in range(100):
            q = np.clip(expit(intercept + offset), 1e-10, 1 - 1e-10)
            grad = float(np.sum(y - q))
            hess = float(np.sum(q * (1 - q)))
            step = grad / max(hess, 1e-10)
            intercept += float(np.clip(step, -5, 5))
            if abs(step) < 1e-9:
                break
        return {"intercept": float(intercept), "slope": None,
                "fit_success": True, "slope_identifiable": False}
    coef = np.array([0.0, 1.0])
    converged = False
    for _ in range(100):
        q = np.clip(expit(design @ coef), 1e-10, 1 - 1e-10)
        w = np.maximum(q * (1 - q), 1e-10)
        hessian = design.T @ (w[:, None] * design)
        gradient = design.T @ (y - q)
        step = np.linalg.solve(hessian, gradient)
        # Damped Newton steps avoid overshooting on highly unbalanced bins.
        scale = min(1.0, 5.0 / max(float(np.max(np.abs(step))), 5.0))
        coef += scale * step
        if np.max(np.abs(scale * step)) < 1e-9:
            converged = True
            break
    return {"intercept": float(coef[0]), "slope": float(coef[1]),
            "fit_success": converged, "slope_identifiable": True}


def metrics_for_oof(rows: pd.DataFrame) -> dict:
    ycls = np.where(rows.reg_home_goals.gt(rows.reg_away_goals), 2,
                    np.where(rows.reg_home_goals.lt(rows.reg_away_goals), 0, 1))
    p3 = rows[["p_reg_away_win", "p_reg_tie", "p_reg_home_win"]].to_numpy(dtype=float)
    p_true = np.clip(p3[np.arange(len(rows)), ycls], 1e-15, 1)
    onehot = np.eye(3)[ycls]
    non_tie = rows.reg_home_goals.ne(rows.reg_away_goals)
    y_home_non_tie = rows.loc[non_tie, "reg_home_goals"].gt(rows.loc[non_tie, "reg_away_goals"]).astype(int)
    p_home_non_tie = rows.loc[non_tie, "p_reg_home_conditional"].to_numpy(dtype=float)
    p_ot = np.clip(rows.p_overtime.to_numpy(dtype=float), 1e-15, 1 - 1e-15)
    y_ot = rows.went_ot.astype(int).to_numpy()
    result = {
        "games": int(len(rows)),
        "reg_home_goals": _mean_metrics(rows.reg_home_goals, rows.exp_reg_home_goals),
        "reg_away_goals": _mean_metrics(rows.reg_away_goals, rows.exp_reg_away_goals),
        "final_home_goals": _mean_metrics(rows.official_final_home_goals, rows.exp_final_home_goals),
        "final_away_goals": _mean_metrics(rows.official_final_away_goals, rows.exp_final_away_goals),
        "reg_total_goals": _mean_metrics(
            rows.reg_home_goals + rows.reg_away_goals,
            rows.exp_reg_home_goals + rows.exp_reg_away_goals,
        ),
        "final_total_goals": _mean_metrics(
            rows.official_final_home_goals + rows.official_final_away_goals,
            rows.exp_final_home_goals + rows.exp_final_away_goals,
        ),
        "reg_score_logloss": float(-rows.logp_reg_score.mean()),
        "final_score_logloss": float(-rows.logp_final_score.mean()),
        "final_total_logloss": float(-rows.logp_final_total.mean()),
        "reg_outcome_multiclass_logloss": float(-np.mean(np.log(p_true))),
        "reg_outcome_multiclass_brier": float(np.mean(np.sum((p3 - onehot) ** 2, axis=1))),
        "reg_outcome_accuracy": float(accuracy_score(ycls, np.argmax(p3, axis=1))),
        "reg_home_win_calibration": {
            "games_non_tied": int(non_tie.sum()),
            "brier": float(np.mean((p_home_non_tie - y_home_non_tie.to_numpy()) ** 2)),
            "log_loss": float(np.mean(-(y_home_non_tie * np.log(np.clip(p_home_non_tie,1e-15,1))
                                        +(1-y_home_non_tie)*np.log(np.clip(1-p_home_non_tie,1e-15,1))))),
            "calibration": binary_calibration(y_home_non_tie.to_numpy(), p_home_non_tie),
        },
        "overtime_brier": float(np.mean((p_ot - y_ot) ** 2)),
        "overtime_logloss": float(np.mean(-(y_ot * np.log(p_ot) + (1-y_ot) * np.log(1-p_ot)))),
        "overtime_predicted_rate": float(np.mean(p_ot)),
        "overtime_observed_rate": float(np.mean(y_ot)),
        "overtime_calibration": binary_calibration(y_ot, p_ot),
        "regulation_tie_calibration": binary_calibration(y_ot, p_ot),
    }
    # Split regulation-tie mass with a training-only OT winner prior to produce
    # full-game probabilities; this keeps regulation and final outcomes distinct.
    p_full_home = rows.p_reg_home_win.to_numpy(dtype=float) + (
        p_ot * rows.p_home_ot_winner.to_numpy(dtype=float)
    )
    y_full_home = rows.official_final_home_goals.gt(rows.official_final_away_goals).astype(int).to_numpy()
    result["full_game_home_win"] = {
        "brier": float(np.mean((p_full_home - y_full_home) ** 2)),
        "log_loss": float(np.mean(-(y_full_home * np.log(np.clip(p_full_home, 1e-15, 1))
                                    + (1-y_full_home) * np.log(np.clip(1-p_full_home, 1e-15, 1))))),
        "accuracy": float(accuracy_score(y_full_home, p_full_home >= 0.5)),
        "calibration": binary_calibration(y_full_home, p_full_home),
    }
    for line in [4.5, 5.5, 6.5]:
        name = str(line).replace(".", "_")
        p_over = rows[f"p_final_over_{name}"].to_numpy(dtype=float)
        y_over = (rows.official_final_home_goals + rows.official_final_away_goals > line).astype(int).to_numpy()
        result[f"final_total_over_{line}"] = {
            "brier": float(np.mean((p_over-y_over)**2)),
            "observed_rate": float(np.mean(y_over)),
            "predicted_rate": float(np.mean(p_over)),
        }
    return result


def make_prediction_rows(games: pd.DataFrame, mu: pd.DataFrame, family: str, param: float,
                         p_home_ot: float, model_name: str, target: int) -> list[dict]:
    data = games.merge(mu, on="game_id", validate="one_to_one")
    output = []
    for r in data.itertuples(index=False):
        mat = joint_score_pmf(r.mu_home, r.mu_away, family, param)
        p_home_reg, p_away_reg, p_tie = outcome_probs(mat)
        fmat = final_score_pmf(mat, p_home_ot)
        tpmf = score_total_pmf(fmat)
        exp_reg_h = float(sum(i * mat[i, :].sum() for i in range(mat.shape[0])))
        exp_reg_a = float(sum(j * mat[:, j].sum() for j in range(mat.shape[1])))
        exp_final_h = float(sum(i * fmat[i, :].sum() for i in range(fmat.shape[0])))
        exp_final_a = float(sum(j * fmat[:, j].sum() for j in range(fmat.shape[1])))
        reg_h, reg_a = int(r.reg_home_goals), int(r.reg_away_goals)
        full_h, full_a = int(r.official_final_home_goals), int(r.official_final_away_goals)
        total_actual = full_h + full_a
        p_final_total = tpmf[total_actual] if total_actual < len(tpmf) else 1e-15
        output.append({
            "game_id": int(r.game_id), "game_date": pd.Timestamp(r.game_date).date().isoformat(),
            "source_season": int(r.source_season), "game_type": r.game_type,
            "home_team": r.home_team, "away_team": r.away_team,
            "reg_home_goals": reg_h, "reg_away_goals": reg_a,
            "official_final_home_goals": full_h, "official_final_away_goals": full_a,
            "went_ot": int(bool(r.went_ot)), "went_shootout": int(bool(r.went_shootout)),
            "model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION,
            "model_variant": model_name, "distribution_family": family,
            "distribution_parameter": float(param), "p_home_ot_winner": float(p_home_ot),
            "rate_mu_home": float(r.mu_home), "rate_mu_away": float(r.mu_away),
            "exp_reg_home_goals": exp_reg_h, "exp_reg_away_goals": exp_reg_a,
            "exp_final_home_goals": exp_final_h, "exp_final_away_goals": exp_final_a,
            "p_reg_home_win": p_home_reg, "p_reg_away_win": p_away_reg,
            "p_reg_tie": p_tie, "p_overtime": p_tie,
            "p_reg_home_conditional": p_home_reg / max(1-p_tie, 1e-12),
            "p_full_home_win": p_home_reg + p_tie * p_home_ot,
            "p_full_away_win": p_away_reg + p_tie * (1-p_home_ot),
            "logp_reg_score": float(np.log(max(mat[reg_h, reg_a], 1e-15))),
            "logp_final_score": float(np.log(max(fmat[full_h, full_a], 1e-15))),
            "logp_final_total": float(np.log(max(p_final_total, 1e-15))),
            "p_final_over_4_5": float(sum(tpmf[int(np.floor(4.5))+1:])),
            "p_final_over_5_5": float(sum(tpmf[int(np.floor(5.5))+1:])),
            "p_final_over_6_5": float(sum(tpmf[int(np.floor(6.5))+1:])),
            "pregame_data_cutoff_date": pd.Timestamp(r.game_date).date().isoformat(),
            "data_quality_flag": "verified_regulation_target;strict_prior_date_features;no_goalie_or_lineup_snapshot",
        })
    return output


def make_score_distribution_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Retain full regulation and final-score PMFs for selected OOF forecasts."""
    rows = []
    for r in predictions.itertuples(index=False):
        reg = joint_score_pmf(
            r.rate_mu_home, r.rate_mu_away, r.distribution_family,
            r.distribution_parameter,
        )
        final = final_score_pmf(reg, r.p_home_ot_winner)
        if not np.isclose(reg.sum(), 1.0, atol=1e-12) or not np.isclose(final.sum(), 1.0, atol=1e-12):
            raise ValueError(f"Score PMF does not sum to one for game {r.game_id}")
        rows.append({
            "game_id": int(r.game_id), "game_date": r.game_date,
            "source_season": int(r.source_season), "home_team": r.home_team,
            "away_team": r.away_team, "model_version": r.model_version,
            "model_variant": r.model_variant,
            "distribution_family": r.distribution_family,
            "distribution_parameter": float(r.distribution_parameter),
            "p_home_ot_winner": float(r.p_home_ot_winner),
            "regulation_max_goals_per_team": MAX_GOALS,
            "final_max_goals_per_team": MAX_GOALS + 1,
            "matrix_order": "row_major; rows=home_goals, columns=away_goals",
            "regulation_score_pmf": reg.ravel().tolist(),
            "final_score_pmf": final.ravel().tolist(),
        })
    return pd.DataFrame(rows)


def write_score_distribution_table(predictions: pd.DataFrame, path: Path) -> None:
    """Write compact float32 PMFs; forecast summaries remain float64 in the ledger."""
    table = pa.Table.from_pandas(make_score_distribution_table(predictions), preserve_index=False)
    for name in ["regulation_score_pmf", "final_score_pmf"]:
        index = table.schema.get_field_index(name)
        table = table.set_column(index, name, table.column(name).cast(pa.list_(pa.float32())))
    pq.write_table(table, path, compression="zstd", compression_level=9)


def main(pbp_path: str, spine_path: str, outdir: str) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pbp = pd.read_parquet(pbp_path)
    spine = pd.read_parquet(spine_path)
    labels = derive_game_targets(pbp, spine)
    label_quality_excluded = int(labels.attrs.get("label_quality_excluded", 0))
    label_quality_excluded_by_season = labels.attrs.get("label_quality_excluded_by_season", {})
    labels = labels.loc[labels.source_season.between(MIN_SOURCE_SEASON, 2026)].copy()
    regular_games = labels.loc[
        labels.game_type.eq("R") & labels.source_season.between(MIN_SOURCE_SEASON, 2026)
    ].copy()

    # Build shared schedule and opponent-adjusted inputs once. Recompute only
    # team/opponent rolling histories for candidate windows, holding the league
    # scoring context at 20 dates in every candidate.
    base_side = build_side_data(pbp, spine, labels, rolling_window=ROLLING_WINDOWS[0])
    side_by_window = {
        window: (base_side if window == ROLLING_WINDOWS[0]
                 else replace_team_history_window(base_side, window))
        for window in ROLLING_WINDOWS
    }
    regular_side = {
        window: frame.loc[frame.game_type.eq("R")].copy()
        for window, frame in side_by_window.items()
    }

    all_rows, fold_rows, window_choices = [], [], {}
    for target in TARGET_SEASONS:
        val_season = target - 1
        test_games = regular_games.loc[regular_games.source_season.eq(target)].copy()
        train_games = regular_games.loc[regular_games.source_season.le(target - 2)].copy()
        val_games = regular_games.loc[regular_games.source_season.eq(val_season)].copy()
        if len(test_games) < 1000 or len(val_games) < 500:
            raise RuntimeError(f"Unexpected regular-season target/validation coverage at {target}")

        window_state = {}
        for window in ROLLING_WINDOWS:
            frame = regular_side[window]
            train = frame.loc[frame.source_season.le(target - 2)].copy()
            val = frame.loc[frame.source_season.eq(val_season)].copy()
            test = frame.loc[frame.source_season.eq(target)].copy()
            features = FEATURE_SETS[CORE_FEATURE_SET]
            alpha, val_count_ll = choose_alpha(train, val, features)
            validation_model = fit_count(train, features, alpha)
            val_rows = val[["game_id", "is_home"]].copy()
            val_rows["mu"] = np.clip(validation_model.predict(val[features]), 1e-4, 20)
            val_mu = val_rows.pivot(index="game_id", columns="is_home", values="mu").reset_index()
            val_mu = val_mu.rename(columns={0: "mu_away", 1: "mu_home"})
            val_joined = val_games.merge(val_mu, on="game_id", validate="one_to_one")
            val_score_ll = float(np.mean([
                -np.log(max(joint_score_pmf(r.mu_home, r.mu_away, "independent_poisson", 0.0)[
                    int(r.reg_home_goals), int(r.reg_away_goals)], 1e-15))
                for r in val_joined.itertuples(index=False)
            ]))
            full_model = fit_count(pd.concat([train, val], ignore_index=True), features, alpha)
            test_rows = test[["game_id", "is_home"]].copy()
            test_rows["mu"] = np.clip(full_model.predict(test[features]), 1e-4, 20)
            test_mu = test_rows.pivot(index="game_id", columns="is_home", values="mu").reset_index()
            test_mu = test_mu.rename(columns={0: "mu_away", 1: "mu_home"})
            window_state[window] = {
                "alpha": alpha, "validation_count_ll": val_count_ll,
                "validation_score_ll": val_score_ll, "val_mu": val_mu,
                "test_mu": test_mu, "train": train, "val": val, "test": test,
            }

        # Choose the history window by prior-season independent-Poisson score
        # likelihood. No target-season result participates in selection.
        chosen_window = min(ROLLING_WINDOWS, key=lambda w: window_state[w]["validation_score_ll"])
        chosen = window_state[chosen_window]
        family, parameter, distribution_scores = validation_distribution_choice(
            val_games, chosen["val_mu"]
        )
        window_choices[str(target)] = {
            "validation_season": val_season,
            "selected_team_history_dates": chosen_window,
            "selection_metric": "independent-Poisson regulation score log loss on S-1",
            "by_window": {
                str(w): {
                    "poisson_alpha": window_state[w]["alpha"],
                    "validation_count_logloss": window_state[w]["validation_count_ll"],
                    "validation_reg_score_logloss": window_state[w]["validation_score_ll"],
                }
                for w in ROLLING_WINDOWS
            },
            "selected_distribution_family": family,
            "selected_distribution_parameter": parameter,
            "selected_window_distribution_scores": distribution_scores,
        }

        p_ot_train = fit_ot_home_rate(train_games)
        p_ot_final = fit_ot_home_rate(pd.concat([train_games, val_games], ignore_index=True))
        variant_predictions = {}
        for window in ROLLING_WINDOWS:
            state = window_state[window]
            name = f"team_window_{window}_independent_poisson"
            variant_predictions[name] = make_prediction_rows(
                test_games, state["test_mu"], "independent_poisson", 0.0,
                p_ot_final, name, target,
            )
            fold_rows.append({
                "target_season": target, "validation_season": val_season,
                "train_through_season": target - 2, "team_history_window": window,
                "selected_for_target": window == chosen_window, "model_variant": name,
                "selected_poisson_alpha": state["alpha"],
                "validation_count_logloss": state["validation_count_ll"],
                "validation_reg_score_logloss": state["validation_score_ll"],
                "distribution_family": "independent_poisson", "distribution_parameter": 0.0,
            })
        selected_name = "nested_window_selected_distribution"
        variant_predictions[selected_name] = make_prediction_rows(
            test_games, chosen["test_mu"], family, parameter, p_ot_final, selected_name, target,
        )
        fold_rows.append({
            "target_season": target, "validation_season": val_season,
            "train_through_season": target - 2, "team_history_window": chosen_window,
            "selected_for_target": True, "model_variant": selected_name,
            "selected_poisson_alpha": chosen["alpha"],
            "validation_count_logloss": chosen["validation_count_ll"],
            "validation_reg_score_logloss": chosen["validation_score_ll"],
            "distribution_family": family, "distribution_parameter": parameter,
        })
        for rows in variant_predictions.values():
            all_rows.extend(rows)

    predictions = pd.DataFrame(all_rows)
    if predictions.duplicated(["game_id", "model_variant"]).any():
        raise ValueError("Duplicate out-of-fold goal predictions")
    if not np.allclose(
        predictions.p_reg_home_win + predictions.p_reg_away_win + predictions.p_reg_tie,
        1.0, atol=1e-8,
    ):
        raise ValueError("Regulation outcome probabilities do not sum to one")
    if not np.allclose(predictions.p_reg_tie, predictions.p_overtime, atol=1e-12):
        raise ValueError("Overtime probability must equal regulation-tie probability")
    if not np.allclose(predictions.p_full_home_win + predictions.p_full_away_win, 1.0, atol=1e-10):
        raise ValueError("Full-game home/away win probabilities do not sum to one")
    predictions.to_parquet(out / "oof_predictions.parquet", index=False)
    selected_oof = predictions.loc[
        predictions.model_variant.eq("nested_window_selected_distribution")
    ].copy()
    write_score_distribution_table(selected_oof, out / "score_distribution_pmf.parquet")
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out / "fold_metrics.csv", index=False)

    summary = {
        "model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION,
        "source_version": SOURCE_VERSION, "retrieval_date": RETRIEVAL_DATE,
        "validation_design": "nested rolling origin; choose team window and score law on S-1; fit through S-1; score S",
        "target_seasons": TARGET_SEASONS,
        "target_season_games": {str(s): int(regular_games.source_season.eq(s).sum()) for s in TARGET_SEASONS},
        "regulation_target_validation": {
            "regulation_tie_matches_went_ot_for_2022_2026_regular": True,
            "canonical_winner_crosscheck": "pass",
            "shootout_handling": "PBP regulation tie retained; add one official final goal to shootout winner",
            "historical_rows_excluded_for_label_quality": label_quality_excluded,
            "rows_excluded_by_source_season": label_quality_excluded_by_season,
        },
        "team_history_windows_compared": ROLLING_WINDOWS,
        "league_scoring_window_dates_fixed": LEAGUE_SCORING_WINDOW,
        "window_selection": window_choices,
        "features": {CORE_FEATURE_SET: FEATURE_SETS[CORE_FEATURE_SET]},
        "score_distribution_export": {
            "file": "score_distribution_pmf.parquet",
            "rows": "one selected nested-model prediction per target game",
            "regulation_grid": f"{MAX_GOALS + 1}x{MAX_GOALS + 1}; row-major home goals by away goals",
            "final_grid": f"{MAX_GOALS + 2}x{MAX_GOALS + 2}; includes training-only overtime winner split",
        },
        "goalie_lineup_status": "excluded: no timestamped pregame starter/lineup snapshot in current historical warehouse",
        "win_probability_or_market_inputs": "none",
        "overtime_winner_method": "empirical home OT win rate from training games with symmetric Beta(10,10) prior",
        "models": {},
    }
    for name, group in predictions.groupby("model_variant", sort=True):
        summary["models"][name] = {
            "pooled": metrics_for_oof(group),
            "season_metrics": {
                str(int(season)): metrics_for_oof(g)
                for season, g in group.groupby("source_season", sort=True)
            },
        }

    # Context check for the selected model. Schedule state is independent of
    # the candidate rolling window and comes from the canonical game spine.
    core_oof = predictions.loc[predictions.model_variant.eq("nested_window_selected_distribution")].copy()
    context_rows = []
    state = regular_side[20].groupby(["source_season", "game_id"]).agg(
        home_b2b=("h_sched_b2b", "first"), away_b2b=("a_sched_b2b", "first")
    ).reset_index()
    for season in TARGET_SEASONS:
        z = core_oof.loc[core_oof.source_season.eq(season)].copy()
        season_state = state.loc[state.source_season.eq(season)]
        for hb, ab, label in [(0, 0, "neither"), (1, 0, "home_only"), (0, 1, "away_only"), (1, 1, "both")]:
            ids = set(season_state.loc[season_state.home_b2b.eq(hb) & season_state.away_b2b.eq(ab), "game_id"])
            g = z.loc[z.game_id.isin(ids)]
            if not g.empty:
                m = metrics_for_oof(g)
                context_rows.append({
                    "target_season": season, "context": "B2B state", "level": label,
                    "games": m["games"], "reg_home_mae": m["reg_home_goals"]["mae"],
                    "reg_away_mae": m["reg_away_goals"]["mae"],
                    "reg_score_logloss": m["reg_score_logloss"],
                    "overtime_brier": m["overtime_brier"],
                    "reg_outcome_brier": m["reg_outcome_multiclass_brier"],
                })
    pd.DataFrame(context_rows).to_parquet(out / "context_metrics.parquet", index=False)

    reliability = []
    for name, group in predictions.groupby("model_variant", sort=True):
        for key, outcome in [("p_reg_home_conditional", "home_reg_win"),
                             ("p_full_home_win", "home_win"), ("p_reg_tie", "went_ot"),
                             ("p_final_over_5_5", None)]:
            z = group.copy()
            z["bin"] = pd.qcut(z[key], q=10, labels=False, duplicates="drop")
            for bin_no, b in z.groupby("bin", dropna=True):
                if outcome == "home_reg_win":
                    b = b.loc[b.reg_home_goals.ne(b.reg_away_goals)]
                    if b.empty:
                        continue
                    observed = (b.reg_home_goals > b.reg_away_goals).mean()
                elif outcome == "home_win":
                    observed = (b.official_final_home_goals > b.official_final_away_goals).mean()
                elif outcome == "went_ot":
                    observed = b.went_ot.mean()
                else:
                    observed = (b.official_final_home_goals + b.official_final_away_goals > 5.5).mean()
                reliability.append({
                    "model_variant": name, "probability": key, "bin": int(bin_no),
                    "games": int(len(b)), "predicted": float(b[key].mean()),
                    "observed": float(observed), "min_probability": float(b[key].min()),
                    "max_probability": float(b[key].max()),
                })
    pd.DataFrame(reliability).to_parquet(out / "reliability_bins.parquet", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbp", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.pbp, args.spine, args.out)
