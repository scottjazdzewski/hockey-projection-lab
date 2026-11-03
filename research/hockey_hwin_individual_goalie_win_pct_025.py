"""Test each named goalie's individual prior win record on HWIN probabilities.

The goalie history follows the individual goalie ID across all teams. Its
feature is that goalie's prior team wins minus the frozen Hockey Lab Game
champion's expected wins, shrunk by prior starts. A home-minus-away difference
adjusts only the frozen win-probability logit; team goalie mix and goal rates
are not model inputs.

The target starter IDs come from postgame goalie boxes, so this measures a
retrospective upper-bound effect until timestamped pregame starter snapshots
are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit


EXPERIMENT_ID = "EXP-0040"
MODEL_ID = "HOCKEY-LAB-HWIN-INDIVIDUAL-GOALIE-WIN-PCT-025-RESEARCH"
BASE_VARIANT = "team_strength_004_plus_b2b"
TARGET_SEASONS = [2024, 2025, 2026]
WINDOWS = [None, 5, 10, 20]
PRIORS = [5, 15, 30, 60]
PENALTIES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 4007


def normalise_ids(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_games(base_path: str, starter_path: str) -> pd.DataFrame:
    base = pd.read_parquet(base_path)
    base = base.loc[
        base.model_variant.eq(BASE_VARIANT)
        & base.game_type.eq("R")
        & base.source_season.between(2022, 2026),
        ["game_id", "game_date", "source_season", "home_team", "away_team", "home_win", "p_home"],
    ].copy()
    base["game_id"] = normalise_ids(base.game_id)
    base["game_date"] = pd.to_datetime(base.game_date, errors="coerce").dt.normalize()
    base["source_season"] = base.source_season.astype(int)
    base["home_win"] = base.home_win.astype(int)
    base["p_base"] = base.p_home.astype(float).clip(1e-8, 1.0-1e-8)
    if base.game_id.duplicated().any() or base.game_date.isna().any():
        raise ValueError("Frozen HWIN ledger has duplicate IDs or invalid dates")
    if len(base) != 5 * 1312:
        raise ValueError(f"Expected 6,560 regular-season HWIN games; found {len(base)}")

    starters = pd.read_parquet(starter_path)
    starters["game_id"] = normalise_ids(starters.game_id)
    starters = starters.loc[
        starters.starter_bool.fillna(False).astype(bool)
        & starters.source_season.between(2022, 2026)
        & starters.game_id.isin(base.game_id),
        ["game_id", "source_season", "goalie_id_key", "home_away", "team_abbrev", "player_name"],
    ].copy()
    starters["source_season"] = starters.source_season.astype(int)
    starters["goalie_id_key"] = starters.goalie_id_key.astype("string")
    if starters.duplicated(["game_id", "home_away"]).any():
        raise ValueError("Multiple target starters assigned to a game side")
    if starters.groupby("game_id").size().ne(2).any():
        raise ValueError("Every target game must have both actual starter IDs")
    home = starters.loc[starters.home_away.eq("home"), ["game_id", "goalie_id_key", "team_abbrev", "player_name"]].rename(columns={
        "goalie_id_key": "home_goalie_id", "team_abbrev": "home_goalie_team", "player_name": "home_goalie_name",
    })
    away = starters.loc[starters.home_away.eq("away"), ["game_id", "goalie_id_key", "team_abbrev", "player_name"]].rename(columns={
        "goalie_id_key": "away_goalie_id", "team_abbrev": "away_goalie_team", "player_name": "away_goalie_name",
    })
    games = base.merge(home, on="game_id", how="inner", validate="one_to_one")
    games = games.merge(away, on="game_id", how="inner", validate="one_to_one")
    if len(games) != len(base):
        raise ValueError(f"Actual starter coverage {len(games)}/{len(base)}")
    if not games.home_team.astype(str).eq(games.home_goalie_team.astype(str)).all():
        raise ValueError("Home starter team does not match game side")
    if not games.away_team.astype(str).eq(games.away_goalie_team.astype(str)).all():
        raise ValueError("Away starter team does not match game side")
    if games.duplicated(["home_team", "game_date"]).any() or games.duplicated(["away_team", "game_date"]).any():
        raise ValueError("Multiple same-date games for a team prevent date-lag validation")
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def add_individual_goalie_history(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    home = pd.DataFrame({
        "game_id": games.game_id, "game_date": games.game_date, "source_season": games.source_season,
        "goalie_id": games.home_goalie_id.astype("string"), "goalie_name": games.home_goalie_name.astype("string"),
        "team": games.home_team.astype("string"), "side": "home",
        "team_win": games.home_win.to_numpy(float), "expected_win": games.p_base.to_numpy(float),
    })
    away = pd.DataFrame({
        "game_id": games.game_id, "game_date": games.game_date, "source_season": games.source_season,
        "goalie_id": games.away_goalie_id.astype("string"), "goalie_name": games.away_goalie_name.astype("string"),
        "team": games.away_team.astype("string"), "side": "away",
        "team_win": 1.0-games.home_win.to_numpy(float), "expected_win": 1.0-games.p_base.to_numpy(float),
    })
    long = pd.concat([home, away], ignore_index=True).sort_values(
        ["game_date", "game_id", "side"], kind="stable"
    ).reset_index(drop=True)
    if long.duplicated(["goalie_id", "game_date"]).any():
        raise ValueError("A goalie has multiple team starts on one date")
    long["win_residual"] = long.team_win - long.expected_win
    parts = []
    for _, group in long.groupby("goalie_id", sort=False):
        x = group.sort_values(["game_date", "game_id"], kind="stable").copy()
        x["prior_goalie_starts"] = np.arange(len(x), dtype=float)
        x["prior_goalie_wins"] = x.team_win.cumsum().shift(1).fillna(0.0)
        x["prior_goalie_expected_wins"] = x.expected_win.cumsum().shift(1).fillna(0.0)
        x["prior_goalie_win_residual"] = x.win_residual.cumsum().shift(1).fillna(0.0)
        x["raw_prior_goalie_win_pct"] = np.divide(
            x.prior_goalie_wins, x.prior_goalie_starts,
            out=np.full(len(x), np.nan), where=x.prior_goalie_starts.to_numpy() > 0,
        )
        x["prior_expected_goalie_win_pct"] = np.divide(
            x.prior_goalie_expected_wins, x.prior_goalie_starts,
            out=np.full(len(x), np.nan), where=x.prior_goalie_starts.to_numpy() > 0,
        )
        for window in (5, 10, 20):
            x[f"prior_wins_r{window}"] = x.team_win.rolling(window, min_periods=1).sum().shift(1).fillna(0.0)
            x[f"prior_expected_wins_r{window}"] = x.expected_win.rolling(window, min_periods=1).sum().shift(1).fillna(0.0)
            x[f"prior_residual_r{window}"] = x.win_residual.rolling(window, min_periods=1).sum().shift(1).fillna(0.0)
        for prior in PRIORS:
            x[f"individual_goalie_win_residual_career_k{prior}"] = (
                x.prior_goalie_win_residual / (x.prior_goalie_starts + prior)
            )
            for window in (5, 10, 20):
                n = np.minimum(x.prior_goalie_starts, window)
                x[f"individual_goalie_win_residual_r{window}_k{prior}"] = (
                    x[f"prior_residual_r{window}"] / (n + prior)
                )
        parts.append(x)
    history = pd.concat(parts, ignore_index=True)
    features = [c for c in history if c.startswith("individual_goalie_win_residual_")]
    diagnostics = ["prior_goalie_starts", "raw_prior_goalie_win_pct", "prior_expected_goalie_win_pct"]
    wide = history.pivot(index="game_id", columns="side", values=features + diagnostics)
    wide.columns = [f"{side}_{name}" for name, side in wide.columns]
    wide = wide.reset_index()
    full = games.merge(wide, on="game_id", how="left", validate="one_to_one")
    all_history_cols = [c for c in full if c.startswith(("home_individual_goalie_win_residual_", "away_individual_goalie_win_residual_"))]
    full[all_history_cols] = full[all_history_cols].fillna(0.0)
    for c in ["home_prior_goalie_starts", "away_prior_goalie_starts"]:
        full[c] = full[c].fillna(0.0)
    return full, history


def feature_for(window: int | None, prior: int) -> str:
    return f"individual_goalie_win_residual_{'career' if window is None else f'r{window}'}_k{prior}"


def fit_beta(frame: pd.DataFrame, feature: str, penalty: float) -> float:
    x = frame[f"home_{feature}"].to_numpy(float) - frame[f"away_{feature}"].to_numpy(float)
    y = frame.home_win.to_numpy(float)
    offset = logit(frame.p_base.to_numpy(float))

    def objective(beta: np.ndarray) -> float:
        eta = np.clip(offset + beta[0]*x, -25.0, 25.0)
        return float(np.mean(np.logaddexp(0.0, eta)-y*eta) + 0.5*penalty*beta[0]**2)

    result = minimize(objective, np.zeros(1), method="L-BFGS-B", bounds=[(0.0, 20.0)],
                      options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000})
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Individual goalie residual fit failed: {result.message}")
    return float(result.x[0])


def predict(frame: pd.DataFrame, feature: str | None, beta: float) -> np.ndarray:
    eta = logit(frame.p_base.to_numpy(float))
    if feature is not None:
        eta += beta * (frame[f"home_{feature}"].to_numpy(float) - frame[f"away_{feature}"].to_numpy(float))
    p = expit(np.clip(eta, -25.0, 25.0))
    if not np.allclose(p + (1-p), 1.0, atol=1e-12):
        raise ValueError("Home/away probabilities fail sum-to-one check")
    return p


def calibration(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    z = logit(np.clip(np.asarray(p, dtype=float), 1e-8, 1-1e-8))
    def objective(theta: np.ndarray) -> float:
        eta = np.clip(theta[0] + theta[1]*z, -25.0, 25.0)
        return float(np.mean(np.logaddexp(0.0, eta)-y*eta))
    result = minimize(objective, np.array([0.0,1.0]), method="L-BFGS-B")
    return {"intercept": float(result.x[0]), "slope": float(result.x[1]), "converged": bool(result.success)}


def metrics(frame: pd.DataFrame, p: np.ndarray) -> dict:
    y = frame.home_win.to_numpy(float)
    p = np.clip(np.asarray(p, dtype=float), 1e-10, 1-1e-10)
    ll = -(y*np.log(p)+(1-y)*np.log1p(-p))
    return {
        "games": int(len(y)), "log_loss": float(ll.mean()),
        "brier": float(np.mean((p-y)**2)),
        "accuracy": float(np.mean((p>=0.5)==y.astype(bool))),
        "mean_home_probability": float(p.mean()), "home_win_rate": float(y.mean()),
        "calibration_intercept_slope": calibration(y,p),
    }


def paired_block_ci(pred: pd.DataFrame, reps: int = BOOTSTRAP_REPS) -> dict:
    y = pred.home_win.to_numpy(float)
    p0 = np.clip(pred.p_base.to_numpy(float), 1e-10, 1-1e-10)
    p1 = np.clip(pred.p_home_candidate.to_numpy(float), 1e-10, 1-1e-10)
    ll0 = -(y*np.log(p0)+(1-y)*np.log1p(-p0))
    ll1 = -(y*np.log(p1)+(1-y)*np.log1p(-p1))
    b0,b1=(p0-y)**2,(p1-y)**2
    dates=pd.to_datetime(pred.game_date).dt.normalize()
    block=((dates-dates.min()).dt.days//14).to_numpy()
    unique=np.unique(block); members=[np.flatnonzero(block==b) for b in unique]
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    dl,db=np.empty(reps),np.empty(reps)
    for i in range(reps):
        sample=np.concatenate([members[j] for j in rng.integers(0,len(members),size=len(members))])
        dl[i]=(ll1[sample]-ll0[sample]).mean(); db[i]=(b1[sample]-b0[sample]).mean()
    return {
        "log_loss_delta_candidate_minus_base":float((ll1-ll0).mean()),
        "log_loss_95pct_14_day_block_ci":[float(x) for x in np.quantile(dl,[.025,.975])],
        "brier_delta_candidate_minus_base":float((b1-b0).mean()),
        "brier_95pct_14_day_block_ci":[float(x) for x in np.quantile(db,[.025,.975])],
        "bootstrap_replicates":reps,"calendar_blocks":int(len(unique)),
    }


def run(base_path: str, starter_path: str, outdir: str) -> dict:
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    games,history=add_individual_goalie_history(load_games(base_path,starter_path))
    all_pred=[]; folds={}
    for target in TARGET_SEASONS:
        train=games.loc[games.source_season.le(target-2)].copy()
        val=games.loc[games.source_season.eq(target-1)].copy()
        test=games.loc[games.source_season.eq(target)].copy()
        if len(train)<1000 or len(val)!=1312 or len(test)!=1312:
            raise ValueError(f"Incomplete rolling-origin season {target}")
        base_val=metrics(val,val.p_base.to_numpy(float))
        choices=[{"feature":None,"penalty":None,"beta":0.0,"validation_log_loss":base_val["log_loss"]}]
        for window in WINDOWS:
            for prior in PRIORS:
                feature=feature_for(window,prior)
                for penalty in PENALTIES:
                    beta=fit_beta(train,feature,penalty)
                    pval=predict(val,feature,beta)
                    choices.append({"feature":feature,"penalty":penalty,"beta":beta,
                                    "validation_log_loss":metrics(val,pval)["log_loss"]})
        selected=min(choices,key=lambda r:r["validation_log_loss"])
        if selected["feature"] is None or selected["validation_log_loss"]>=base_val["log_loss"]:
            feature,penalty,beta=None,None,0.0
        else:
            feature,penalty=selected["feature"],float(selected["penalty"])
            beta=fit_beta(pd.concat([train,val],ignore_index=True),feature,penalty)
        pbase=test.p_base.to_numpy(float); pcandidate=predict(test,feature,beta)
        pred=test[["game_id","game_date","source_season","home_team","away_team","home_goalie_id","away_goalie_id","home_goalie_name","away_goalie_name","home_win","p_base","home_prior_goalie_starts","away_prior_goalie_starts","home_raw_prior_goalie_win_pct","away_raw_prior_goalie_win_pct","home_prior_expected_goalie_win_pct","away_prior_expected_goalie_win_pct"]].copy()
        pred["p_home_candidate"]=pcandidate; pred["p_away_base"]=1-pbase; pred["p_away_candidate"]=1-pcandidate
        pred["selected_feature"]=feature or "baseline_no_goalie"; pred["selected_penalty"]=penalty; pred["refit_beta"]=beta
        if feature:
            pred["home_individual_goalie_win_residual"]=test[f"home_{feature}"].to_numpy(float)
            pred["away_individual_goalie_win_residual"]=test[f"away_{feature}"].to_numpy(float)
        else:
            pred["home_individual_goalie_win_residual"]=0.0; pred["away_individual_goalie_win_residual"]=0.0
        all_pred.append(pred)
        folds[str(target)]={
            "training_seasons":sorted(int(s) for s in train.source_season.unique()),
            "validation_season":target-1,"test_season":target,"games":int(len(test)),
            "selected_feature":feature or "baseline_no_goalie","selected_ridge_penalty":penalty,"refit_beta":beta,
            "validation_base_log_loss":base_val["log_loss"],"validation_selected_log_loss":float(selected["validation_log_loss"]),
            "test_baseline":metrics(test,pbase),"test_candidate":metrics(test,pcandidate),
            "mean_absolute_probability_shift_pp":float(100*np.mean(np.abs(pcandidate-pbase))),
            "mean_signed_probability_shift_pp":float(100*np.mean(pcandidate-pbase)),
            "starter_individual_prior_history_available_pct":float(np.mean((test.home_prior_goalie_starts>0)&(test.away_prior_goalie_starts>0))),
        }
    pred=pd.concat(all_pred,ignore_index=True).sort_values(["game_date","game_id"]).reset_index(drop=True)
    summary={
        "experiment_id":EXPERIMENT_ID,"model_id":MODEL_ID,"status":"RESEARCH; no promotion or public export",
        "question":"Does an individual named goalie's own prior team win record, adjusted for the frozen champion's expected wins across all teams, improve projected game win percentage?",
        "baseline":"Frozen Hockey Lab Game v1.0.0 Team Strength 004 + B2B OOF probabilities",
        "target_seasons":TARGET_SEASONS,"target_games":int(len(pred)),
        "feature":"For each individual goalie ID across all franchises, prior starts' team wins minus HWIN champion-expected wins, shrunk by prior starts. Home-goalie residual minus away-goalie residual adjusts the frozen win logit.",
        "not_team_mix":"The feature follows the individual named goalie across teams. It does not compare the goalie to a team's previous goalie rotation.",
        "starter_identity_caveat":"Target goalie identity comes from the postgame goalie box, so this is retrospective upper-bound evidence until a timestamped pregame starter feed is established.",
        "history_scope":"HWIN out-of-fold regular-season predictions available from source seasons 2022–2026; each feature uses only the individual goalie's strictly prior starts.",
        "selection":"For target S, train through S-2; select goalie memory, shrinkage and ridge penalty on S-1 by win log loss; refit through S-1; test on S. Baseline is retained when it wins validation.",
        "markets_or_goal_model":"No markets or goals/xG model inputs; this is a direct projected win-percentage experiment.",
        "same_day_leakage_check":"All history is shifted by one goalie start; same-date team duplicates and same-date goalie starts are asserted absent.",
        "folds":folds,"pooled_baseline":metrics(pred,pred.p_base.to_numpy(float)),
        "pooled_candidate":metrics(pred,pred.p_home_candidate.to_numpy(float)),
        "pooled_mean_absolute_probability_shift_pp":float(100*np.mean(np.abs(pred.p_home_candidate-pred.p_base))),
        "pooled_mean_signed_probability_shift_pp":float(100*np.mean(pred.p_home_candidate-pred.p_base)),
        "paired_14_day_block_bootstrap":paired_block_ci(pred),
        "home_away_probability_sum_max_abs_error":float(np.max(np.abs(pred.p_home_candidate+pred.p_away_candidate-1.0))),
        "decision":"Research only; no forced nonzero adjustment. Promote only after stable proper-score improvement and replacement of retrospective target starter identity with timestamped pregame starter data.",
        "artifacts":{"predictions":"oof_individual_goalie_win_pct.parquet"},
    }
    pred.to_parquet(out/"oof_individual_goalie_win_pct.parquet",index=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline",default="research/exports/HWIN_PREVIOUS_SEASON_BATTERY_011_20261103/results/goalie_battery_011/oof_predictions.parquet")
    parser.add_argument("--starters",default="research/.tmp_goalie_probability_016/goalie/data/goalie_feature_output/goalie_game_features.parquet")
    parser.add_argument("--out",default="research/.tmp_goalie_probability_016/out/hwin_individual_goalie_win_pct_025")
    args=parser.parse_args()
    run(args.baseline,args.starters,args.out)
