"""Test named goalie skill directly on frozen Hockey Lab win probabilities.

For each game, the sole feature is the difference between the two actual
named starters' EXP-0035 calibrated, prior-only GSAx-per-30-shot ratings. A
nonnegative coefficient adjusts the frozen HWIN champion logit. No team
goalie mix, team-goalie W-L record, goal rate or market input is used.

Target starter IDs are from postgame goalie boxes, so this is a retrospective
upper-bound test until time-stamped pregame starter snapshots are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit


EXPERIMENT_ID = "EXP-0041"
MODEL_ID = "HOCKEY-LAB-HWIN-INDIVIDUAL-GOALIE-SKILL-026-RESEARCH"
BASE_VARIANT = "team_strength_004_plus_b2b"
TARGET_SEASONS = [2024, 2025, 2026]
PENALTIES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 4103
SKILL_GAP_SCALE = 0.1


def norm_ids(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True)


def load_matched(base_path: str, starter_path: str, skill_path: str) -> tuple[pd.DataFrame, int]:
    base = pd.read_parquet(base_path)
    base = base.loc[
        base.model_variant.eq(BASE_VARIANT)
        & base.game_type.eq("R")
        & base.source_season.between(2022, 2026),
        ["game_id", "game_date", "source_season", "home_team", "away_team", "home_win", "p_home"],
    ].copy()
    base["game_id"] = norm_ids(base.game_id)
    base["game_date"] = pd.to_datetime(base.game_date, errors="coerce").dt.normalize()
    base["source_season"] = base.source_season.astype(int)
    base["home_win"] = base.home_win.astype(int)
    base["p_base"] = base.p_home.astype(float).clip(1e-8, 1-1e-8)
    if len(base) != 5 * 1312 or base.game_id.duplicated().any():
        raise ValueError("Frozen HWIN champion OOF ledger has an unexpected game count or duplicates")

    starters = pd.read_parquet(starter_path)
    starters["game_id"] = norm_ids(starters.game_id)
    starters = starters.loc[
        starters.starter_bool.fillna(False).astype(bool)
        & starters.source_season.between(2022, 2026)
        & starters.game_id.isin(base.game_id),
        ["game_id", "source_season", "goalie_id_key", "home_away", "team_abbrev", "player_name", "game_date"],
    ].copy()
    starters["source_season"] = starters.source_season.astype(int)
    starters["goalie_id_key"] = starters.goalie_id_key.astype("string")
    starters["game_date"] = pd.to_datetime(starters.game_date, errors="coerce").dt.normalize()
    if starters.duplicated(["game_id", "home_away"]).any() or starters.groupby("game_id").size().ne(2).any():
        raise ValueError("Target starter identity mapping is ambiguous")

    skill = pd.read_parquet(skill_path)
    skill = skill.loc[
        skill.game_type.eq("R") & skill.source_season.between(2018, 2026),
        ["game_id", "game_date", "goalie_id_key", "team_abbrev", "predicted_skill_delta_per30",
         "selected_feature", "prior_starts"],
    ].copy()
    skill["game_id"] = norm_ids(skill.game_id)
    skill["goalie_id_key"] = skill.goalie_id_key.astype("string")
    skill["game_date"] = pd.to_datetime(skill.game_date, errors="coerce").dt.normalize()
    if skill.duplicated(["game_id", "goalie_id_key"]).any():
        raise ValueError("Calibrated goalie ratings have duplicate game-goalie rows")

    goalie = starters.merge(
        skill,
        on=["game_id", "goalie_id_key", "team_abbrev"],
        how="left", validate="one_to_one", suffixes=("", "_skill"),
    )
    goalie["skill_rating_fallback"] = goalie.predicted_skill_delta_per30.isna()
    fallback_count = int(goalie.skill_rating_fallback.sum())
    for idx in goalie.index[goalie.skill_rating_fallback]:
        row = goalie.loc[idx]
        prior = skill.loc[
            skill.goalie_id_key.eq(str(row.goalie_id_key))
            & skill.game_date.lt(row.game_date)
        ].sort_values("game_date")
        if prior.empty:
            goalie.loc[idx, "predicted_skill_delta_per30"] = 0.0
            goalie.loc[idx, "selected_feature"] = "neutral_no_prior_rating"
            goalie.loc[idx, "prior_starts"] = 0.0
        else:
            last = prior.iloc[-1]
            goalie.loc[idx, "predicted_skill_delta_per30"] = float(last.predicted_skill_delta_per30)
            goalie.loc[idx, "selected_feature"] = f"carry_forward:{last.selected_feature}"
            goalie.loc[idx, "prior_starts"] = float(last.prior_starts)
    if goalie.predicted_skill_delta_per30.isna().any():
        raise ValueError("Missing individual goalie rating after prior-only fallback")

    def side_data(side: str, prefix: str) -> pd.DataFrame:
        x = goalie.loc[goalie.home_away.eq(side)].rename(columns={
            "goalie_id_key": f"{prefix}_goalie_id", "team_abbrev": f"{prefix}_goalie_team",
            "player_name": f"{prefix}_goalie_name", "predicted_skill_delta_per30": f"{prefix}_goalie_skill",
            "selected_feature": f"{prefix}_skill_feature", "prior_starts": f"{prefix}_goalie_prior_starts",
            "skill_rating_fallback": f"{prefix}_skill_rating_fallback",
        })
        return x[["game_id", f"{prefix}_goalie_id", f"{prefix}_goalie_team", f"{prefix}_goalie_name",
                  f"{prefix}_goalie_skill", f"{prefix}_skill_feature", f"{prefix}_goalie_prior_starts",
                  f"{prefix}_skill_rating_fallback"]]

    games = base.merge(side_data("home", "home"), on="game_id", how="inner", validate="one_to_one")
    games = games.merge(side_data("away", "away"), on="game_id", how="inner", validate="one_to_one")
    if len(games) != len(base):
        raise ValueError("Named goalie coverage does not match HWIN baseline")
    if not games.home_team.astype(str).eq(games.home_goalie_team.astype(str)).all():
        raise ValueError("Home goalie team does not match the game")
    if not games.away_team.astype(str).eq(games.away_goalie_team.astype(str)).all():
        raise ValueError("Away goalie team does not match the game")
    games["individual_goalie_skill_gap"] = games.home_goalie_skill.astype(float) - games.away_goalie_skill.astype(float)
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True), fallback_count


def fit_beta(frame: pd.DataFrame, penalty: float) -> float:
    y = frame.home_win.to_numpy(float)
    offset = logit(frame.p_base.to_numpy(float))
    x = frame.individual_goalie_skill_gap.to_numpy(float) / SKILL_GAP_SCALE

    def objective(beta: np.ndarray) -> float:
        eta = np.clip(offset + beta[0]*x, -25.0, 25.0)
        return float(np.mean(np.logaddexp(0.0, eta)-y*eta) + 0.5*penalty*beta[0]**2)

    result = minimize(objective, np.zeros(1), method="L-BFGS-B", bounds=[(0.0, 20.0)],
                      options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000})
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Named goalie skill fit failed: {result.message}")
    return float(result.x[0])


def predict(frame: pd.DataFrame, beta: float) -> np.ndarray:
    eta = logit(frame.p_base.to_numpy(float))
    eta += beta * frame.individual_goalie_skill_gap.to_numpy(float) / SKILL_GAP_SCALE
    p = expit(np.clip(eta, -25.0, 25.0))
    if not np.allclose(p + (1.0-p), 1.0, atol=1e-12):
        raise ValueError("Home/away probabilities fail sum-to-one check")
    return p


def calibration(y: np.ndarray, p: np.ndarray) -> dict:
    y=np.asarray(y,float); z=logit(np.clip(np.asarray(p,float),1e-8,1-1e-8))
    def objective(theta):
        eta=np.clip(theta[0]+theta[1]*z,-25,25)
        return float(np.mean(np.logaddexp(0,eta)-y*eta))
    result=minimize(objective,np.array([0.0,1.0]),method="L-BFGS-B")
    return {"intercept":float(result.x[0]),"slope":float(result.x[1]),"converged":bool(result.success)}


def metrics(frame: pd.DataFrame, p: np.ndarray) -> dict:
    y=frame.home_win.to_numpy(float); p=np.clip(np.asarray(p,float),1e-10,1-1e-10)
    ll=-(y*np.log(p)+(1-y)*np.log1p(-p))
    return {"games":int(len(y)),"log_loss":float(ll.mean()),"brier":float(np.mean((p-y)**2)),
            "accuracy":float(np.mean((p>=.5)==y.astype(bool))),"mean_home_probability":float(p.mean()),
            "home_win_rate":float(y.mean()),"calibration_intercept_slope":calibration(y,p)}


def paired_ci(pred: pd.DataFrame) -> dict:
    y=pred.home_win.to_numpy(float); p0=np.clip(pred.p_base.to_numpy(float),1e-10,1-1e-10)
    p1=np.clip(pred.p_home_candidate.to_numpy(float),1e-10,1-1e-10)
    ll0=-(y*np.log(p0)+(1-y)*np.log1p(-p0)); ll1=-(y*np.log(p1)+(1-y)*np.log1p(-p1))
    b0,b1=(p0-y)**2,(p1-y)**2
    dates=pd.to_datetime(pred.game_date).dt.normalize(); block=((dates-dates.min()).dt.days//14).to_numpy()
    unique=np.unique(block); members=[np.flatnonzero(block==b) for b in unique]
    rng=np.random.default_rng(BOOTSTRAP_SEED); dl=np.empty(BOOTSTRAP_REPS); db=np.empty(BOOTSTRAP_REPS)
    for i in range(BOOTSTRAP_REPS):
        ix=np.concatenate([members[j] for j in rng.integers(0,len(members),len(members))])
        dl[i]=(ll1[ix]-ll0[ix]).mean(); db[i]=(b1[ix]-b0[ix]).mean()
    return {"log_loss_delta_candidate_minus_base":float((ll1-ll0).mean()),
            "log_loss_95pct_14_day_block_ci":[float(x) for x in np.quantile(dl,[.025,.975])],
            "brier_delta_candidate_minus_base":float((b1-b0).mean()),
            "brier_95pct_14_day_block_ci":[float(x) for x in np.quantile(db,[.025,.975])],
            "bootstrap_replicates":BOOTSTRAP_REPS,"calendar_blocks":int(len(unique))}


def run(base_path: str, starter_path: str, skill_path: str, outdir: str) -> dict:
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    games,fallback_count=load_matched(base_path,starter_path,skill_path)
    pred_folds=[]; folds={}
    for target in TARGET_SEASONS:
        train=games.loc[games.source_season.le(target-2)].copy()
        val=games.loc[games.source_season.eq(target-1)].copy()
        test=games.loc[games.source_season.eq(target)].copy()
        if len(train)<1000 or len(val)!=1312 or len(test)!=1312:
            raise ValueError(f"Incomplete rolling-origin fold {target}")
        base_val=metrics(val,val.p_base.to_numpy(float))
        choices=[{"penalty":None,"beta":0.0,"validation_log_loss":base_val["log_loss"],"family":"baseline_no_goalie"}]
        for penalty in PENALTIES:
            beta=fit_beta(train,penalty); pval=predict(val,beta)
            choices.append({"penalty":penalty,"beta":beta,"validation_log_loss":metrics(val,pval)["log_loss"],"family":"individual_goalie_skill"})
        selected=min(choices,key=lambda z:z["validation_log_loss"])
        if selected["family"]=="baseline_no_goalie" or selected["validation_log_loss"]>=base_val["log_loss"]:
            beta=0.0; penalty=None; family="baseline_no_goalie"
        else:
            penalty=float(selected["penalty"]); family="individual_goalie_skill"
            beta=fit_beta(pd.concat([train,val],ignore_index=True),penalty)
        pbase=test.p_base.to_numpy(float); pcandidate=predict(test,beta)
        p=test[["game_id","game_date","source_season","home_team","away_team","home_goalie_id","away_goalie_id",
                "home_goalie_name","away_goalie_name","home_win","p_base","home_goalie_skill","away_goalie_skill",
                "home_skill_feature","away_skill_feature","home_goalie_prior_starts","away_goalie_prior_starts",
                "home_skill_rating_fallback","away_skill_rating_fallback","individual_goalie_skill_gap"]].copy()
        p["p_home_candidate"]=pcandidate; p["p_away_base"]=1-pbase; p["p_away_candidate"]=1-pcandidate
        p["selected_family"]=family; p["selected_penalty"]=penalty; p["refit_beta_per_0p1_gsax30_gap"]=beta
        pred_folds.append(p)
        folds[str(target)]={
            "training_seasons":sorted(int(s) for s in train.source_season.unique()),"validation_season":target-1,
            "test_season":target,"games":int(len(test)),"selected_family":family,"selected_ridge_penalty":penalty,
            "refit_beta_per_0p1_gsax30_gap":beta,"validation_base_log_loss":base_val["log_loss"],
            "validation_selected_log_loss":float(selected["validation_log_loss"]),
            "test_baseline":metrics(test,pbase),"test_candidate":metrics(test,pcandidate),
            "mean_absolute_probability_shift_pp":float(100*np.mean(np.abs(pcandidate-pbase))),
            "mean_signed_probability_shift_pp":float(100*np.mean(pcandidate-pbase)),
            "mean_abs_goalie_skill_gap_per30":float(test.individual_goalie_skill_gap.abs().mean()),
        }
    pred=pd.concat(pred_folds,ignore_index=True).sort_values(["game_date","game_id"]).reset_index(drop=True)
    summary={
        "experiment_id":EXPERIMENT_ID,"model_id":MODEL_ID,"status":"RESEARCH; no promotion or public export",
        "question":"Does the exact named starter's calibrated individual GSAx/30 skill change and improve frozen HWIN projected win percentage?",
        "baseline":"Frozen Hockey Lab Game v1.0.0 Team Strength 004 + B2B OOF win probabilities",
        "target_seasons":TARGET_SEASONS,"target_games":int(len(pred)),
        "individual_goalie_feature":"Home actual starter's EXP-0035 calibrated prior-only GSAx/30 rating minus away actual starter's same rating. A single nonnegative coefficient adjusts the champion home-win logit.",
        "not_team_mix":"This compares the two individual goalies starting in the game; it does not compare either goalie to a team's previous goalie rotation.",
        "starter_identity_caveat":"Target starter IDs are from postgame goalie boxes, so this is retrospective upper-bound evidence pending timestamped pregame goalie snapshots.",
        "selection":"For each target season S, train through S-2, tune only ridge penalty on S-1, refit through S-1, and test on S. A no-goalie baseline is retained if validation log loss does not improve.",
        "skill_rating_fallback_rows":fallback_count,"folds":folds,
        "pooled_baseline":metrics(pred,pred.p_base.to_numpy(float)),"pooled_candidate":metrics(pred,pred.p_home_candidate.to_numpy(float)),
        "pooled_mean_absolute_probability_shift_pp":float(100*np.mean(np.abs(pred.p_home_candidate-pred.p_base))),
        "pooled_mean_signed_probability_shift_pp":float(100*np.mean(pred.p_home_candidate-pred.p_base)),
        "paired_14_day_block_bootstrap":paired_ci(pred),
        "home_away_probability_sum_max_abs_error":float(np.max(np.abs(pred.p_home_candidate+pred.p_away_candidate-1))),
        "decision":"Research only. Do not force a nonzero goalie effect or promote absent repeatable proper-score gain and a pregame starter timestamp.",
        "artifacts":{"predictions":"oof_individual_goalie_skill_win_pct.parquet"},
    }
    pred.to_parquet(out/"oof_individual_goalie_skill_win_pct.parquet",index=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline",default="research/exports/HWIN_PREVIOUS_SEASON_BATTERY_011_20261103/results/goalie_battery_011/oof_predictions.parquet")
    parser.add_argument("--starters",default="research/.tmp_goalie_probability_016/goalie/data/goalie_feature_output/goalie_game_features.parquet")
    parser.add_argument("--skill",default="research/.tmp_goalie_probability_016/out/goalie_pool_calibration_021/goalie_pool_oof_calibration.parquet")
    parser.add_argument("--out",default="research/.tmp_goalie_probability_016/out/hwin_individual_goalie_skill_026")
    args=parser.parse_args()
    run(args.baseline,args.starters,args.skill,args.out)
