"""Age-curve and exposure calibration for a goalie-only logit adjustment.

This is a research candidate. It uses nested chronological seasons and an
independent team-only baseline. It cannot directly promote a production model:
the baseline feature version is HWIN-GOALIE-ROLLING-003, so a candidate export
must be recalibrated against the active production backbone before activation.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0,"research")
import hwin_goalie_layer_001 as gl
import hwin_rolling_backtest_001 as rb

TARGET_SEASONS=[2022,2023,2024,2025,2026]
C_GRID=[0.05,0.1,0.25,0.5,1.0,2.0]
SHOT_K=[100,200,400]
START_K=[0,5,15]
MINUTE_K=[0,600,1800]
TEAM_FEATURES=["elo_logit","d_fivev5_sog_share_r20","d_fivev5_xg_share_r20",
               "d_goals_non_en_r20","d_rest_days"]


def metrics(y,p):
    y=np.asarray(y,dtype=int)
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    order=np.argsort(p)
    bins=np.array_split(order,min(10,len(order))) if len(order) else []
    calibration_error=sum(
        len(idx)/len(y)*abs(float(y[idx].mean())-float(p[idx].mean()))
        for idx in bins if len(idx)
    ) if len(y) else float("nan")
    return {
        "n":int(len(y)),
        "brier":float(brier_score_loss(y,p)),
        "log_loss":float(log_loss(y,p,labels=[0,1])),
        "accuracy":float(accuracy_score(y,p>=0.5)),
        "calibration":float(calibration_error),
        "mean_pred":float(p.mean()),
        "actual_rate":float(y.mean())
    }


def fit_age_curve(goalies, through_season):
    """Fit a shot-weighted quadratic SV% curve using only earlier seasons."""
    g=goalies.copy()
    g["source_season"]=pd.to_numeric(g["source_season"],errors="coerce")
    g["age_years"]=pd.to_numeric(g["age_years"],errors="coerce")
    g["shots_against_pbp"]=pd.to_numeric(g["shots_against_pbp"],errors="coerce")
    g["saves_pbp"]=pd.to_numeric(g["saves_pbp"],errors="coerce")
    g=g[(g["source_season"]<=through_season)&g["starter_bool"].fillna(False)&
        g["age_years"].between(18,50)&(g["shots_against_pbp"]>0)].copy()
    if len(g)<100:
        raise RuntimeError(f"Insufficient prior goalie rows to fit age curve: {len(g)}")
    age=g["age_years"].to_numpy(float)
    shots=g["shots_against_pbp"].to_numpy(float)
    sv=(g["saves_pbp"]/g["shots_against_pbp"]).to_numpy(float)
    center=float(np.average(age,weights=shots))
    scale=float(np.sqrt(np.average((age-center)**2,weights=shots)))
    if not np.isfinite(scale) or scale<=0:
        raise RuntimeError("Invalid goalie age scale")
    z=(age-center)/scale
    design=np.column_stack([np.ones(len(z)),z,z*z])
    sw=np.sqrt(shots)
    coef=np.linalg.lstsq(design*sw[:,None],sv*sw,rcond=None)[0]
    return {"center":center,"scale":scale,"coefficients":[float(v) for v in coef]}


def curve_value(age,curve):
    if not np.isfinite(age):
        return np.nan
    z=(float(age)-curve["center"])/curve["scale"]
    b0,b1,b2=curve["coefficients"]
    value=b0+b1*z+b2*z*z
    return float(value) if 0<value<1 else np.nan


def goalie_ratings(goalies,curve,shots_k,starts_k,minutes_k):
    """Prior-only, age-residual rating with calibrated exposure shrinkage."""
    g=goalies[goalies["starter_bool"].fillna(False)].copy()
    g["game_date"]=pd.to_datetime(g["game_date"],errors="coerce")
    g["goalie_id_key"]=g["goalie_id_key"].astype("string")
    for col in ["age_years","shots_against_pbp","saves_pbp","toi_seconds_box"]:
        g[col]=pd.to_numeric(g[col],errors="coerce")
    g=g.sort_values(["goalie_id_key","game_date","game_id"]).reset_index(drop=True)
    g["age_prior"]=g["age_years"].map(lambda age:curve_value(age,curve))
    g["age_prior"]=g["age_prior"].fillna(float(curve["coefficients"][0]))
    g["expected_saves_age"]=g["shots_against_pbp"].fillna(0)*g["age_prior"]
    rows=[]
    for pid,z in g.groupby("goalie_id_key",sort=False):
        shots=z["shots_against_pbp"].fillna(0).to_numpy(float)
        saves=z["saves_pbp"].fillna(0).to_numpy(float)
        toi=np.maximum(z["toi_seconds_box"].fillna(0).to_numpy(float)/60.0,0)
        expected=z["expected_saves_age"].fillna(0).to_numpy(float)
        age=z["age_years"].to_numpy(float)
        prior_shots=np.concatenate([[0.0],np.cumsum(shots)[:-1]])
        prior_saves=np.concatenate([[0.0],np.cumsum(saves)[:-1]])
        prior_expected=np.concatenate([[0.0],np.cumsum(expected)[:-1]])
        prior_minutes=np.concatenate([[0.0],np.cumsum(toi)[:-1]])
        prior_starts=np.arange(len(z),dtype=float)
        residual=np.divide(prior_saves-prior_expected,prior_shots,
                           out=np.zeros(len(z),dtype=float),where=prior_shots>0)
        shot_rel=np.divide(prior_shots,prior_shots+shots_k,
                           out=np.zeros(len(z),dtype=float),where=(prior_shots+shots_k)>0)
        start_rel=(prior_starts/(prior_starts+starts_k) if starts_k>0
                   else np.where(prior_starts>0,1.0,0.0))
        minute_rel=(prior_minutes/(prior_minutes+minutes_k) if minutes_k>0
                    else np.where(prior_minutes>0,1.0,0.0))
        reliability=(shot_rel+start_rel+minute_rel)/3.0
        current_prior=np.array([curve_value(a,curve) for a in age],dtype=float)
        rating=current_prior+reliability*residual
        block=z[["game_id","home_away","goalie_id_key","player_name"]].copy()
        block["goalie_rating"]=rating
        block["prior_starts"]=prior_starts.astype(int)
        block["prior_shots"]=prior_shots
        block["prior_minutes"]=prior_minutes
        rows.append(block)
    out=pd.concat(rows,ignore_index=True)
    return out


def add_goalie_diff(x,ratings):
    h=ratings[ratings["home_away"].astype("string").str.lower().eq("home")].drop_duplicates("game_id")
    a=ratings[ratings["home_away"].astype("string").str.lower().eq("away")].drop_duplicates("game_id")
    h=h[["game_id","goalie_rating","prior_starts","prior_shots","prior_minutes"]].rename(
        columns={c:"h_"+c for c in ["goalie_rating","prior_starts","prior_shots","prior_minutes"]})
    a=a[["game_id","goalie_rating","prior_starts","prior_shots","prior_minutes"]].rename(
        columns={c:"a_"+c for c in ["goalie_rating","prior_starts","prior_shots","prior_minutes"]})
    z=x.merge(h,on="game_id",how="left").merge(a,on="game_id",how="left")
    z["d_goalie_rating"]=pd.to_numeric(z["h_goalie_rating"],errors="coerce")-pd.to_numeric(z["a_goalie_rating"],errors="coerce")
    return z


def fit_base(train,val,test):
    best=None
    for C in C_GRID:
        pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                       ("m",LogisticRegression(C=C,max_iter=1800))])
        pipe.fit(train[TEAM_FEATURES],train.home_win)
        pv=pipe.predict_proba(val[TEAM_FEATURES])[:,1]
        ll=log_loss(val.home_win,np.clip(pv,1e-6,1-1e-6),labels=[0,1])
        if best is None or ll<best[0]:
            best=(ll,C,pipe,pv)
    C=best[1]
    train_only_pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                              ("m",LogisticRegression(C=C,max_iter=1800))])
    train_only_pipe.fit(train[TEAM_FEATURES],train.home_win)
    val_prob=train_only_pipe.predict_proba(val[TEAM_FEATURES])[:,1]
    fit=pd.concat([train,val],ignore_index=True)
    final_pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                         ("m",LogisticRegression(C=C,max_iter=1800))])
    final_pipe.fit(fit[TEAM_FEATURES],fit.home_win)
    test_prob=final_pipe.predict_proba(test[TEAM_FEATURES])[:,1]
    return C,val_prob,test_prob


def fit_beta(p,y,diff):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    y=np.asarray(y,dtype=float)
    d=np.asarray(diff,dtype=float)
    keep=np.isfinite(p)&np.isfinite(y)&np.isfinite(d)
    p,y,d=p[keep],y[keep],d[keep]
    if len(y)<100:
        raise RuntimeError(f"Insufficient paired goalie validation rows: {len(y)}")
    base=np.log(p/(1-p))
    beta=0.0
    for _ in range(100):
        z=np.clip(base+beta*d,-30,30)
        q=1/(1+np.exp(-z))
        grad=np.sum((q-y)*d)
        hess=np.sum(q*(1-q)*d*d)+1e-12
        step=grad/hess
        beta-=step
        if abs(step)<1e-9:
            break
    return float(beta),keep


def apply_beta(p,diff,beta):
    p=np.asarray(p,dtype=float)
    d=np.asarray(diff,dtype=float)
    out=np.clip(p,1e-6,1-1e-6).copy()
    keep=np.isfinite(d)
    z=np.log(out[keep]/(1-out[keep]))+beta*d[keep]
    out[keep]=1/(1+np.exp(-np.clip(z,-30,30)))
    return out


def main(pbp_path,spine_path,goalie_path,outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    pbp=pd.read_parquet(pbp_path)
    spine=pd.read_parquet(spine_path)
    goalies=pd.read_parquet(goalie_path)
    spine=spine[spine.result_available.fillna(False)&spine.game_type.astype("string").isin(["R","P"])].copy()
    spine["game_date"]=pd.to_datetime(spine.game_date,errors="coerce")
    spine["source_season"]=pd.to_numeric(spine.source_season,errors="coerce")
    x=gl.build_team_features(pbp)

    elo={}
    for k,ha,ret in rb.ELO_GRID:
        key=f"k{k}_h{ha}_r{ret}"
        elo[key]=rb.elo_all(spine,k,ha,ret)

    rows=[]
    selected=[]
    for target in TARGET_SEASONS:
        val_season=target-1
        train=x[x.source_season.le(target-2)].copy()
        val=x[x.source_season.eq(val_season)].copy()
        test=x[x.source_season.eq(target)].copy()
        if min(len(train),len(val),len(test))<100:
            continue

        elo_choice=None
        for key,pred in elo.items():
            v=val[["game_id","home_win"]].merge(pred[pred.source_season.eq(val_season)][["game_id","p"]],on="game_id")
            if len(v)<100: continue
            ll=log_loss(v.home_win,np.clip(v.p,1e-6,1-1e-6),labels=[0,1])
            if elo_choice is None or ll<elo_choice[0]:
                elo_choice=(ll,key)
        key=elo_choice[1]
        el=elo[key][["game_id","p"]].rename(columns={"p":"elo_p"})
        z=x.merge(el,on="game_id",how="left")
        z["elo_logit"]=np.log(np.clip(z.elo_p,1e-6,1-1e-6)/np.clip(1-z.elo_p,1e-6,1-1e-6))
        train=z[z.source_season.le(target-2)].copy()
        val=z[z.source_season.eq(val_season)].copy()
        test=z[z.source_season.eq(target)].copy()
        base_c,base_val_p,base_test_p=fit_base(train,val,test)
        base_metrics=metrics(test.home_win,base_test_p)

        age_curve=fit_age_curve(goalies,through_season=target-2)
        best=None
        for sk in SHOT_K:
            for nk in START_K:
                for mk in MINUTE_K:
                    ratings=goalie_ratings(goalies,age_curve,sk,nk,mk)
                    zratings=add_goalie_diff(z,ratings)
                    train_g=zratings[zratings.source_season.le(target-2)].copy()
                    val_g=zratings[zratings.source_season.eq(val_season)].copy()
                    test_g=zratings[zratings.source_season.eq(target)].copy()
                    diff=val_g["d_goalie_rating"].to_numpy(float)
                    beta,keep=fit_beta(base_val_p,val_g.home_win.to_numpy(float),diff)
                    pv=apply_beta(base_val_p,diff,beta)
                    val_ll=log_loss(val_g.home_win.to_numpy(int)[keep],pv[keep],labels=[0,1])
                    if best is None or val_ll<best["validation_log_loss"]:
                        best={"shots_k":sk,"starts_k":nk,"minutes_k":mk,"beta":beta,
                              "validation_log_loss":float(val_ll),"ratings":ratings}
        chosen=best
        rating_frame=add_goalie_diff(z,chosen["ratings"])
        test_g=rating_frame[rating_frame.source_season.eq(target)].copy()
        adjusted=apply_beta(base_test_p,test_g["d_goalie_rating"].to_numpy(float),chosen["beta"])
        adjusted_metrics=metrics(test_g.home_win,adjusted)
        row={
            "target_season":target,
            "validation_season":val_season,
            "games":adjusted_metrics["n"],
            "base_c":base_c,
            "shots_k":chosen["shots_k"],
            "starts_k":chosen["starts_k"],
            "minutes_k":chosen["minutes_k"],
            "probability_beta":chosen["beta"],
            "validation_log_loss":chosen["validation_log_loss"],
            "backbone_brier":base_metrics["brier"],
            "goalie_brier":adjusted_metrics["brier"],
            "delta_brier":adjusted_metrics["brier"]-base_metrics["brier"],
            "backbone_log_loss":base_metrics["log_loss"],
            "goalie_log_loss":adjusted_metrics["log_loss"],
            "delta_log_loss":adjusted_metrics["log_loss"]-base_metrics["log_loss"],
            "backbone_accuracy":base_metrics["accuracy"],
            "goalie_accuracy":adjusted_metrics["accuracy"],
            "goalie_calibration":adjusted_metrics["calibration"],
            "age_curve":age_curve
        }
        rows.append(row)
        selected.append(chosen)
        print(json.dumps({k:v for k,v in row.items() if k!="age_curve"},flush=True))

    if not rows:
        raise RuntimeError("No goalie rolling folds produced")
    result=pd.DataFrame(rows)
    result.to_csv(out/"rolling_age_exposure_backtest.csv",index=False)
    n=result.games.to_numpy(float)
    base_brier=float(np.average(result.backbone_brier,weights=n))
    goalie_brier=float(np.average(result.goalie_brier,weights=n))
    base_ll=float(np.average(result.backbone_log_loss,weights=n))
    goalie_ll=float(np.average(result.goalie_log_loss,weights=n))
    perf={
        "games_scored":int(result.games.sum()),
        "accuracy":float(np.average(result.goalie_accuracy,weights=n)),
        "brier":goalie_brier,
        "log_loss":goalie_ll,
        "calibration":float(np.average(result.goalie_calibration,weights=n))
    }
    # Candidate parameters are deliberately not promoted. The production
    # feature backbone differs from this research baseline and must be refit.
    curve=fit_age_curve(goalies,through_season=2026)
    selected_cfg=Counter((x["shots_k"],x["starts_k"],x["minutes_k"]) for x in selected)
    mode,mode_count=selected_cfg.most_common(1)[0]
    betas=[float(x["beta"]) for x in selected]
    candidate={
        "schema_version":"1.0",
        "model_id":"HOCKEY-LAB-GAME-GOALIE",
        "version":"candidate-0.2.0",
        "status":"CANDIDATE",
        "feature_version":"hwin-goalie-rolling-003+goalie-adjustment-candidate",
        "baseline_feature_version":"HWIN-GOALIE-ROLLING-003",
        "trained_through":"2025-26",
        "sample":{"games_scored":perf["games_scored"],"start_date":"2022 rolling-origin fold","end_date":"2026 rolling-origin fold"},
        "performance_metrics":perf,
        "calibration":{
            "age_curve":curve,
            "shrinkage":{"shots_k":mode[0],"starts_k":mode[1],"minutes_k":mode[2]},
            "probability_beta":float(np.median(betas))
        },
        "rolling_origin":{
            "weighted_backbone_brier":base_brier,
            "weighted_goalie_brier":goalie_brier,
            "delta_brier":goalie_brier-base_brier,
            "weighted_backbone_log_loss":base_ll,
            "weighted_goalie_log_loss":goalie_ll,
            "delta_log_loss":goalie_ll-base_ll,
            "selected_shrinkage_folds":mode_count,
            "folds":result[[ "target_season","games","shots_k","starts_k","minutes_k","probability_beta","backbone_brier","goalie_brier","delta_brier","backbone_log_loss","goalie_log_loss","delta_log_loss" ]].to_dict(orient="records")
        },
        "promotion_note":"RESEARCH ONLY. This candidate uses HWIN-GOALIE-ROLLING-003 as its team baseline, not the current team-strength-004+b2b-007 production backbone. Refit and validate against the production baseline before promotion."
    }
    with open(out/"goalie-adjustment-candidate.json","w") as f:
        json.dump(candidate,f,indent=2)
    summary={
        "folds":len(result),
        "games":int(result.games.sum()),
        "weighted_backbone_brier":base_brier,
        "weighted_goalie_brier":goalie_brier,
        "delta_brier":goalie_brier-base_brier,
        "weighted_backbone_log_loss":base_ll,
        "weighted_goalie_log_loss":goalie_ll,
        "delta_log_loss":goalie_ll-base_ll,
        "selected_shrinkage":candidate["calibration"]["shrinkage"],
        "probability_beta_median":candidate["calibration"]["probability_beta"],
        "age_curve":curve,
        "promotion_status":"CANDIDATE_ONLY"
    }
    with open(out/"summary.json","w") as f: json.dump(summary,f,indent=2)
    print("FINAL",json.dumps(summary,indent=2))


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--pbp",required=True)
    p.add_argument("--spine",required=True)
    p.add_argument("--goalie",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    main(a.pbp,a.spine,a.goalie,a.out)
