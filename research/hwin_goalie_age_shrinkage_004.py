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
import hwin_team_strength_004 as ts
sys.path.insert(0,"src/production")
import hockey_lab_game_v1 as prod

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
    return {
        "center":center,
        "scale":scale,
        "age_min":float(age.min()),
        "age_max":float(age.max()),
        "coefficients":[float(v) for v in coef]
    }


def curve_value(age,curve):
    if not np.isfinite(age):
        return np.nan
    age=float(age)
    if "age_min" in curve and "age_max" in curve:
        age=float(np.clip(age,curve["age_min"],curve["age_max"]))
    z=(age-curve["center"])/curve["scale"]
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


def fit_base(train,val,test,features,C):
    train_pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                         ("m",LogisticRegression(C=C,max_iter=2500))])
    train_pipe.fit(train[features],train.home_win.astype(int))
    val_prob=train_pipe.predict_proba(val[features])[:,1]
    final_pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                         ("m",LogisticRegression(C=C,max_iter=2500))])
    final_pipe.fit(pd.concat([train,val],ignore_index=True)[features],
                   pd.concat([train,val],ignore_index=True).home_win.astype(int))
    return val_prob,final_pipe.predict_proba(test[features])[:,1]


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
    pbp=pd.read_parquet(pbp_path); spine=pd.read_parquet(spine_path)
    goalies=pd.read_parquet(goalie_path)
    spine=spine[spine.game_type.astype("string").isin(["R","P"])].copy()
    spine["game_date"]=pd.to_datetime(spine.game_date,errors="coerce")
    spine["source_season"]=pd.to_numeric(spine.source_season,errors="coerce")
    spine["home_win"]=pd.to_numeric(spine.home_win,errors="coerce")
    games=ts.add_flat_and_ewm_features(ts.game_event_table(pbp))
    x=games.merge(prod.build_schedule_features(spine),on="game_id",how="left")
    for alpha,retain in ts.OA_GRID:
        oa,tag=ts.opponent_adjusted_features(games,alpha,retain)
        x=x.merge(oa,on="game_id",how="left")
    elo={}
    for k,ha,retain in ts.ELO_GRID:
        elo[f"k{k}_h{ha}_r{retain}"]=prod.elo_with_future(spine,k,ha,retain)

    rows=[]; selected=[]
    for target in TARGET_SEASONS:
        vs=target-1
        train=x[x.source_season.le(target-2)&x.home_win.notna()].copy()
        val=x[x.source_season.eq(vs)&x.home_win.notna()].copy()
        test=x[x.source_season.eq(target)&x.home_win.notna()].copy()
        if min(len(train),len(val),len(test))<100: continue
        ec=None
        for key,pred in elo.items():
            v=val[["game_id","home_win"]].merge(
                pred[pred.source_season.eq(vs)][["game_id","elo_p"]],on="game_id",how="inner")
            if len(v)<100: continue
            ll=log_loss(v.home_win.astype(int),np.clip(v.elo_p,1e-6,1-1e-6),labels=[0,1])
            if ec is None or ll<ec[0]: ec=(ll,key)
        if ec is None: continue
        z=x.merge(elo[ec[1]][["game_id","elo_p"]],on="game_id",how="left")
        z["elo_logit"]=np.log(np.clip(z.elo_p,1e-6,1-1e-6)/np.clip(1-z.elo_p,1e-6,1-1e-6))
        train=z[z.source_season.le(target-2)&z.home_win.notna()].copy()
        val=z[z.source_season.eq(vs)&z.home_win.notna()].copy()
        test=z[z.source_season.eq(target)&z.home_win.notna()].copy()
        best=None
        for alpha,retain in ts.OA_GRID:
            tag=f"a{alpha:.2f}_r{retain:.2f}".replace(".","p")
            feats=["elo_logit","d_sog_share_r20","d_xg_share_r20","d_goals_r20","d_rest_days",
                   f"oa_sog_diff_{tag}",f"oa_xg_diff_{tag}",
                   "h_sched_b2b","a_sched_b2b","d_sched_b2b"]
            for C in C_GRID:
                pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                               ("m",LogisticRegression(C=C,max_iter=2500))])
                pipe.fit(train[feats],train.home_win.astype(int))
                pv=np.clip(pipe.predict_proba(val[feats])[:,1],1e-6,1-1e-6)
                ll=log_loss(val.home_win.astype(int),pv,labels=[0,1])
                if best is None or ll<best["ll"]: best={"ll":ll,"tag":tag,"C":C,"features":feats}
        feats=best["features"]
        bpv,bpt=fit_base(train,val,test,feats,best["C"])
        age_curve=fit_age_curve(goalies,through_season=target-2)
        chosen_params=None
        for sk in SHOT_K:
            for nk in START_K:
                for mk in MINUTE_K:
                    ratings=goalie_ratings(goalies,age_curve,sk,nk,mk)
                    zr=add_goalie_diff(z,ratings)
                    vg=zr[zr.source_season.eq(vs)&zr.home_win.notna()]
                    vp=val[["game_id","home_win"]].copy(); vp["base_p"]=bpv
                    vp=vp.merge(vg[["game_id","d_goalie_rating"]],on="game_id",how="inner")
                    beta,keep=fit_beta(vp.base_p.to_numpy(float),vp.home_win.to_numpy(float),
                                       vp.d_goalie_rating.to_numpy(float))
                    adj=apply_beta(vp.base_p.to_numpy(float),vp.d_goalie_rating.to_numpy(float),beta)
                    vll=log_loss(vp.home_win.to_numpy(int)[keep],adj[keep],labels=[0,1])
                    if chosen_params is None or vll<chosen_params["validation_log_loss"]:
                        chosen_params={"shots_k":sk,"starts_k":nk,"minutes_k":mk,"beta":beta,
                            "validation_log_loss":float(vll),"ratings":ratings}
        ch=chosen_params
        tg=add_goalie_diff(z,ch["ratings"])
        tg=tg[tg.source_season.eq(target)&tg.home_win.notna()]
        tp=test[["game_id","home_win"]].copy(); tp["base_p"]=bpt
        tp=tp.merge(tg[["game_id","d_goalie_rating"]],on="game_id",how="inner")
        adj=apply_beta(tp.base_p.to_numpy(float),tp.d_goalie_rating.to_numpy(float),ch["beta"])
        am=metrics(tp.home_win.to_numpy(int),adj); bm=metrics(tp.home_win.to_numpy(int),tp.base_p.to_numpy(float))
        row={"target_season":target,"validation_season":vs,"games":am["n"],"base_c":best["C"],
             "oa_tag":best["tag"],"shots_k":ch["shots_k"],"starts_k":ch["starts_k"],
             "minutes_k":ch["minutes_k"],"probability_beta":ch["beta"],
             "validation_log_loss":ch["validation_log_loss"],
             "backbone_brier":bm["brier"],"goalie_brier":am["brier"],
             "delta_brier":am["brier"]-bm["brier"],
             "backbone_log_loss":bm["log_loss"],"goalie_log_loss":am["log_loss"],
             "delta_log_loss":am["log_loss"]-bm["log_loss"],
             "backbone_accuracy":bm["accuracy"],"goalie_accuracy":am["accuracy"],
             "goalie_calibration":am["calibration"],"age_curve":age_curve}
        rows.append(row); selected.append(ch)
        print(json.dumps({k:v for k,v in row.items() if k!="age_curve"}),flush=True)
    if not rows: raise RuntimeError("No production-baseline goalie rolling folds produced")
    result=pd.DataFrame(rows); result.to_csv(out/"rolling_age_exposure_backtest.csv",index=False)
    n=result.games.to_numpy(float)
    bb=float(np.average(result.backbone_brier,weights=n)); gb=float(np.average(result.goalie_brier,weights=n))
    bl=float(np.average(result.backbone_log_loss,weights=n)); gl=float(np.average(result.goalie_log_loss,weights=n))
    perf={"games_scored":int(result.games.sum()),
          "accuracy":float(np.average(result.goalie_accuracy,weights=n)),
          "brier":gb,"log_loss":gl,
          "calibration":float(np.average(result.goalie_calibration,weights=n))}
    curve=fit_age_curve(goalies,through_season=max(TARGET_SEASONS))
    mode,count=Counter((v["shots_k"],v["starts_k"],v["minutes_k"]) for v in selected).most_common(1)[0]
    candidate={"schema_version":"1.0","model_id":"HOCKEY-LAB-GAME-GOALIE",
      "version":"candidate-production-0.3.0","status":"CANDIDATE",
      "feature_version":prod.FEATURE_VERSION+"+goalie-adjustment-prospective-001",
      "baseline_feature_version":prod.FEATURE_VERSION,"trained_through":"2025-26",
      "sample":{"games_scored":perf["games_scored"],"start_date":"2022 rolling-origin fold","end_date":"2026 rolling-origin fold"},
      "performance_metrics":perf,
      "calibration":{"age_curve":curve,"shrinkage":{"shots_k":mode[0],"starts_k":mode[1],"minutes_k":mode[2]},
                     "probability_beta":float(np.median([v["beta"] for v in selected]))},
      "rolling_origin":{"weighted_backbone_brier":bb,"weighted_goalie_brier":gb,"delta_brier":gb-bb,
        "weighted_backbone_log_loss":bl,"weighted_goalie_log_loss":gl,"delta_log_loss":gl-bl,
        "selected_shrinkage_folds":count,
        "folds":result.drop(columns=["age_curve"]).to_dict(orient="records")},
      "promotion_note":"Fitted against the active production baseline. Candidate requires rolling-fold review before prospective activation."}
    (out/"goalie-adjustment-candidate.json").write_text(json.dumps(candidate,indent=2))
    summary={"folds":len(result),"games":int(result.games.sum()),
      "weighted_backbone_brier":bb,"weighted_goalie_brier":gb,"delta_brier":gb-bb,
      "weighted_backbone_log_loss":bl,"weighted_goalie_log_loss":gl,"delta_log_loss":gl-bl,
      "selected_shrinkage":candidate["calibration"]["shrinkage"],
      "probability_beta_median":candidate["calibration"]["probability_beta"],
      "baseline_feature_version":prod.FEATURE_VERSION,"promotion_status":"CANDIDATE"}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--pbp",required=True)
    p.add_argument("--spine",required=True)
    p.add_argument("--goalie",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    main(a.pbp,a.spine,a.goalie,a.out)
