import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0,"research")
import hwin_team_strength_004 as ts
import hwin_goalie_layer_001 as gl

C_GRID=[0.05,0.1,0.25,0.5,1.0,2.0]

def met(y,p):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return {"n":int(len(y)),
            "brier":float(brier_score_loss(y,p)),
            "log_loss":float(log_loss(y,p,labels=[0,1])),
            "accuracy":float(accuracy_score(y,p>=.5))}

def fit_tuned(train,val,test,features):
    best=None
    for C in C_GRID:
        pipe=Pipeline([("imp",SimpleImputer(strategy="median")),
                       ("sc",StandardScaler()),
                       ("m",LogisticRegression(C=C,max_iter=2000))])
        pipe.fit(train[features],train.home_win)
        pv=pipe.predict_proba(val[features])[:,1]
        ll=log_loss(val.home_win,np.clip(pv,1e-6,1-1e-6),labels=[0,1])
        if best is None or ll<best[0]: best=(ll,C)
    C=best[1]
    fit=pd.concat([train,val],ignore_index=True)
    pipe=Pipeline([("imp",SimpleImputer(strategy="median")),
                   ("sc",StandardScaler()),
                   ("m",LogisticRegression(C=C,max_iter=2000))])
    pipe.fit(fit[features],fit.home_win)
    pt=pipe.predict_proba(test[features])[:,1]
    return C,met(test.home_win,pt)

def merge_goalies(x,goalies):
    gp=gl.build_goalie_priors(goalies)
    cols=["sv_r20","sog_gsax30_r20","sv_shrunk_prior"]
    base=["game_id","home_away"]+cols
    h=gp[gp.home_away.astype("string").str.lower().eq("home")][base].drop_duplicates("game_id")
    a=gp[gp.home_away.astype("string").str.lower().eq("away")][base].drop_duplicates("game_id")
    h=h.rename(columns={c:f"h_{c}" for c in cols})
    a=a.rename(columns={c:f"a_{c}" for c in cols})
    x=x.merge(h.drop(columns=["home_away"]),on="game_id",how="left")
    x=x.merge(a.drop(columns=["home_away"]),on="game_id",how="left")
    for c in cols:
        x[f"d_g_{c}"]=pd.to_numeric(x[f"h_{c}"],errors="coerce")-pd.to_numeric(x[f"a_{c}"],errors="coerce")
    return x

def main(pbp_path,spine_path,goalie_path,outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    pbp=pd.read_parquet(pbp_path)
    spine=pd.read_parquet(spine_path)
    goalie=pd.read_parquet(goalie_path)

    spine=spine[spine.result_available.fillna(False)&spine.game_type.astype("string").isin(["R","P"])].copy()
    spine["game_date"]=pd.to_datetime(spine.game_date,errors="coerce")
    spine["source_season"]=pd.to_numeric(spine.source_season,errors="coerce")

    g=ts.add_flat_and_ewm_features(ts.game_event_table(pbp))
    g=merge_goalies(g,goalie)

    # Precompute leakage-safe Elo.
    elo={}
    for k,ha,ret in ts.ELO_GRID:
        key=f"k{k}_h{ha}_r{ret}"
        elo[key]=ts.elo_all(spine,k,ha,ret)

    # Precompute opponent-adjusted candidates.
    for alpha,ret in ts.OA_GRID:
        z,tag=ts.opponent_adjusted_features(g,alpha,ret)
        g=g.merge(z,on="game_id",how="left")

    rows=[]
    feature_choices={}

    for target in [2022,2023,2024,2025,2026]:
        val_season=target-1

        # Elo choice selected on prior validation season only.
        elo_choice=None
        for key,pred in elo.items():
            vv=g[g.source_season.eq(val_season)][["game_id","home_win"]].merge(
                pred[pred.source_season.eq(val_season)][["game_id","elo_p"]],on="game_id")
            if len(vv)<100: continue
            ll=log_loss(vv.home_win,np.clip(vv.elo_p,1e-6,1-1e-6),labels=[0,1])
            if elo_choice is None or ll<elo_choice[0]: elo_choice=(ll,key)
        ekey=elo_choice[1]
        x=g.merge(elo[ekey][["game_id","elo_p"]],on="game_id",how="left")
        x["elo_logit"]=np.log(np.clip(x.elo_p,1e-6,1-1e-6)/np.clip(1-x.elo_p,1e-6,1-1e-6))

        train=x[x.source_season.le(target-2)]
        val=x[x.source_season.eq(val_season)]
        test=x[x.source_season.eq(target)]

        # Choose opponent-adjusted settings using validation only, with the proven team backbone.
        oa_choice=None
        for alpha,ret in ts.OA_GRID:
            tag=f"a{alpha:.2f}_r{ret:.2f}".replace(".","p")
            feats=["elo_logit","d_sog_share_r20","d_xg_share_r20","d_goals_r20","d_rest_days",
                   f"oa_sog_diff_{tag}",f"oa_xg_diff_{tag}"]
            for C0 in C_GRID:
                pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                               ("m",LogisticRegression(C=C0,max_iter=2000))])
                pipe.fit(train[feats],train.home_win)
                pv=pipe.predict_proba(val[feats])[:,1]
                ll=log_loss(val.home_win,np.clip(pv,1e-6,1-1e-6),labels=[0,1])
                if oa_choice is None or ll<oa_choice[0]: oa_choice=(ll,tag,C0)
        _,tag,_=oa_choice

        team_best=["elo_logit","d_sog_share_r20","d_xg_share_r20","d_goals_r20","d_rest_days",
                   f"oa_sog_diff_{tag}",f"oa_xg_diff_{tag}"]

        specs={
            "team_best":team_best,
            "team_plus_goalie_shrunk_sv":team_best+["d_g_sv_shrunk_prior"],
            "team_plus_goalie_sv20":team_best+["d_g_sv_r20"],
            "team_plus_goalie_gsax20":team_best+["d_g_sog_gsax30_r20"],
            "team_plus_goalie_sv_gsax":team_best+["d_g_sv_r20","d_g_sog_gsax30_r20"],
            "team_plus_goalie_all":team_best+["d_g_sv_shrunk_prior","d_g_sv_r20","d_g_sog_gsax30_r20"],
        }
        feature_choices[str(target)]={"elo":ekey,"opponent_adjustment":tag}

        for name,features in specs.items():
            C,mm=fit_tuned(train,val,test,features)
            rows.append({"target_season":target,"model":name,"elo_choice":ekey,"oa_choice":tag,"C":C,**mm})

    res=pd.DataFrame(rows)
    res.to_csv(out/"rolling_fusion_005.csv",index=False)

    summary={}
    for model,z in res.groupby("model"):
        summary[model]={
            "folds":int(len(z)),
            "games":int(z.n.sum()),
            "weighted_brier":float(np.average(z.brier,weights=z.n)),
            "weighted_log_loss":float(np.average(z.log_loss,weights=z.n)),
            "weighted_accuracy":float(np.average(z.accuracy,weights=z.n)),
            "brier_by_season":{str(int(r.target_season)):float(r.brier) for r in z.itertuples()},
            "log_loss_by_season":{str(int(r.target_season)):float(r.log_loss) for r in z.itertuples()},
            "accuracy_by_season":{str(int(r.target_season)):float(r.accuracy) for r in z.itertuples()},
        }
    base=summary["team_best"]["weighted_brier"]
    for m in summary:
        summary[m]["delta_brier_vs_team_best"]=summary[m]["weighted_brier"]-base
        summary[m]["seasons_better_than_team_best"]=0 if m=="team_best" else int(sum(
            summary[m]["brier_by_season"][s] < summary["team_best"]["brier_by_season"][s]
            for s in summary["team_best"]["brier_by_season"]
        ))

    with open(out/"summary.json","w") as f:
        json.dump({"summary":summary,"feature_choices":feature_choices},f,indent=2)

    print(res.to_string(index=False))
    print("\nSUMMARY")
    print(json.dumps({"summary":summary,"feature_choices":feature_choices},indent=2))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--pbp",required=True)
    p.add_argument("--spine",required=True)
    p.add_argument("--goalie",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    main(a.pbp,a.spine,a.goalie,a.out)
