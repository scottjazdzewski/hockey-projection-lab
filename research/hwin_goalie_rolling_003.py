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
import hwin_goalie_layer_001 as gl
import hwin_rolling_backtest_001 as rb

C_GRID=[0.05,0.1,0.25,0.5,1.0,2.0]

def met(y,p):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return {"n":int(len(y)),"brier":float(brier_score_loss(y,p)),
            "log_loss":float(log_loss(y,p,labels=[0,1])),
            "accuracy":float(accuracy_score(y,p>=.5))}

def fit_tuned(train,val,test,features):
    best=None
    for C in C_GRID:
        pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                       ("m",LogisticRegression(C=C,max_iter=1800))])
        pipe.fit(train[features],train.home_win)
        pv=pipe.predict_proba(val[features])[:,1]
        ll=log_loss(val.home_win,np.clip(pv,1e-6,1-1e-6),labels=[0,1])
        if best is None or ll<best[0]: best=(ll,C)
    C=best[1]
    fit=pd.concat([train,val],ignore_index=True)
    pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                   ("m",LogisticRegression(C=C,max_iter=1800))])
    pipe.fit(fit[features],fit.home_win)
    p=pipe.predict_proba(test[features])[:,1]
    return C,met(test.home_win,p)

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

    x=gl.build_team_features(pbp)
    x=merge_goalies(x,goalie)

    # Precompute Elo candidates.
    elo={}
    for k,ha,r in rb.ELO_GRID:
        key=f"k{k}_h{ha}_r{r}"
        elo[key]=rb.elo_all(spine,k,ha,r)

    rows=[]
    for target in [2022,2023,2024,2025,2026]:
        val_season=target-1
        # Select Elo only on the preceding season.
        choice=None
        for key,pred in elo.items():
            v=x[x.source_season.eq(val_season)][["game_id","home_win"]].merge(
                pred[pred.source_season.eq(val_season)][["game_id","p"]],on="game_id")
            if len(v)<100: continue
            ll=log_loss(v.home_win,np.clip(v.p,1e-6,1-1e-6),labels=[0,1])
            if choice is None or ll<choice[0]: choice=(ll,key)
        key=choice[1]
        ep=elo[key][["game_id","p"]].rename(columns={"p":"elo_p"})
        z=x.merge(ep,on="game_id",how="left")
        z["elo_logit"]=np.log(np.clip(z.elo_p,1e-6,1-1e-6)/np.clip(1-z.elo_p,1e-6,1-1e-6))

        team=["elo_logit","d_fivev5_sog_share_r20","d_fivev5_xg_share_r20","d_goals_non_en_r20","d_rest_days"]
        specs={
            "backbone":team,
            "plus_sv20":team+["d_g_sv_r20"],
            "plus_sog_gsax20":team+["d_g_sog_gsax30_r20"],
            "plus_sv20_gsax20":team+["d_g_sv_r20","d_g_sog_gsax30_r20"],
            "plus_shrunk_sv":team+["d_g_sv_shrunk_prior"],
        }
        train=z[z.source_season.le(target-2)]
        val=z[z.source_season.eq(val_season)]
        test=z[z.source_season.eq(target)]
        for name,features in specs.items():
            C,mm=fit_tuned(train,val,test,features)
            rows.append({"target_season":target,"model":name,"elo_choice":key,"C":C,**mm})

    res=pd.DataFrame(rows)
    res.to_csv(out/"rolling_goalie_backtest.csv",index=False)
    summary={}
    for model,g in res.groupby("model"):
        summary[model]={
            "folds":int(len(g)),
            "games":int(g.n.sum()),
            "weighted_brier":float(np.average(g.brier,weights=g.n)),
            "weighted_log_loss":float(np.average(g.log_loss,weights=g.n)),
            "weighted_accuracy":float(np.average(g.accuracy,weights=g.n)),
            "brier_by_season":{str(int(r.target_season)):float(r.brier) for r in g.itertuples()}
        }
    base=summary["backbone"]["weighted_brier"]
    for model in summary:
        summary[model]["delta_brier_vs_backbone"]=summary[model]["weighted_brier"]-base
    with open(out/"summary.json","w") as f: json.dump(summary,f,indent=2)
    print(res.to_string(index=False))
    print("\nSUMMARY")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--pbp",required=True); p.add_argument("--spine",required=True); p.add_argument("--goalie",required=True); p.add_argument("--out",required=True)
    a=p.parse_args(); main(a.pbp,a.spine,a.goalie,a.out)
