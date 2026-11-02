import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EVENTS=["fivev5_sog_share","fivev5_xg_share","goals_non_en"]
C_GRID=[0.05,0.1,0.25,0.5,1.0,2.0]
ELO_GRID=[(5,20,.7),(10,20,.7),(10,40,.7),(5,40,.7),(10,20,.55),(10,40,.55),(10,20,.85)]

def met(y,p):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return {"n":int(len(y)),"brier":float(brier_score_loss(y,p)),
            "log_loss":float(log_loss(y,p,labels=[0,1])),
            "accuracy":float(accuracy_score(y,p>=.5)),
            "mean_pred":float(p.mean()),"actual_rate":float(y.mean())}

def ep(rh,ra,ha):
    return 1/(1+10**(-(rh+ha-ra)/400))

def elo_all(g,k,ha,retain):
    ratings={}; prev=None; rows=[]
    for r in g.sort_values(["game_date","game_id"]).itertuples(index=False):
        s=int(r.source_season)
        if prev is None: prev=s
        elif s!=prev:
            ratings={t:1500+retain*(v-1500) for t,v in ratings.items()}
            prev=s
        h=str(r.home_team_abbr); a=str(r.away_team_abbr)
        rh=ratings.get(h,1500.); ra=ratings.get(a,1500.)
        p=ep(rh,ra,ha)
        y=int(bool(r.home_win))
        rows.append((r.game_id,s,p))
        d=k*(y-p); ratings[h]=rh+d; ratings[a]=ra-d
    return pd.DataFrame(rows,columns=["game_id","source_season","p"])

def event_table(pbp):
    d=pbp.copy()
    d["game_date"]=pd.to_datetime(d["game_date"],errors="coerce")
    d["source_season"]=pd.to_numeric(d["source_season"],errors="coerce")
    d=d[d["game_type"].astype("string").isin(["R","P"])].copy()
    d=d.sort_values(["event_team_abbr","game_date","game_id"])
    prev=d.groupby("event_team_abbr")["game_date"].shift(1)
    gap=(d["game_date"]-prev).dt.total_seconds()/86400
    seg=gap.gt(240).fillna(True).groupby(d["event_team_abbr"]).cumsum()
    keys=[d["event_team_abbr"],seg]
    for c in EVENTS:
        vals=pd.to_numeric(d[c],errors="coerce")
        d[f"{c}_r20"]=vals.groupby(keys,sort=False).transform(lambda s:s.shift(1).rolling(20,min_periods=8).mean())
    d["rest_days"]=gap.clip(0,14)
    rc=[f"{c}_r20" for c in EVENTS]+["rest_days"]
    h=d[d.home_away.eq("home")][["game_id","source_season","game_date","win"]+rc].copy()
    a=d[d.home_away.eq("away")][["game_id"]+rc].copy()
    h=h.rename(columns={"win":"home_win",**{c:f"h_{c}" for c in rc}})
    a=a.rename(columns={c:f"a_{c}" for c in rc})
    x=h.merge(a,on="game_id",how="inner")
    for c in rc:
        x[f"d_{c}"]=pd.to_numeric(x[f"h_{c}"],errors="coerce")-pd.to_numeric(x[f"a_{c}"],errors="coerce")
    return x

def fit_predict(train,val,test,features):
    best=None
    for C in C_GRID:
        pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                       ("m",LogisticRegression(C=C,max_iter=1500))])
        pipe.fit(train[features],train.home_win)
        pv=pipe.predict_proba(val[features])[:,1]
        score=log_loss(val.home_win,np.clip(pv,1e-6,1-1e-6),labels=[0,1])
        if best is None or score<best[0]: best=(score,C)
    C=best[1]
    fit=pd.concat([train,val],ignore_index=True)
    pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                   ("m",LogisticRegression(C=C,max_iter=1500))])
    pipe.fit(fit[features],fit.home_win)
    pt=pipe.predict_proba(test[features])[:,1]
    return C,met(test.home_win,pt),pt

def main(pbp_path,spine_path,outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    pbp=pd.read_parquet(pbp_path)
    spine=pd.read_parquet(spine_path)
    spine=spine[spine.result_available.fillna(False)&spine.game_type.astype("string").isin(["R","P"])].copy()
    spine["game_date"]=pd.to_datetime(spine.game_date,errors="coerce")
    spine["source_season"]=pd.to_numeric(spine.source_season,errors="coerce")
    x=event_table(pbp)
    backbone=["d_fivev5_sog_share_r20","d_fivev5_xg_share_r20","d_goals_non_en_r20","d_rest_days"]

    # Precompute Elo candidates.
    elo={}
    for k,ha,r in ELO_GRID:
        key=f"k{k}_h{ha}_r{r}"
        elo[key]=elo_all(spine,k,ha,r)
    rows=[]
    for target in [2022,2023,2024,2025,2026]:
        val_season=target-1
        train=x[x.source_season.le(target-2)].copy()
        val=x[x.source_season.eq(val_season)].copy()
        test=x[x.source_season.eq(target)].copy()
        if min(len(train),len(val),len(test))==0: continue

        # Elo tune strictly on prior season, evaluate target season.
        elo_choice=None
        for key,pred in elo.items():
            vv=val[["game_id","home_win"]].merge(pred[pred.source_season.eq(val_season)][["game_id","p"]],on="game_id")
            if len(vv)<100: continue
            ll=log_loss(vv.home_win,np.clip(vv.p,1e-6,1-1e-6),labels=[0,1])
            if elo_choice is None or ll<elo_choice[0]: elo_choice=(ll,key)
        key=elo_choice[1]
        epred=elo[key]
        tt=test[["game_id","home_win"]].merge(epred[epred.source_season.eq(target)][["game_id","p"]],on="game_id")
        em=met(tt.home_win,tt.p)
        rows.append({"target_season":target,"model":"elo","tuned_on":val_season,"choice":key,**em})

        # Event backbone.
        C,mm,pp=fit_predict(train,val,test,backbone)
        rows.append({"target_season":target,"model":"event_backbone","tuned_on":val_season,"choice":f"C={C}",**mm})

        # Fusion uses the selected Elo candidate as a feature; tune logistic C on prior season.
        ex=epred[["game_id","p"]].rename(columns={"p":"elo_p"})
        allx=x.merge(ex,on="game_id",how="left")
        allx["elo_logit"]=np.log(np.clip(allx.elo_p,1e-6,1-1e-6)/np.clip(1-allx.elo_p,1e-6,1-1e-6))
        tr=allx[allx.source_season.le(target-2)]
        va=allx[allx.source_season.eq(val_season)]
        te=allx[allx.source_season.eq(target)]
        C,fm,fp=fit_predict(tr,va,te,["elo_logit"]+backbone)
        rows.append({"target_season":target,"model":"fusion_backbone","tuned_on":val_season,"choice":f"{key};C={C}",**fm})

        # Constant home baseline from training history only.
        rate=float(spine[spine.source_season.le(target-1)].home_win.astype(float).mean())
        rows.append({"target_season":target,"model":"constant_home","tuned_on":val_season,"choice":f"p={rate:.6f}",**met(test.home_win,np.repeat(rate,len(test)))})

    res=pd.DataFrame(rows)
    res.to_csv(out/"rolling_backtest.csv",index=False)
    summary={}
    for model,g in res.groupby("model"):
        summary[model]={
            "folds":int(len(g)),
            "games":int(g.n.sum()),
            "mean_brier":float(g.brier.mean()),
            "weighted_brier":float(np.average(g.brier,weights=g.n)),
            "mean_log_loss":float(g.log_loss.mean()),
            "weighted_log_loss":float(np.average(g.log_loss,weights=g.n)),
            "weighted_accuracy":float(np.average(g.accuracy,weights=g.n)),
            "brier_by_season":{str(int(r.target_season)):float(r.brier) for r in g.itertuples()}
        }
    with open(out/"summary.json","w") as f: json.dump(summary,f,indent=2)
    print(res.to_string(index=False))
    print("\nSUMMARY")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--pbp",required=True); p.add_argument("--spine",required=True); p.add_argument("--out",required=True)
    a=p.parse_args(); main(a.pbp,a.spine,a.out)
