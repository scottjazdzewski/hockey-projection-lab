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
    return {"n":int(len(y)),"brier":float(brier_score_loss(y,p)),
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
    cols=["sv_shrunk_prior"]
    base=["game_id","home_away"]+cols
    h=gp[gp.home_away.astype("string").str.lower().eq("home")][base].drop_duplicates("game_id")
    a=gp[gp.home_away.astype("string").str.lower().eq("away")][base].drop_duplicates("game_id")
    h=h.rename(columns={"sv_shrunk_prior":"h_sv_shrunk_prior"})
    a=a.rename(columns={"sv_shrunk_prior":"a_sv_shrunk_prior"})
    x=x.merge(h.drop(columns=["home_away"]),on="game_id",how="left")
    x=x.merge(a.drop(columns=["home_away"]),on="game_id",how="left")
    x["d_g_sv_shrunk_prior"]=pd.to_numeric(x.h_sv_shrunk_prior,errors="coerce")-pd.to_numeric(x.a_sv_shrunk_prior,errors="coerce")
    return x

def schedule_features(spine):
    s=spine[["game_id","game_date","source_season","home_team_abbr","away_team_abbr"]].copy()
    s["game_date"]=pd.to_datetime(s.game_date,errors="coerce")
    h=s.rename(columns={"home_team_abbr":"team","away_team_abbr":"opp"}).assign(home_away="home")
    a=s.rename(columns={"away_team_abbr":"team","home_team_abbr":"opp"}).assign(home_away="away")
    long=pd.concat([h,a],ignore_index=True).sort_values(["team","game_date","game_id"]).reset_index(drop=True)

    rows=[]
    for team,g in long.groupby("team",sort=False):
        g=g.sort_values(["game_date","game_id"])
        hist_dates=[]
        hist_locs=[]
        away_streak=0
        for r in g.itertuples(index=False):
            d=r.game_date
            prev_date=hist_dates[-1] if hist_dates else pd.NaT
            rest=(d-prev_date).days if hist_dates and pd.notna(d) and pd.notna(prev_date) else np.nan
            prior3=sum(1 for x in hist_dates if pd.notna(d) and (d-x).days>=1 and (d-x).days<=3)
            prior5=sum(1 for x in hist_dates if pd.notna(d) and (d-x).days>=1 and (d-x).days<=5)
            prior7=sum(1 for x in hist_dates if pd.notna(d) and (d-x).days>=1 and (d-x).days<=7)
            prev_away=(hist_locs[-1]=="away") if hist_locs else False
            prev2_away=(len(hist_locs)>=2 and hist_locs[-1]=="away" and hist_locs[-2]=="away")
            current_home=(r.home_away=="home")
            current_away=not current_home
            if current_away:
                trip_no=away_streak+1
            else:
                trip_no=0
            rows.append({
                "game_id":r.game_id,"team":team,"home_away":r.home_away,
                "sched_rest_days":rest,
                "sched_b2b":float(pd.notna(rest) and rest<=1),
                "sched_3in4":float(prior3>=2),
                "sched_4in6":float(prior5>=3),
                "sched_5in8":float(prior7>=4),
                "sched_prior3_games":float(prior3),
                "sched_prior5_games":float(prior5),
                "sched_prior7_games":float(prior7),
                "sched_prev_away":float(prev_away),
                "sched_home_after_road":float(current_home and prev_away),
                "sched_home_after_2road":float(current_home and prev2_away),
                "sched_away_trip_no":float(trip_no),
            })
            hist_dates.append(d); hist_locs.append(r.home_away)
            away_streak = away_streak+1 if current_away else 0
    z=pd.DataFrame(rows)
    cols=[c for c in z.columns if c.startswith("sched_")]
    hh=z[z.home_away.eq("home")][["game_id"]+cols].rename(columns={c:f"h_{c}" for c in cols})
    aa=z[z.home_away.eq("away")][["game_id"]+cols].rename(columns={c:f"a_{c}" for c in cols})
    out=hh.merge(aa,on="game_id",how="inner")
    for c in cols:
        out[f"d_{c}"]=pd.to_numeric(out[f"h_{c}"],errors="coerce")-pd.to_numeric(out[f"a_{c}"],errors="coerce")
    return out,cols

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
    sf,sched_cols=schedule_features(spine)
    g=g.merge(sf,on="game_id",how="left")

    elo={}
    for k,ha,ret in ts.ELO_GRID:
        key=f"k{k}_h{ha}_r{ret}"; elo[key]=ts.elo_all(spine,k,ha,ret)
    for alpha,ret in ts.OA_GRID:
        z,tag=ts.opponent_adjusted_features(g,alpha,ret); g=g.merge(z,on="game_id",how="left")

    rows=[]
    for target in [2022,2023,2024,2025,2026]:
        val_season=target-1
        ec=None
        for key,pred in elo.items():
            vv=g[g.source_season.eq(val_season)][["game_id","home_win"]].merge(
                pred[pred.source_season.eq(val_season)][["game_id","elo_p"]],on="game_id")
            if len(vv)<100: continue
            ll=log_loss(vv.home_win,np.clip(vv.elo_p,1e-6,1-1e-6),labels=[0,1])
            if ec is None or ll<ec[0]: ec=(ll,key)
        ekey=ec[1]
        x=g.merge(elo[ekey][["game_id","elo_p"]],on="game_id",how="left")
        x["elo_logit"]=np.log(np.clip(x.elo_p,1e-6,1-1e-6)/np.clip(1-x.elo_p,1e-6,1-1e-6))
        train=x[x.source_season.le(target-2)]; val=x[x.source_season.eq(val_season)]; test=x[x.source_season.eq(target)]

        oc=None
        for alpha,ret in ts.OA_GRID:
            tag=f"a{alpha:.2f}_r{ret:.2f}".replace(".","p")
            feats=["elo_logit","d_sog_share_r20","d_xg_share_r20","d_goals_r20","d_rest_days",
                   f"oa_sog_diff_{tag}",f"oa_xg_diff_{tag}","d_g_sv_shrunk_prior"]
            for C0 in C_GRID:
                p=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                            ("m",LogisticRegression(C=C0,max_iter=2000))])
                p.fit(train[feats],train.home_win)
                pv=p.predict_proba(val[feats])[:,1]
                ll=log_loss(val.home_win,np.clip(pv,1e-6,1-1e-6),labels=[0,1])
                if oc is None or ll<oc[0]: oc=(ll,tag)
        tag=oc[1]
        base=["elo_logit","d_sog_share_r20","d_xg_share_r20","d_goals_r20","d_rest_days",
              f"oa_sog_diff_{tag}",f"oa_xg_diff_{tag}","d_g_sv_shrunk_prior"]

        density=["d_sched_3in4","d_sched_4in6","d_sched_5in8",
                 "d_sched_prior3_games","d_sched_prior5_games","d_sched_prior7_games"]
        location=["h_sched_home_after_road","h_sched_home_after_2road",
                  "a_sched_prev_away","a_sched_away_trip_no"]
        b2b=["h_sched_b2b","a_sched_b2b","d_sched_b2b"]
        specs={
            "fusion005_base":base,
            "plus_b2b":base+b2b,
            "plus_density":base+density,
            "plus_location_sequence":base+location,
            "plus_schedule_all":base+b2b+density+location,
        }
        for name,features in specs.items():
            C,mm=fit_tuned(train,val,test,features)
            rows.append({"target_season":target,"model":name,"elo_choice":ekey,"oa_choice":tag,"C":C,**mm})

    res=pd.DataFrame(rows); res.to_csv(out/"rolling_schedule_006.csv",index=False)
    summary={}
    for model,z in res.groupby("model"):
        summary[model]={
            "games":int(z.n.sum()),
            "weighted_brier":float(np.average(z.brier,weights=z.n)),
            "weighted_log_loss":float(np.average(z.log_loss,weights=z.n)),
            "weighted_accuracy":float(np.average(z.accuracy,weights=z.n)),
            "brier_by_season":{str(int(r.target_season)):float(r.brier) for r in z.itertuples()}
        }
    baseb=summary["fusion005_base"]["weighted_brier"]
    for m in summary:
        summary[m]["delta_brier_vs_fusion005"]=summary[m]["weighted_brier"]-baseb
        summary[m]["seasons_better_than_base"]=0 if m=="fusion005_base" else int(sum(
            summary[m]["brier_by_season"][s]<summary["fusion005_base"]["brier_by_season"][s]
            for s in summary["fusion005_base"]["brier_by_season"]))
    with open(out/"summary.json","w") as f: json.dump(summary,f,indent=2)
    print(res.to_string(index=False)); print("\nSUMMARY"); print(json.dumps(summary,indent=2))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--pbp",required=True); p.add_argument("--spine",required=True); p.add_argument("--goalie",required=True); p.add_argument("--out",required=True)
    a=p.parse_args(); main(a.pbp,a.spine,a.goalie,a.out)
