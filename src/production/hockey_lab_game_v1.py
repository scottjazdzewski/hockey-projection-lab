import json, math
from pathlib import Path
from bisect import bisect_left
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0,"research")
import hwin_team_strength_004 as ts

MODEL_ID="HOCKEY-LAB-GAME"
MODEL_VERSION="1.0.0"
FEATURE_VERSION="team-strength-004+b2b-007"
C_GRID=[0.05,0.1,0.25,0.5,1.0,2.0]

PERFORMANCE={
    "games_scored":6993,
    "accuracy":0.5991705991705991,
    "brier":0.23472630331662248,
    "log_loss":0.6615400452176158,
    "calibration":0.9978947675336999,
    "start_date":"rolling-origin 2022",
    "end_date":"rolling-origin 2026"
}

def build_schedule_features(spine):
    s=spine[["game_id","game_date","source_season","home_team_abbr","away_team_abbr"]].copy()
    s["game_date"]=pd.to_datetime(s.game_date,errors="coerce").dt.normalize()
    s=s.dropna(subset=["game_date"])
    h=s.rename(columns={"home_team_abbr":"team","away_team_abbr":"opp"}).assign(home_away="home")
    a=s.rename(columns={"away_team_abbr":"team","home_team_abbr":"opp"}).assign(home_away="away")
    long=pd.concat([h,a],ignore_index=True).sort_values(["team","game_date","game_id"])
    rows=[]
    for team,group in long.groupby("team",sort=False):
        history_dates=[]; dates_with_games=[]; locations_by_date={}
        for d,day_games in group.groupby("game_date",sort=True):
            prev_date=dates_with_games[-1] if dates_with_games else pd.NaT
            rest=(d-prev_date).days if pd.notna(prev_date) else np.nan
            prior3=len(history_dates)-bisect_left(history_dates,d-pd.Timedelta(days=3))
            prior5=len(history_dates)-bisect_left(history_dates,d-pd.Timedelta(days=5))
            prior7=len(history_dates)-bisect_left(history_dates,d-pd.Timedelta(days=7))
            last_locations=locations_by_date.get(prev_date,[]) if pd.notna(prev_date) else []
            prev_away=float(last_locations[0]=="away") if len(last_locations)==1 else np.nan
            prior_two=[locations_by_date[x] for x in dates_with_games[-2:]]
            prev2_away=float(all(x[0]=="away" for x in prior_two)) if len(prior_two)==2 and all(len(x)==1 for x in prior_two) else np.nan
            away_trip_no=1.0
            if len(day_games)!=1: away_trip_no=np.nan
            else:
                for prior_date in reversed(dates_with_games):
                    loc=locations_by_date[prior_date]
                    if len(loc)!=1: away_trip_no=np.nan; break
                    if loc[0]!="away": break
                    away_trip_no+=1
            for r in day_games.itertuples(index=False):
                current_home=r.home_away=="home"
                rows.append({
                    "game_id":r.game_id,"team":team,"home_away":r.home_away,
                    "sched_rest_days":rest,
                    "sched_b2b":float(pd.notna(rest) and rest<=1),
                    "sched_3in4":float(prior3>=2),"sched_4in6":float(prior5>=3),"sched_5in8":float(prior7>=4),
                    "sched_prior3_games":float(prior3),"sched_prior5_games":float(prior5),"sched_prior7_games":float(prior7),
                    "sched_prev_away":prev_away,
                    "sched_home_after_road":float(prev_away==1.0) if current_home and pd.notna(prev_away) else (0.0 if not current_home else np.nan),
                    "sched_home_after_2road":float(prev2_away==1.0) if current_home and pd.notna(prev2_away) else (0.0 if not current_home else np.nan),
                    "sched_away_trip_no":0.0 if current_home else away_trip_no,
                })
            dates_with_games.append(d); locations_by_date[d]=day_games.home_away.tolist(); history_dates.extend([d]*len(day_games))
    z=pd.DataFrame(rows)
    cols=[c for c in z.columns if c.startswith("sched_")]
    hh=z[z.home_away.eq("home")][["game_id"]+cols].rename(columns={c:f"h_{c}" for c in cols})
    aa=z[z.home_away.eq("away")][["game_id"]+cols].rename(columns={c:f"a_{c}" for c in cols})
    out=hh.merge(aa,on="game_id",how="inner")
    for c in cols:
        out[f"d_{c}"]=pd.to_numeric(out[f"h_{c}"],errors="coerce")-pd.to_numeric(out[f"a_{c}"],errors="coerce")
    return out

def elo_with_future(spine,k,ha,retain):
    ratings={}; prev_season=None; rows=[]
    s=spine.sort_values(["game_date","game_id"])
    for r in s.itertuples(index=False):
        season=int(r.source_season)
        if prev_season is None: prev_season=season
        elif season!=prev_season:
            ratings={t:1500+retain*(x-1500) for t,x in ratings.items()}; prev_season=season
        h=str(r.home_team_abbr); a=str(r.away_team_abbr)
        rh=ratings.get(h,1500.); ra=ratings.get(a,1500.)
        p=ts.elo_prob(rh,ra,ha); rows.append((r.game_id,season,p))
        y=getattr(r,"home_win",np.nan)
        if pd.notna(y):
            d=k*(int(bool(y))-p); ratings[h]=rh+d; ratings[a]=ra-d
    return pd.DataFrame(rows,columns=["game_id","source_season","elo_p"])

def fit_pipeline(train,features,C):
    p=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",LogisticRegression(C=C,max_iter=2500))])
    p.fit(train[features],train.home_win.astype(int))
    return p

def tune(train,val,features):
    best=None
    for C in C_GRID:
        p=fit_pipeline(train,features,C)
        pred=np.clip(p.predict_proba(val[features])[:,1],1e-6,1-1e-6)
        ll=log_loss(val.home_win.astype(int),pred,labels=[0,1])
        if best is None or ll<best[0]: best=(ll,C)
    return best[1],best[0]

def fair_ml(p):
    if not np.isfinite(p) or p<=0 or p>=1: return None
    return int(round(-100*p/(1-p))) if p>=.5 else int(round(100*(1-p)/p))

def apply_live_context(projections,context_path):
    if not context_path:
        return projections
    path=Path(context_path)
    if not path.exists():
        return projections
    try:
        ctx=json.loads(path.read_text())
    except Exception:
        return projections
    by_id={str(g.get("game_id")):g for g in ctx.get("games",[])}

    def latest(*vals):
        vals=[v for v in vals if v]
        if not vals: return None
        try:
            return max(vals,key=lambda x: pd.Timestamp(x))
        except Exception:
            return vals[-1]

    for p in projections:
        c=by_id.get(str(p.get("game_id")))
        if not c: continue
        ag=(c.get("away") or {}).get("goalie") or {}
        hg=(c.get("home") or {}).get("goalie") or {}
        goalie_statuses={ag.get("status"),hg.get("status")}
        if goalie_statuses=={"CONFIRMED"}:
            goalie_status="CONFIRMED"
        elif goalie_statuses & {"PROJECTED","CONFIRMED"}:
            goalie_status="PROJECTED"
        else:
            goalie_status="UNKNOWN"
        p["goalie_state"]={
            "status":goalie_status,
            "away":ag.get("name"),
            "home":hg.get("name"),
            "away_status":ag.get("status") or "UNKNOWN",
            "home_status":hg.get("status") or "UNKNOWN",
            "updated_at":latest(
                ag.get("source_timestamp"),hg.get("source_timestamp"),
                ag.get("observed_at"),hg.get("observed_at")
            )
        }

        al=(c.get("away") or {}).get("lineup") or {}
        hl=(c.get("home") or {}).get("lineup") or {}
        lineup_status="PROJECTED" if "PROJECTED" in {al.get("status"),hl.get("status")} else "UNKNOWN"
        p["lineup_state"]={
            "status":lineup_status,
            "away":"Projected lines" if al.get("status")=="PROJECTED" else None,
            "home":"Projected lines" if hl.get("status")=="PROJECTED" else None,
            "updated_at":latest(
                al.get("updated_at"),hl.get("updated_at"),
                al.get("observed_at"),hl.get("observed_at")
            )
        }
        flags=list(p.get("flags") or [])
        for flag in ["goalie_context_dailyfaceoff","lineup_context_dailyfaceoff"]:
            if flag not in flags: flags.append(flag)
        p["flags"]=flags
    return projections

def main(pbp_path,spine_path,schedule_json,outdir,context_path=None):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    pbp=pd.read_parquet(pbp_path)
    spine=pd.read_parquet(spine_path)
    schedule=json.loads(Path(schedule_json).read_text())
    today_games=schedule.get("games",[])
    if not today_games: raise RuntimeError("No games in today's schedule")

    spine["game_date"]=pd.to_datetime(spine.game_date,errors="coerce").dt.normalize()
    spine["source_season"]=pd.to_numeric(spine.source_season,errors="coerce")
    historical=spine[spine.result_available.fillna(False)&spine.game_type.astype("string").isin(["R","P"])].copy()

    base_games=ts.game_event_table(pbp)
    cutoff=pd.to_datetime(base_games.game_date,errors="coerce").max().date().isoformat()

    synth=[]
    for g in today_games:
        synth.append({
            "game_id":int(g["game_id"]),"source_season":2027,"game_date":pd.Timestamp(schedule["date"]),
            "home_team":g["home_abbr"],"away_team":g["away_abbr"],"home_win":np.nan,
            "h_sog":np.nan,"a_sog":np.nan,"h_xg":np.nan,"a_xg":np.nan,
            "h_sog_share":np.nan,"a_sog_share":np.nan,"h_xg_share":np.nan,"a_xg_share":np.nan,
            "h_goals":np.nan,"a_goals":np.nan
        })
    games_for_state=pd.concat([base_games,pd.DataFrame(synth)],ignore_index=True,sort=False)
    g=ts.add_flat_and_ewm_features(games_for_state)

    # Append today's schedule rows if canonical spine does not already contain them.
    current_ids=set(pd.to_numeric(spine.game_id,errors="coerce").dropna().astype(int))
    add_spine=[]
    for x in today_games:
        gid=int(x["game_id"])
        if gid not in current_ids:
            add_spine.append({
                "game_id":gid,"game_date":pd.Timestamp(schedule["date"]),"source_season":2027,
                "home_team_abbr":x["home_abbr"],"away_team_abbr":x["away_abbr"],
                "game_type":"R","home_win":np.nan,"result_available":False
            })
    spine_live=pd.concat([spine,pd.DataFrame(add_spine)],ignore_index=True,sort=False) if add_spine else spine.copy()

    sf=build_schedule_features(spine_live)
    g=g.merge(sf,on="game_id",how="left")

    # Production hyperparameters are selected only on completed 2025-26 (source season 2026).
    train_mask=g.source_season.le(2025)&g.home_win.notna()
    val_mask=g.source_season.eq(2026)&g.home_win.notna()

    # Select Elo using 2026 validation only.
    elo_choice=None; elo_candidates={}
    for k,ha,ret in ts.ELO_GRID:
        key=f"k{k}_h{ha}_r{ret}"
        e=elo_with_future(spine_live,k,ha,ret); elo_candidates[key]=e
        vv=g.loc[val_mask,["game_id","home_win"]].merge(e[["game_id","elo_p"]],on="game_id",how="inner")
        if len(vv)<100: continue
        ll=log_loss(vv.home_win.astype(int),np.clip(vv.elo_p,1e-6,1-1e-6),labels=[0,1])
        if elo_choice is None or ll<elo_choice[0]: elo_choice=(ll,key)
    if elo_choice is None: raise RuntimeError("Could not select production Elo")
    ekey=elo_choice[1]
    x=g.merge(elo_candidates[ekey][["game_id","elo_p"]],on="game_id",how="left")
    x["elo_logit"]=np.log(np.clip(x.elo_p,1e-6,1-1e-6)/np.clip(1-x.elo_p,1e-6,1-1e-6))

    # Compute all opponent-adjusted candidates over completed history + today's synthetic rows.
    tags=[]
    for alpha,ret in ts.OA_GRID:
        z,tag=ts.opponent_adjusted_features(games_for_state,alpha,ret)
        x=x.merge(z,on="game_id",how="left"); tags.append(tag)

    train=x[x.source_season.le(2025)&x.home_win.notna()].copy()
    val=x[x.source_season.eq(2026)&x.home_win.notna()].copy()

    oa_choice=None
    for tag in tags:
        base=["elo_logit","d_sog_share_r20","d_xg_share_r20","d_goals_r20","d_rest_days",
              f"oa_sog_diff_{tag}",f"oa_xg_diff_{tag}","h_sched_b2b","a_sched_b2b","d_sched_b2b"]
        C,ll=tune(train,val,base)
        if oa_choice is None or ll<oa_choice[0]: oa_choice=(ll,tag,C)
    _,tag,C=oa_choice
    features=["elo_logit","d_sog_share_r20","d_xg_share_r20","d_goals_r20","d_rest_days",
              f"oa_sog_diff_{tag}",f"oa_xg_diff_{tag}","h_sched_b2b","a_sched_b2b","d_sched_b2b"]

    final_train=x[x.source_season.le(2026)&x.home_win.notna()].copy()
    model=fit_pipeline(final_train,features,C)
    today_ids={int(g["game_id"]) for g in today_games}
    x["_game_id_int"]=pd.to_numeric(x.game_id,errors="coerce").astype("Int64")
    today=x[x["_game_id_int"].isin(today_ids)].copy()
    if len(today)!=len(today_games): raise RuntimeError(f"Today's feature rows {len(today)} != schedule games {len(today_games)}")
    probs=model.predict_proba(today[features])[:,1]
    pmap={str(int(gid)):float(p) for gid,p in zip(today["_game_id_int"],probs)}
    restmap={}
    for _,r in today.iterrows():
        gid=str(int(r["_game_id_int"]))
        away_rest=r.get("a_sched_rest_days",np.nan)
        home_rest=r.get("h_sched_rest_days",np.nan)

        def rest_label(v):
            if pd.isna(v):
                return None
            gap=int(round(float(v)))
            if gap<=1:
                return "B2B"
            days_off=max(0,gap-1)
            unit="day" if days_off==1 else "days"
            return f"{days_off} {unit} rest"

        def load_flags(prefix):
            out=[]
            if float(r.get(f"{prefix}_sched_3in4",0) or 0)>=1: out.append("3-in-4")
            if float(r.get(f"{prefix}_sched_4in6",0) or 0)>=1: out.append("4-in-6")
            if float(r.get(f"{prefix}_sched_5in8",0) or 0)>=1: out.append("5-in-8")
            return out

        def severity(v):
            if pd.isna(v): return "UNKNOWN"
            gap=int(round(float(v)))
            if gap<=1: return "B2B"
            if gap==2: return "SHORT"
            return "RESTED"

        restmap[gid]={
            "away":rest_label(away_rest),
            "home":rest_label(home_rest),
            "away_flags":load_flags("a"),
            "home_flags":load_flags("h"),
            "away_severity":severity(away_rest),
            "home_severity":severity(home_rest)
        }

    now=pd.Timestamp.now(tz="America/New_York")
    cutoff_ts=pd.Timestamp(cutoff+" 23:59:59",tz="America/New_York")
    projections=[]
    schedule_map={str(g["game_id"]):g for g in today_games}
    for gid,ginfo in schedule_map.items():
        ph=pmap[gid]; pa=1-ph
        projections.append({
            "game_id":int(gid),"game_date":schedule["date"],"start_time_et":ginfo.get("start_time_et"),
            "publish_state":"PUBLISHED",
            "away":{"abbr":ginfo["away_abbr"],"win_probability":pa,"expected_goals":None,"regulation_win_probability":None,"fair_moneyline":fair_ml(pa)},
            "home":{"abbr":ginfo["home_abbr"],"win_probability":ph,"expected_goals":None,"regulation_win_probability":None,"fair_moneyline":fair_ml(ph)},
            "overtime_probability":None,
            "goalie_state":{"status":"UNKNOWN","away":None,"home":None,"updated_at":None},
            "lineup_state":{"status":"UNKNOWN","away":None,"home":None,"updated_at":None},
            "rest_state":{
                "status":"CONFIRMED",
                "away":restmap.get(gid,{}).get("away"),
                "home":restmap.get(gid,{}).get("home"),
                "away_flags":restmap.get(gid,{}).get("away_flags",[]),
                "home_flags":restmap.get(gid,{}).get("home_flags",[]),
                "away_severity":restmap.get(gid,{}).get("away_severity","UNKNOWN"),
                "home_severity":restmap.get(gid,{}).get("home_severity","UNKNOWN"),
                "updated_at":now.isoformat()
            },
            "flags":[f"source_data_through_{cutoff}","no_goalie_layer","b2b_context"]
        })

    projections=apply_live_context(projections,context_path)

    payload={
      "schema_version":"1.0","generated_at":now.isoformat(),"data_cutoff":cutoff_ts.isoformat(),
      "model":{"model_id":MODEL_ID,"version":MODEL_VERSION,"status":"PROMOTED","feature_version":FEATURE_VERSION,"trained_through":"2025-26"},
      "projections":projections
    }
    Path(out/"game-projections.json").write_text(json.dumps(payload,indent=2)+"\n")

    performance={
      "schema_version":"1.0","generated_at":now.isoformat(),
      "model":{"model_id":MODEL_ID,"version":MODEL_VERSION,"internal_engine":"HWIN-SCHEDULE-007-NOGOALIE","feature_version":FEATURE_VERSION},
      "sample":{"games_scored":PERFORMANCE["games_scored"],"start_date":"2022 rolling-origin fold","end_date":"2026 rolling-origin fold"},
      "metrics":{"accuracy":PERFORMANCE["accuracy"],"brier":PERFORMANCE["brier"],"log_loss":PERFORMANCE["log_loss"],"calibration":PERFORMANCE["calibration"]},
      "bands":[]
    }
    Path(out/"model-performance.json").write_text(json.dumps(performance,indent=2)+"\n")
    meta={"elo_choice":ekey,"oa_choice":tag,"C":C,"features":features,"data_cutoff_date":cutoff,"today_games":len(today_games)}
    Path(out/"scorer-manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))
    print(json.dumps([{"game_id":p["game_id"],"away":p["away"]["abbr"],"away_p":p["away"]["win_probability"],"home":p["home"]["abbr"],"home_p":p["home"]["win_probability"]} for p in projections],indent=2))

if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument("--pbp",required=True); a.add_argument("--spine",required=True); a.add_argument("--schedule",required=True); a.add_argument("--out",required=True)
    a.add_argument("--context",required=False,default=None)
    z=a.parse_args(); main(z.pbp,z.spine,z.schedule,z.out,z.context)
