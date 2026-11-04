import json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BASE="https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
SOURCE_PAIRS=[
    (2010,2010),(2011,2011),(2012,2012),(2013,2013),(2014,2014),
    (2016,2015),(2017,2016),(2018,2017),(2019,2018),
    (2020,2020),(2021,2021),(2022,2022),(2023,2023),
    (2024,2024),(2025,2025),(2026,2026),(2027,2027)
]

def norm_id(s):
    return pd.to_numeric(s,errors="coerce").astype("Int64").astype("string")

def fetch_birth_dates(session, player_ids):
    """Fetch NHL birth dates once per unique goalie ID; keep misses explicit."""
    rows=[]
    ids=sorted({str(x) for x in player_ids if pd.notna(x) and str(x).strip()})
    for i,pid in enumerate(ids,1):
        url="https://api-web.nhle.com/v1/player/"+pid+"/landing"
        birth_date=None
        status="unavailable"
        for attempt in range(3):
            try:
                response=session.get(url,timeout=25)
                if response.status_code==404:
                    status="not_found"
                    break
                response.raise_for_status()
                birth_date=(response.json() or {}).get("birthDate")
                status="ok" if birth_date else "missing_birth_date"
                break
            except Exception as exc:
                status="error:"+type(exc).__name__
                if attempt<2:
                    time.sleep(0.5*(attempt+1))
        rows.append({
            "goalie_id_key":pid,
            "birth_date":birth_date,
            "profile_status":status,
            "profile_url":url
        })
        time.sleep(0.08)
        if i%100==0:
            print("PLAYER_BIO",i,"/",len(ids),flush=True)
    return pd.DataFrame(rows)

def toi_seconds(v):
    if pd.isna(v): return np.nan
    s=str(v)
    try:
        if ":" in s:
            parts=s.split(":")
            if len(parts)==2: return float(parts[0])*60+float(parts[1])
            if len(parts)==3: return float(parts[0])*3600+float(parts[1])*60+float(parts[2])
        return float(s)
    except Exception:
        return np.nan

def main(outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    session.headers.update({"User-Agent":"HockeyProjectionLab/goalie-etl-001"})
    manifest=[]; frames=[]; audits=[]

    def dl(url,path):
        path=Path(path)
        with session.get(url,stream=True,timeout=180) as r:
            r.raise_for_status()
            h=hashlib.sha256(); n=0
            with open(path,"wb") as f:
                for ch in r.iter_content(1024*1024):
                    if ch:
                        f.write(ch); h.update(ch); n+=len(ch)
        manifest.append({"url":url,"bytes":n,"sha256":h.hexdigest()})
        return path

    for season_year,pbp_year in SOURCE_PAIRS:
        pu=f"{BASE}/nhl_pbp_full/play_by_play_{pbp_year}.parquet"
        pp=dl(pu,Path("/tmp")/f"pbp_{pbp_year}.parquet")
        p=pd.read_parquet(pp); pp.unlink()

        su=f"{BASE}/nhl_schedules/nhl_schedule_{season_year}.parquet"
        sp=dl(su,Path("/tmp")/f"schedule_{season_year}.parquet")
        sched=pd.read_parquet(sp); sp.unlink()
        sched["game_id"]=pd.to_numeric(sched["game_id"],errors="coerce").astype("Int64")
        valid_ids=set(sched["game_id"].dropna().astype(int).tolist())

        gu=f"{BASE}/nhl_goalie_boxscores/goalie_box_{season_year}.parquet"
        gp=dl(gu,Path("/tmp")/f"goalie_{season_year}.parquet")
        g=pd.read_parquet(gp); gp.unlink()
        g["game_id"]=pd.to_numeric(g["game_id"],errors="coerce").astype("Int64")
        g=g[g["game_id"].isin(valid_ids)].copy()
        g["goalie_id_key"]=norm_id(g["player_id"])
        g["starter_bool"]=g["starter"].fillna(False).astype(bool)
        g["toi_seconds_box"]=g["toi"].map(toi_seconds)
        box_keep=[
            "game_id","goalie_id_key","home_away","team_abbrev","player_name",
            "starter_bool","decision","shots_against","saves","goals_against",
            "save_pctg","toi_seconds_box","game_date"
        ]
        box=g[box_keep].copy()

        p["game_id"]=pd.to_numeric(p["game_id"],errors="coerce").astype("Int64")
        p=p[p["game_id"].isin(valid_ids)].copy()
        et=p["event_type"].astype("string").str.upper() if "event_type" in p.columns else pd.Series("",index=p.index,dtype="string")
        period_type=p["period_type"].astype("string").str.upper() if "period_type" in p.columns else pd.Series("",index=p.index,dtype="string")
        non_so=period_type.ne("SHOOTOUT").fillna(True)
        empty=p["empty_net"].fillna(False).astype(bool) if "empty_net" in p.columns else pd.Series(False,index=p.index)
        team_type=p["event_team_type"].astype("string").str.lower() if "event_team_type" in p.columns else pd.Series(pd.NA,index=p.index,dtype="string")
        shooting_home=team_type.eq("home").fillna(False)
        shooting_away=team_type.eq("away").fillna(False)

        na_id=pd.Series(pd.NA,index=p.index,dtype="string")
        home_gid=norm_id(p["home_goalie_id"]) if "home_goalie_id" in p.columns else na_id.copy()
        away_gid=norm_id(p["away_goalie_id"]) if "away_goalie_id" in p.columns else na_id.copy()
        event_gid=norm_id(p["event_goalie_id"]) if "event_goalie_id" in p.columns else na_id.copy()
        defending=np.where(shooting_home,away_gid,np.where(shooting_away,home_gid,pd.NA))
        defending=pd.Series(defending,index=p.index,dtype="string")
        defending=defending.fillna(event_gid)
        p["goalie_id_key"]=defending

        is_goal=et.eq("GOAL") & non_so & ~empty
        is_sog=et.isin(["SHOT","GOAL"]) & non_so & ~empty
        is_unblocked=et.isin(["SHOT","GOAL","MISSED_SHOT"]) & non_so & ~empty
        xg=pd.to_numeric(p["xg"],errors="coerce").fillna(0.0) if "xg" in p.columns else pd.Series(0.0,index=p.index)
        fivev5=(p["strength_state"].astype("string").str.lower().eq("5v5").fillna(False)
                if "strength_state" in p.columns else pd.Series(False,index=p.index))
        high=xg.ge(.20)

        # Rebound proxy: same shooting team, unblocked attempt within 3 seconds.
        p["_unblocked"]=is_unblocked
        p["_shoot_team"]=p["event_team_abbr"].astype("string") if "event_team_abbr" in p.columns else pd.Series(pd.NA,index=p.index,dtype="string")
        p["_secs"]=pd.to_numeric(p["game_seconds"],errors="coerce") if "game_seconds" in p.columns else pd.Series(np.nan,index=p.index)
        if "event_idx" not in p.columns: p["event_idx"]=np.arange(len(p))
        p=p.sort_values(["game_id","event_idx"],kind="stable")
        prev_team=p.groupby("game_id")["_shoot_team"].shift(1)
        prev_unblocked=p.groupby("game_id")["_unblocked"].shift(1).fillna(False)
        prev_secs=p.groupby("game_id")["_secs"].shift(1)
        rebound=is_unblocked.loc[p.index] & prev_unblocked.astype(bool) & p["_shoot_team"].eq(prev_team) & (p["_secs"]-prev_secs).between(0,3)

        p["shots_against_pbp"]=is_sog.loc[p.index].astype("int16")
        p["goals_against_pbp"]=is_goal.loc[p.index].astype("int16")
        p["xg_against"]=np.where(is_unblocked.loc[p.index],xg.loc[p.index],0.0)
        p["sog_xg_against"]=np.where(is_sog.loc[p.index],xg.loc[p.index],0.0)
        p["fivev5_shots_against"]=np.where(fivev5.loc[p.index]&is_sog.loc[p.index],1,0)
        p["fivev5_goals_against"]=np.where(fivev5.loc[p.index]&is_goal.loc[p.index],1,0)
        p["fivev5_xg_against"]=np.where(fivev5.loc[p.index]&is_unblocked.loc[p.index],xg.loc[p.index],0.0)
        p["high_xg_against"]=np.where(high.loc[p.index]&is_unblocked.loc[p.index],1,0)
        p["rebound_attempts_against"]=rebound.astype("int16")
        p["rebound_xg_against"]=np.where(rebound,xg.loc[p.index],0.0)

        shot=p[p["goalie_id_key"].notna() & is_unblocked.loc[p.index]].copy()
        agg=shot.groupby(["game_id","goalie_id_key"],dropna=False).agg(
            shots_against_pbp=("shots_against_pbp","sum"),
            goals_against_pbp=("goals_against_pbp","sum"),
            xg_against=("xg_against","sum"),
            sog_xg_against=("sog_xg_against","sum"),
            fivev5_shots_against=("fivev5_shots_against","sum"),
            fivev5_goals_against=("fivev5_goals_against","sum"),
            fivev5_xg_against=("fivev5_xg_against","sum"),
            high_xg_against=("high_xg_against","sum"),
            rebound_attempts_against=("rebound_attempts_against","sum"),
            rebound_xg_against=("rebound_xg_against","sum"),
        ).reset_index()
        agg["saves_pbp"]=agg["shots_against_pbp"]-agg["goals_against_pbp"]
        agg["save_pct_pbp"]=np.where(agg["shots_against_pbp"]>0,agg["saves_pbp"]/agg["shots_against_pbp"],np.nan)
        agg["gsax"]=agg["xg_against"]-agg["goals_against_pbp"]
        agg["sog_gsax"]=agg["sog_xg_against"]-agg["goals_against_pbp"]
        agg["gsax_per_30_shots"]=np.where(agg["shots_against_pbp"]>0,agg["gsax"]/agg["shots_against_pbp"]*30,np.nan)
        agg["fivev5_gsax"]=agg["fivev5_xg_against"]-agg["fivev5_goals_against"]

        merged=box.merge(agg,on=["game_id","goalie_id_key"],how="left")
        metric_cols=[
            "shots_against_pbp","goals_against_pbp","xg_against","sog_xg_against","sog_gsax","fivev5_shots_against",
            "fivev5_goals_against","fivev5_xg_against","high_xg_against",
            "rebound_attempts_against","rebound_xg_against","saves_pbp","gsax","fivev5_gsax"
        ]
        for c in metric_cols: merged[c]=pd.to_numeric(merged[c],errors="coerce").fillna(0.0)
        merged["save_pct_pbp"]=np.where(merged["shots_against_pbp"]>0,merged["saves_pbp"]/merged["shots_against_pbp"],np.nan)
        merged["gsax_per_30_shots"]=np.where(merged["shots_against_pbp"]>0,merged["gsax"]/merged["shots_against_pbp"]*30,np.nan)
        merged["source_season"]=season_year
        merged["source_pbp_file"]=pbp_year
        merged["game_date"]=pd.to_datetime(merged["game_date"],errors="coerce")

        starter=merged[merged["starter_bool"]].copy()
        audits.append({
            "season":season_year,
            "pbp_file":pbp_year,
            "goalie_rows":len(merged),
            "games":int(merged["game_id"].nunique()),
            "starter_rows":len(starter),
            "starter_games":int(starter["game_id"].nunique()),
            "starter_with_pbp_shots":int((starter["shots_against_pbp"]>0).sum()),
            "starter_pbp_coverage":float((starter["shots_against_pbp"]>0).mean()) if len(starter) else None,
            "modern_goalie_on_ice_ids":bool("home_goalie_id" in p.columns and "away_goalie_id" in p.columns),
            "box_pbp_shots_mae":float((pd.to_numeric(starter["shots_against"],errors="coerce")-starter["shots_against_pbp"]).abs().mean()) if len(starter) else None,
        })
        frames.append(merged)
        print("SEASON",audits[-1],flush=True)

    allg=pd.concat(frames,ignore_index=True,sort=False)
    allg["goalie_id_key"]=allg["goalie_id_key"].astype("string")
    allg=allg.sort_values(["game_date","game_id","home_away","goalie_id_key"]).reset_index(drop=True)

    bios=fetch_birth_dates(session,allg["goalie_id_key"].dropna().unique())
    bios.to_csv(out/"goalie_birth_date_lookup.csv",index=False)
    allg=allg.merge(bios[["goalie_id_key","birth_date"]],on="goalie_id_key",how="left")
    allg["birth_date"]=pd.to_datetime(allg["birth_date"],errors="coerce")
    allg["game_date"]=pd.to_datetime(allg["game_date"],errors="coerce")
    allg["age_years"]=(allg["game_date"]-allg["birth_date"]).dt.days/365.2425
    allg.loc[~allg["age_years"].between(18,50),"age_years"]=np.nan

    starter_rows=allg[allg["starter_bool"].fillna(False)]
    age_coverage_by_season=starter_rows.groupby("source_season")["age_years"].apply(lambda s:float(s.notna().mean()))
    for audit in audits:
        audit["starter_age_coverage"]=age_coverage_by_season.get(audit["season"])
    dup=int(allg.duplicated(["game_id","goalie_id_key"]).sum())
    if dup:
        raise RuntimeError(f"duplicate goalie-game keys={dup}")
    allg.to_parquet(out/"goalie_game_features.parquet",index=False)
    pd.DataFrame(audits).to_csv(out/"coverage_by_season.csv",index=False)
    pd.DataFrame(manifest).to_csv(out/"source_manifest.csv",index=False)
    summary={
        "rows":len(allg),
        "games":int(allg["game_id"].nunique()),
        "starter_rows":int(allg["starter_bool"].sum()),
        "starter_games":int(allg.loc[allg["starter_bool"],"game_id"].nunique()),
        "starter_pbp_shot_coverage":float((allg.loc[allg["starter_bool"],"shots_against_pbp"]>0).mean()),
        "starter_age_coverage":float(allg.loc[allg["starter_bool"],"age_years"].notna().mean()),
        "starter_rows_missing_age":int((allg["starter_bool"] & allg["age_years"].isna()).sum()),
        "goalies_with_bio_lookup":int(bios["profile_status"].eq("ok").sum()),
        "goalies_attempted_bio_lookup":int(len(bios)),
        "duplicate_goalie_game_keys":dup,
        "seasons":sorted(int(x) for x in allg["source_season"].unique()),
        "note":"Same-game goalie metrics are descriptive. HWIN research must shift/roll them before the game being predicted. sog_gsax uses only shots-on-goal/goals and is the cross-era comparable xG-based goalie metric; full unblocked xG attribution has better modern coverage."
    }
    with open(out/"summary.json","w") as f: json.dump(summary,f,indent=2,default=str)
    print("FINAL",json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    import argparse
    q=argparse.ArgumentParser(); q.add_argument("--out",required=True)
    a=q.parse_args(); main(a.out)
