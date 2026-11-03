import json, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

TZ=ZoneInfo("America/New_York")
GOALIE_URL="https://www.dailyfaceoff.com/starting-goalies"
LINE_URL="https://www.dailyfaceoff.com/teams/{slug}/line-combinations"
HEADERS={"User-Agent":"HockeyProjectionLab/0.1 (+private research; public pages only)"}
ALLOWED_GROUPS={"f1","f2","f3","f4","d1","d2","d3","pp1","pp2","pk1","pk2","g"}

def fetch_next(url):
    r=requests.get(url,headers=HEADERS,timeout=45)
    r.raise_for_status()
    soup=BeautifulSoup(r.text,"html.parser")
    node=soup.find("script",id="__NEXT_DATA__")
    if not node or not node.string:
        raise RuntimeError(f"Missing __NEXT_DATA__ on {url}")
    return json.loads(node.string).get("props",{}).get("pageProps",{})

def normalize_goalie(rec,side,observed):
    prefix="home" if side=="home" else "away"
    name=rec.get(prefix+"GoalieName")
    strength=rec.get(prefix+"NewsStrengthName")
    status="UNKNOWN"
    if name:
        status="CONFIRMED" if str(strength or "").strip().lower()=="confirmed" else "PROJECTED"
    return {
      "name":name,
      "status":status,
      "observed_at":observed,
      "source_timestamp":rec.get(prefix+"NewsCreatedAt"),
      "source_name":rec.get(prefix+"NewsSourceName"),
      "source_url":rec.get(prefix+"NewsSourceUrl"),
      "player_id":rec.get(prefix+"GoalieId")
    }

def empty_lineup(observed):
    return {
      "status":"UNKNOWN","observed_at":observed,"updated_at":None,
      "source_name":None,"source_url":None,"groups":{},"injuries":[]
    }

def normalize_lineup(pp,observed,page_url):
    c=pp.get("combinations") or {}
    players=c.get("players") or []
    groups={}
    for p in players:
        gid=p.get("groupIdentifier")
        if gid not in ALLOWED_GROUPS:
            continue
        groups.setdefault(gid,[]).append({
          "name":str(p.get("name") or "").strip(),
          "position":p.get("positionIdentifier"),
          "player_id":p.get("playerId")
        })
    injuries=[]
    seen=set()
    for p in players:
        injury=p.get("injuryStatus")
        gtd=bool(p.get("gameTimeDecision"))
        if not injury and not gtd:
            continue
        key=p.get("playerId") or p.get("name")
        if key in seen: continue
        seen.add(key)
        injuries.append({
          "name":str(p.get("name") or "").strip(),
          "injury_status":injury,
          "game_time_decision":gtd,
          "player_id":p.get("playerId")
        })
    essential=all(groups.get(k) for k in ["f1","f2","f3","f4","d1","d2","d3"])
    return {
      "status":"PROJECTED" if essential else "UNKNOWN",
      "observed_at":observed,
      "updated_at":c.get("updatedAt"),
      "source_name":c.get("sourceName"),
      "source_url":c.get("source") or page_url,
      "groups":groups,
      "injuries":injuries
    },c.get("teamAbbreviation")

def aggregate_status(a,b,kind):
    vals={a,b}
    if kind=="goalie":
        if vals=={"CONFIRMED"}: return "CONFIRMED"
        if "PROJECTED" in vals or "CONFIRMED" in vals: return "PROJECTED"
        return "UNKNOWN"
    if "PROJECTED" in vals: return "PROJECTED"
    return "UNKNOWN"

def latest_stamp(*vals):
    clean=[v for v in vals if v]
    if not clean: return None
    try:
        return max(clean,key=lambda x: datetime.fromisoformat(str(x).replace("Z","+00:00")))
    except Exception:
        return clean[-1]

def update_projection_context(context_path,projection_path):
    if not projection_path.exists(): return
    ctx=json.loads(context_path.read_text())
    proj=json.loads(projection_path.read_text())
    by_id={str(g["game_id"]):g for g in ctx.get("games",[])}
    changed=False
    for p in proj.get("projections",[]):
        c=by_id.get(str(p.get("game_id")))
        if not c: continue
        ag=c["away"]["goalie"]; hg=c["home"]["goalie"]
        al=c["away"]["lineup"]; hl=c["home"]["lineup"]
        new_goalie={
          "status":aggregate_status(ag["status"],hg["status"],"goalie"),
          "away":ag.get("name"),"home":hg.get("name"),
          "updated_at":latest_stamp(ag.get("source_timestamp"),hg.get("source_timestamp"),ag.get("observed_at"),hg.get("observed_at"))
        }
        new_lineup={
          "status":aggregate_status(al["status"],hl["status"],"lineup"),
          "away":"Projected lines" if al["status"]=="PROJECTED" else None,
          "home":"Projected lines" if hl["status"]=="PROJECTED" else None,
          "updated_at":latest_stamp(al.get("updated_at"),hl.get("updated_at"),al.get("observed_at"),hl.get("observed_at"))
        }
        if p.get("goalie_state")!=new_goalie:
            p["goalie_state"]=new_goalie; changed=True
        if p.get("lineup_state")!=new_lineup:
            p["lineup_state"]=new_lineup; changed=True
        flags=list(p.get("flags") or [])
        for flag in ["goalie_context_dailyfaceoff","lineup_context_dailyfaceoff"]:
            if flag not in flags: flags.append(flag); changed=True
        p["flags"]=flags
    if changed:
        projection_path.write_text(json.dumps(proj,indent=2)+"\n")

def main():
    root=Path(".")
    schedule=json.loads((root/"public/data/today-games.json").read_text())
    date=schedule.get("date")
    games=schedule.get("games") or []
    observed=datetime.now(TZ).isoformat()
    if not games:
        raise RuntimeError("No games in today-games.json")

    goalie_pp=fetch_next(GOALIE_URL)
    goalie_rows=goalie_pp.get("data") or []
    goalie_by_match={}
    slug_by_abbr={}
    for r in goalie_rows:
        if str(r.get("date") or "")[:10] != str(date):
            continue
        aa=r.get("awayTeamAbbreviation"); ha=r.get("homeTeamAbbreviation")
        if aa and r.get("awayTeamSlug"): slug_by_abbr[aa]=r.get("awayTeamSlug")
        if ha and r.get("homeTeamSlug"): slug_by_abbr[ha]=r.get("homeTeamSlug")
        goalie_by_match[(aa,ha)]=r

    playing=sorted({g["away_abbr"] for g in games}|{g["home_abbr"] for g in games})
    lineups={}
    source_errors=[]
    for i,abbr in enumerate(playing):
        slug=slug_by_abbr.get(abbr)
        if not slug:
            source_errors.append(f"missing_slug:{abbr}")
            continue
        url=LINE_URL.format(slug=slug)
        try:
            pp=fetch_next(url)
            lineup,reported_abbr=normalize_lineup(pp,observed,url)
            if reported_abbr and reported_abbr!=abbr:
                source_errors.append(f"abbr_mismatch:{abbr}:{reported_abbr}")
                continue
            lineups[abbr]=lineup
        except Exception as e:
            source_errors.append(f"lineup_fetch:{abbr}:{type(e).__name__}")
        time.sleep(0.35)

    out_games=[]
    for g in games:
        aa=g["away_abbr"]; ha=g["home_abbr"]
        rec=goalie_by_match.get((aa,ha))
        away_goalie=normalize_goalie(rec,"away",observed) if rec else {
          "name":None,"status":"UNKNOWN","observed_at":observed,"source_timestamp":None,"source_name":None,"source_url":None,"player_id":None
        }
        home_goalie=normalize_goalie(rec,"home",observed) if rec else {
          "name":None,"status":"UNKNOWN","observed_at":observed,"source_timestamp":None,"source_name":None,"source_url":None,"player_id":None
        }
        out_games.append({
          "game_id":g["game_id"],
          "away":{"abbr":aa,"goalie":away_goalie,"lineup":lineups.get(aa,empty_lineup(observed))},
          "home":{"abbr":ha,"goalie":home_goalie,"lineup":lineups.get(ha,empty_lineup(observed))}
        })

    payload={
      "schema_version":"1.0","date":date,"generated_at":observed,
      "provider":{
        "name":"Daily Faceoff public pages",
        "access_mode":"PUBLIC_PAGE_NEXT_DATA",
        "goalie_url":GOALIE_URL,
        "lineup_url_pattern":LINE_URL,
        "robots_policy":"Public pages allowed; /api/ excluded and not used"
      },
      "games":out_games
    }
    outpath=root/"public/data/live-context.json"
    outpath.write_text(json.dumps(payload,indent=2)+"\n")
    update_projection_context(outpath,root/"public/data/game-projections.json")

    confirmed=sum(1 for g in out_games for side in ["away","home"] if g[side]["goalie"]["status"]=="CONFIRMED")
    projected=sum(1 for g in out_games for side in ["away","home"] if g[side]["goalie"]["status"]=="PROJECTED")
    lineups_ok=sum(1 for abbr in playing if lineups.get(abbr,{}).get("status")=="PROJECTED")
    print(json.dumps({
      "date":date,"games":len(out_games),"teams":len(playing),
      "goalies_confirmed":confirmed,"goalies_projected":projected,
      "lineups_projected":lineups_ok,"source_errors":source_errors
    },indent=2))

if __name__=="__main__":
    main()
