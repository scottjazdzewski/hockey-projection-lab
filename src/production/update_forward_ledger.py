#!/usr/bin/env python3
"""Freeze and settle Hockey Lab forward-test game records.

Forecast blocks are immutable once written. Result/evaluation blocks may be
filled after a game becomes final. This keeps forward testing free of
post-puck-drop information leakage.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT=Path(__file__).resolve().parents[2]
TZ=ZoneInfo("America/New_York")
GAMES_DIR=ROOT/"artifacts"/"forward"/"games"
SUMMARY_PATH=ROOT/"public"/"data"/"forward-ledger-summary.json"
SCHEDULE_PATH=ROOT/"public"/"data"/"today-games.json"
PROJECTIONS_PATH=ROOT/"public"/"data"/"game-projections.json"
CONTEXT_PATH=ROOT/"public"/"data"/"live-context.json"
FINAL_STATES={"FINAL","OFF"}

def read_json(path: Path, default=None):
    if not path.exists():
        return deepcopy(default)
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False)+"\n", encoding="utf-8")

def parse_dt(value):
    if not value:
        return None
    try:
        dt=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt=dt.replace(tzinfo=TZ)
    return dt.astimezone(TZ)

def now_et():
    override=os.environ.get("HOCKEY_LAB_NOW")
    parsed=parse_dt(override) if override else None
    return parsed or datetime.now(TZ)

def safe_num(v):
    return v if isinstance(v,(int,float)) and math.isfinite(v) else None

def projected_game_map():
    payload=read_json(PROJECTIONS_PATH,{}) or {}
    return payload, {str(x.get("game_id")):x for x in payload.get("projections",[])}

def context_game_map():
    payload=read_json(CONTEXT_PATH,{}) or {}
    return payload, {str(x.get("game_id")):x for x in payload.get("games",[])}

def side_context(ctx, side):
    team=(ctx or {}).get(side) or {}
    return {
        "abbr": team.get("abbr"),
        "goalie": deepcopy(team.get("goalie") or {
            "status":"UNKNOWN","name":None,"source_timestamp":None,"source_url":None
        }),
        "lineup": deepcopy(team.get("lineup") or {"status":"UNKNOWN"})
    }

def freeze_records(min_before: float, max_before: float):
    schedule=read_json(SCHEDULE_PATH,{}) or {}
    projection_payload, projections=projected_game_map()
    context_payload, contexts=context_game_map()
    now=now_et()
    created=[]

    if projection_payload.get("model",{}).get("status")!="PROMOTED":
        print("No freeze: production model is not PROMOTED")
        return created

    for game in schedule.get("games",[]):
        gid=str(game.get("game_id"))
        start_raw=game.get("start_time_et")
        if not gid or gid=="None" or not start_raw:
            continue
        try:
            start=datetime.fromisoformat(start_raw).astimezone(TZ)
        except Exception:
            continue
        minutes=(start-now).total_seconds()/60
        if minutes < min_before or minutes > max_before:
            continue

        out=GAMES_DIR/f"{gid}.json"
        if out.exists():
            continue

        p=projections.get(gid)
        if not p or p.get("publish_state")!="PUBLISHED":
            print(f"SKIP {gid}: no PUBLISHED projection")
            continue

        ctx=contexts.get(gid) or {}
        away=game.get("away_abbr")
        home=game.get("home_abbr")
        if p.get("away",{}).get("abbr")!=away or p.get("home",{}).get("abbr")!=home:
            raise RuntimeError(f"{gid}: projection matchup mismatch")

        model=projection_payload.get("model") or {}
        generated=parse_dt(projection_payload.get("generated_at"))
        cutoff=parse_dt(projection_payload.get("data_cutoff"))
        context_generated=parse_dt(context_payload.get("generated_at"))
        if not generated or generated>start:
            print(f"SKIP {gid}: projection timestamp is not pregame")
            continue
        if not cutoff or cutoff>generated or cutoff>start:
            print(f"SKIP {gid}: data cutoff is not valid pregame evidence")
            continue
        if context_generated and context_generated>start:
            print(f"SKIP {gid}: live context timestamp is after puck drop")
            continue

        record={
            "schema_version":"1.0",
            "game_id":game.get("game_id"),
            "game_date":schedule.get("date") or start.date().isoformat(),
            "start_time_et":start.isoformat(),
            "matchup":{
                "away_abbr":away,
                "away_name":game.get("away_name"),
                "home_abbr":home,
                "home_name":game.get("home_name")
            },
            "frozen_at":now.isoformat(),
            "minutes_before_start":round(minutes,2),
            "forecast":{
                "model":{
                    "model_id":model.get("model_id"),
                    "version":model.get("version"),
                    "feature_version":model.get("feature_version"),
                    "status":model.get("status")
                },
                "generated_at":projection_payload.get("generated_at"),
                "data_cutoff":projection_payload.get("data_cutoff"),
                "context_generated_at":context_payload.get("generated_at"),
                "away":{
                    "abbr":p.get("away",{}).get("abbr"),
                    "win_probability":safe_num(p.get("away",{}).get("win_probability")),
                    "fair_moneyline":p.get("away",{}).get("fair_moneyline"),
                    "expected_goals":safe_num(p.get("away",{}).get("expected_goals")),
                    "regulation_win_probability":safe_num(p.get("away",{}).get("regulation_win_probability"))
                },
                "home":{
                    "abbr":p.get("home",{}).get("abbr"),
                    "win_probability":safe_num(p.get("home",{}).get("win_probability")),
                    "fair_moneyline":p.get("home",{}).get("fair_moneyline"),
                    "expected_goals":safe_num(p.get("home",{}).get("expected_goals")),
                    "regulation_win_probability":safe_num(p.get("home",{}).get("regulation_win_probability"))
                },
                "overtime_probability":safe_num(p.get("overtime_probability")),
                "rest_state":deepcopy(p.get("rest_state")),
                # Preserve the component probabilities and forecast-time goalie
                # adjustment inputs so future calibration audits can compare
                # team-only and goalie-adjusted predictions without rebuilding
                # them from postgame data.
                "probability_components":{
                    "team_only":{
                        "away_win_probability":safe_num(p.get("away",{}).get("team_only_win_probability")),
                        "home_win_probability":safe_num(p.get("home",{}).get("team_only_win_probability"))
                    },
                    "goalie_adjustment_status":p.get("goalie_adjustment_status"),
                    "goalie_adjustment":deepcopy(p.get("goalie_adjustment"))
                },
                "model_flags":deepcopy(p.get("flags") or []),
                "goalie_context":{
                    "away":side_context(ctx,"away"),
                    "home":side_context(ctx,"home")
                },
                "lineup_state":deepcopy(p.get("lineup_state")),
                "market_benchmark":{
                    "status":"NOT_CAPTURED",
                    "away_implied_probability":None,
                    "home_implied_probability":None,
                    "source":None,
                    "captured_at":None,
                    "capture_requested_at":None,
                    "minimum_books_required":3,
                    "book_count":0,
                    "total_book_count":0,
                    "home_probability_min":None,
                    "home_probability_max":None,
                    "home_probability_std":None,
                    "earliest_odds_timestamp":None,
                    "latest_odds_timestamp":None,
                    "books":[]
                }
            },
            "result":{
                "status":"OPEN",
                "away_score":None,
                "home_score":None,
                "winner_abbr":None,
                "last_period_type":None,
                "went_ot":None,
                "went_shootout":None,
                "source":None,
                "settled_at":None
            },
            "evaluation":None
        }
        write_json(out,record)
        created.append(gid)
        print(f"FROZEN {gid} {away}@{home} at {minutes:.1f} minutes before start")
    return created

def fetch_scores(date: str):
    url=f"https://api-web.nhle.com/v1/score/{date}"
    r=requests.get(url,timeout=45,headers={"User-Agent":"HockeyProjectionLab/1.0 forward-ledger"})
    r.raise_for_status()
    payload=r.json()
    return {str(g.get("id")):g for g in payload.get("games",[])}, url

def evaluate(record):
    f=record["forecast"]
    r=record["result"]
    ap=safe_num(f.get("away",{}).get("win_probability"))
    hp=safe_num(f.get("home",{}).get("win_probability"))
    away_win=1 if r.get("winner_abbr")==record["matchup"]["away_abbr"] else 0
    home_win=1-away_win
    predicted=None
    if ap is not None and hp is not None:
        predicted=record["matchup"]["away_abbr"] if ap>hp else record["matchup"]["home_abbr"]
    actual_prob=ap if away_win else hp
    eps=1e-15

    ev={
        "predicted_winner_abbr":predicted,
        "winner_correct":None if predicted is None else predicted==r.get("winner_abbr"),
        "probability_assigned_to_winner":actual_prob,
        "brier":None if hp is None else (hp-home_win)**2,
        "log_loss":None if actual_prob is None else -math.log(min(1-eps,max(eps,actual_prob))),
        "overtime_brier":None,
        "away_goal_abs_error":None,
        "home_goal_abs_error":None,
        "goal_mae_per_team":None,
        "total_goals_abs_error":None,
        "goal_diff_abs_error":None
    }
    otp=safe_num(f.get("overtime_probability"))
    if otp is not None and r.get("went_ot") is not None:
        ev["overtime_brier"]=(otp-(1 if r["went_ot"] else 0))**2

    ax=safe_num(f.get("away",{}).get("expected_goals"))
    hx=safe_num(f.get("home",{}).get("expected_goals"))
    if ax is not None and hx is not None:
        ag=r.get("away_score")
        hg=r.get("home_score")
        if isinstance(ag,(int,float)) and isinstance(hg,(int,float)):
            ae=abs(ax-ag)
            he=abs(hx-hg)
            ev["away_goal_abs_error"]=ae
            ev["home_goal_abs_error"]=he
            ev["goal_mae_per_team"]=(ae+he)/2
            ev["total_goals_abs_error"]=abs((ax+hx)-(ag+hg))
            ev["goal_diff_abs_error"]=abs((hx-ax)-(hg-ag))
    return ev

def settle_records():
    GAMES_DIR.mkdir(parents=True,exist_ok=True)
    open_records=[]
    for path in sorted(GAMES_DIR.glob("*.json")):
        rec=read_json(path,{}) or {}
        if rec.get("result",{}).get("status")!="FINAL":
            open_records.append((path,rec))

    by_date={}
    for path,rec in open_records:
        by_date.setdefault(rec.get("game_date"),[]).append((path,rec))

    settled=[]
    for date,items in by_date.items():
        if not date:
            continue
        games,url=fetch_scores(date)
        for path,rec in items:
            gid=str(rec.get("game_id"))
            game=games.get(gid)
            if not game or str(game.get("gameState","")).upper() not in FINAL_STATES:
                continue
            away=game.get("awayTeam") or {}
            home=game.get("homeTeam") or {}
            ag=away.get("score")
            hg=home.get("score")
            if not isinstance(ag,(int,float)) or not isinstance(hg,(int,float)) or ag==hg:
                continue
            last=((game.get("gameOutcome") or {}).get("lastPeriodType") or "").upper() or None
            rec["result"]={
                "status":"FINAL",
                "away_score":ag,
                "home_score":hg,
                "winner_abbr":rec["matchup"]["away_abbr"] if ag>hg else rec["matchup"]["home_abbr"],
                "last_period_type":last,
                "went_ot":last in {"OT","SO"},
                "went_shootout":last=="SO",
                "source":url,
                "settled_at":now_et().isoformat()
            }
            rec["evaluation"]=evaluate(rec)
            write_json(path,rec)
            settled.append(gid)
            print(f"SETTLED {gid} {ag}-{hg}")
    return settled

def avg(vals):
    vals=[v for v in vals if isinstance(v,(int,float)) and math.isfinite(v)]
    return None if not vals else sum(vals)/len(vals)

def build_summary():
    rows=[]
    for path in sorted(GAMES_DIR.glob("*.json")):
        r=read_json(path,{}) or {}
        f=r.get("forecast") or {}
        res=r.get("result") or {}
        ev=r.get("evaluation") or {}
        rows.append({
            "game_id":r.get("game_id"),
            "game_date":r.get("game_date"),
            "start_time_et":r.get("start_time_et"),
            "away_abbr":r.get("matchup",{}).get("away_abbr"),
            "home_abbr":r.get("matchup",{}).get("home_abbr"),
            "model_id":f.get("model",{}).get("model_id"),
            "model_version":f.get("model",{}).get("version"),
            "feature_version":f.get("model",{}).get("feature_version"),
            "frozen_at":r.get("frozen_at"),
            "status":res.get("status"),
            "winner_abbr":res.get("winner_abbr"),
            "predicted_winner_abbr":ev.get("predicted_winner_abbr"),
            "winner_correct":ev.get("winner_correct"),
            "brier":ev.get("brier"),
            "log_loss":ev.get("log_loss"),
            "probability_assigned_to_winner":ev.get("probability_assigned_to_winner"),
            "goal_mae_per_team":ev.get("goal_mae_per_team"),
            "total_goals_abs_error":ev.get("total_goals_abs_error")
        })

    settled=[x for x in rows if x["status"]=="FINAL"]
    scored=[x for x in settled if isinstance(x.get("winner_correct"),bool)]
    xg=[x for x in settled if isinstance(x.get("goal_mae_per_team"),(int,float))]

    versions={}
    for x in settled:
        key=f"{x.get('model_id') or 'UNKNOWN'}:{x.get('model_version') or 'UNKNOWN'}:{x.get('feature_version') or 'UNKNOWN'}"
        versions.setdefault(key,[]).append(x)

    by_model={}
    for key,xs in versions.items():
        ss=[x for x in xs if isinstance(x.get("winner_correct"),bool)]
        by_model[key]={
            "games_settled":len(xs),
            "accuracy":avg([1 if x["winner_correct"] else 0 for x in ss]),
            "brier":avg([x.get("brier") for x in ss]),
            "log_loss":avg([x.get("log_loss") for x in ss]),
            "avg_probability_assigned_to_winner":avg([x.get("probability_assigned_to_winner") for x in ss])
        }

    by_pick_side={}
    for side in ("away","home"):
        side_rows=[]
        for x in settled:
            side_abbr=x.get(side+"_abbr")
            if x.get("predicted_winner_abbr")==side_abbr and isinstance(x.get("winner_correct"),bool):
                side_rows.append(x)
        by_pick_side[side]={
            "games":len(side_rows),
            "accuracy":avg([1 if x["winner_correct"] else 0 for x in side_rows]),
            "brier":avg([x.get("brier") for x in side_rows]),
            "log_loss":avg([x.get("log_loss") for x in side_rows])
        }

    summary={
        "schema_version":"1.0",
        "generated_at":now_et().isoformat(),
        "sample":{
            "games_frozen":len(rows),
            "games_settled":len(settled),
            "games_open":len(rows)-len(settled),
            "winner_games_scored":len(scored),
            "xg_games_scored":len(xg)
        },
        "metrics":{
            "accuracy":avg([1 if x["winner_correct"] else 0 for x in scored]),
            "brier":avg([x.get("brier") for x in scored]),
            "log_loss":avg([x.get("log_loss") for x in scored]),
            "avg_probability_assigned_to_winner":avg([x.get("probability_assigned_to_winner") for x in scored]),
            "xg_mae_per_team":avg([x.get("goal_mae_per_team") for x in xg]),
            "total_goals_mae":avg([x.get("total_goals_abs_error") for x in xg])
        },
        "by_model":by_model,
        "by_pick_side":by_pick_side,
        "games":sorted(rows,key=lambda x:(x.get("game_date") or "",x.get("start_time_et") or "",str(x.get("game_id"))),reverse=True)
    }
    prior=read_json(SUMMARY_PATH,{}) or {}
    def material(x):
        y=deepcopy(x or {})
        y.pop("generated_at",None)
        return y
    if material(prior)!=material(summary):
        write_json(SUMMARY_PATH,summary)
        return summary
    return prior

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["freeze","settle","both"],default="both")
    ap.add_argument("--min-before",type=float,default=4.0,
                    help="Do not freeze closer than this many minutes before start")
    ap.add_argument("--max-before",type=float,default=18.0,
                    help="Freeze only within this many minutes before start")
    args=ap.parse_args()

    GAMES_DIR.mkdir(parents=True,exist_ok=True)
    frozen=[]
    settled=[]
    if args.mode in {"freeze","both"}:
        frozen=freeze_records(args.min_before,args.max_before)
    if args.mode in {"settle","both"}:
        settled=settle_records()
    summary=build_summary()
    print(json.dumps({
        "frozen":frozen,
        "settled":settled,
        "sample":summary["sample"],
        "metrics":summary["metrics"]
    },indent=2))

if __name__=="__main__":
    main()
