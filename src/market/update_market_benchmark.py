#!/usr/bin/env python3
"""Independent, benchmark-only NHL moneyline consensus. Never imported by the model.

Only saved PRE-PUCK-DROP price snapshots may become the last-pregame benchmark.
A one-book quote is explicitly not represented as a multi-book consensus.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
TZ = ZoneInfo("America/New_York")
SCHEDULE = ROOT / "public/data/today-games.json"
OUTPUT = ROOT / "public/data/market-benchmark.json"
ARCHIVE = ROOT / "research/market_benchmarks/live"
API = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"

NAMES = {
    "anaheim ducks":"ANA","boston bruins":"BOS","buffalo sabres":"BUF",
    "carolina hurricanes":"CAR","columbus blue jackets":"CBJ","calgary flames":"CGY",
    "chicago blackhawks":"CHI","colorado avalanche":"COL","dallas stars":"DAL",
    "detroit red wings":"DET","edmonton oilers":"EDM","florida panthers":"FLA",
    "los angeles kings":"LAK","minnesota wild":"MIN","montreal canadiens":"MTL",
    "new jersey devils":"NJD","nashville predators":"NSH","new york islanders":"NYI",
    "new york rangers":"NYR","ottawa senators":"OTT","philadelphia flyers":"PHI",
    "pittsburgh penguins":"PIT","seattle kraken":"SEA","san jose sharks":"SJS",
    "st louis blues":"STL","saint louis blues":"STL","tampa bay lightning":"TBL",
    "toronto maple leafs":"TOR","utah mammoth":"UTA","utah hockey club":"UTA",
    "vancouver canucks":"VAN","vegas golden knights":"VGK",
    "winnipeg jets":"WPG","washington capitals":"WSH",
}

def now_utc():
    override = os.environ.get("HOCKEY_LAB_NOW")
    if override:
        return date_time(override)
    return datetime.now(timezone.utc)

def date_time(s):
    if not s:
        return None
    try:
        result = datetime.fromisoformat(str(s).replace("Z","+00:00"))
        if result.tzinfo is None:
            return None
        return result.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None

def stamp(t):
    return t.isoformat().replace("+00:00","Z")

def team(name):
    n = re.sub(r"[^a-z0-9 ]","",str(name).lower())
    n = " ".join(n.split())
    return NAMES.get(n)

def probability(price):
    if type(price) not in (int, float) or not math.isfinite(price) or abs(price)<100:
        return None
    return -price/(100-price) if price<0 else 100/(100+price)

def load_json(path,default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

def event_key(event):
    return (team(event.get("away_team")),team(event.get("home_team")))

def book_quotes(event,start,now):
    """Two-sided 2-way h2h quotes only, timestamped, fresh, and pre-start."""
    found = {}
    away_name = event.get("away_team")
    home_name = event.get("home_team")
    for book in event.get("bookmakers") or []:
        key = str(book.get("key") or "").strip().lower()
        if not key:
            continue
        for m in book.get("markets") or []:
            if m.get("key") != "h2h":
                continue
            when = date_time(m.get("last_update") or book.get("last_update"))
            if not when or when>now+timedelta(minutes=2) or when>start:
                continue
            if now-when>timedelta(minutes=90):
                continue
            outcomes = {o.get("name"):o.get("price") for o in m.get("outcomes") or []}
            if len(outcomes)!=2 or away_name not in outcomes or home_name not in outcomes:
                continue
            ap = outcomes[away_name]
            hp = outcomes[home_name]
            ia,ih=probability(ap),probability(hp)
            if ia is None or ih is None:
                continue
            fair=ia/(ia+ih)
            if not (0.01<=fair<=0.99):
                continue
            row={
                "bookmaker":key,"away_ml":int(round(ap)),"home_ml":int(round(hp)),
                "away_no_vig":round(fair,7),"home_no_vig":round(1-fair,7),
                "quote_updated_at":stamp(when),
            }
            if key not in found or row["quote_updated_at"]>found[key]["quote_updated_at"]:
                found[key]=row
    return [found[k] for k in sorted(found)]

def build_row(game,event,now):
    start=date_time(game.get("start_time_et"))
    when=date_time(event.get("commence_time"))
    if not start or not when or now>=start or abs((when-start).total_seconds())>3600:
        return None
    if event_key(event)!=(game.get("away_abbr"),game.get("home_abbr")):
        return None
    books=book_quotes(event,start,now)
    if not books:
        return None
    a=float(statistics.median(b["away_no_vig"] for b in books))
    price_away=int(round(statistics.median(b["away_ml"] for b in books)))
    price_home=int(round(statistics.median(b["home_ml"] for b in books)))
    return {
        "game_id":game["game_id"],"away_abbr":game["away_abbr"],
        "home_abbr":game["home_abbr"],"start_time_et":game["start_time_et"],
        "market_ml_away":price_away,"market_ml_home":price_home,
        "no_vig_away":round(a,7),"no_vig_home":round(1-a,7),
        "captured_at":stamp(now),
        "last_quote_at":max(b["quote_updated_at"] for b in books),
        "book_count":len(books),"books":books,
        "status":"CURRENT" if len(books)>=2 else "SINGLE_BOOK",
        "source":"The Odds API","source_url":"https://the-odds-api.com/",
    }

def fetch_feed(key):
    params=urlencode({"apiKey":key,"regions":"us","markets":"h2h","oddsFormat":"american","dateFormat":"iso"})
    with urlopen(Request(API+"?"+params,headers={"User-Agent":"HockeyLab-MarketBenchmark/1.0"}),timeout=40) as r:
        return json.load(r)

def compose(schedule,events,previous,now):
    day=now.astimezone(TZ).date().isoformat()
    if schedule.get("date")!=day:
        raise ValueError("Schedule date is not today; refusing mismatched market snapshot")
    old={str(x.get("game_id")):x for x in previous.get("games",[])} if previous.get("date")==day else {}
    by_match={}
    for e in events:
        key=event_key(e)
        if all(key):
            by_match.setdefault(key,[]).append(e)
    rows=[]
    for game in schedule.get("games",[]):
        start=date_time(game.get("start_time_et"))
        if not start:
            continue
        gid=str(game.get("game_id"))
        existing=old.get(gid)
        # Hard freeze: source fetched after puck drop may NEVER replace pregame data.
        if now>=start:
            if existing:
                saved=date_time(existing.get("captured_at"))
                if existing.get("source")=="The Odds API" and saved and saved<start:
                    minutes=(start-saved).total_seconds()/60
                    existing["status"]="CLOSING" if minutes<=25 and existing.get("book_count",0)>=2 else "LAST_PREGAME"
                rows.append(existing)
            continue
        candidates=by_match.get((game.get("away_abbr"),game.get("home_abbr")),[])
        built=[x for e in candidates if (x:=build_row(game,e,now))]
        if built:
            rows.append(max(built,key=lambda x:(x["book_count"],x["last_quote_at"])))
        elif existing:
            rows.append(existing)
    return {
        "schema_version":"1.0","date":day,"generated_at":stamp(now),
        "source":"The Odds API","source_url":"https://the-odds-api.com/",
        "source_tier":"MULTIBOOK_WHEN_AT_LEAST_2_BOOKS",
        "benchmark_only":True,
        "method":"Median of per-sportsbook two-way, de-vigged away win probabilities; displayed market moneyline is median across book prices.",
        "games":rows,
    }

def self_test():
    t=date_time("2026-11-03T23:50:00Z")
    game={"game_id":101,"away_abbr":"CAR","home_abbr":"NSH","start_time_et":"2026-11-04T00:00:00Z"}
    def bk(name,ap,hp):
        return {"key":name,"markets":[{"key":"h2h","last_update":"2026-11-03T23:49:00Z",
          "outcomes":[{"name":"Carolina Hurricanes","price":ap},{"name":"Nashville Predators","price":hp}]}]}
    feed={"away_team":"Carolina Hurricanes","home_team":"Nashville Predators",
      "commence_time":"2026-11-04T00:00:00Z","bookmakers":[bk("a",-200,170),bk("b",-195,165)]}
    row=build_row(game,feed,t)
    assert row and row["book_count"]==2 and row["status"]=="CURRENT"
    assert abs(row["no_vig_away"]+row["no_vig_home"]-1)<1e-8
    assert team("Utah Mammoth")=="UTA"
    assert probability(-200)==2/3
    bad=dict(feed,bookmakers=[bk("late",-200,170)])
    bad["bookmakers"][0]["markets"][0]["last_update"]="2026-11-04T00:01:00Z"
    assert build_row(game,bad,t) is None
    slate={"date":"2026-11-03","games":[game]}
    pre=compose(slate,[feed],{"games":[]},t)
    frozen=compose(slate,[],pre,date_time("2026-11-04T00:05:00Z"))
    assert frozen["games"][0]["status"]=="CLOSING"
    assert frozen["games"][0]["captured_at"]==row["captured_at"]
    assert compose(slate,[dict(feed,bookmakers=[bk("new",-500,420)])],frozen,date_time("2026-11-04T00:20:00Z"))["games"][0]["market_ml_away"]==row["market_ml_away"]
    # November 3 is EST (UTC-5): fixed checks are 9 AM, noon, 3 PM, 5 PM, 6 PM ET.
    fixed_allowed=["2026-11-03T14:00:00Z","2026-11-03T17:00:00Z",
                   "2026-11-03T20:00:00Z","2026-11-03T22:00:00Z",
                   "2026-11-03T23:00:00Z"]
    for at in fixed_allowed:
        assert eligible_poll_slot(date_time(at).astimezone(TZ),slate,set()) is not None, at
    # Final close window: one pull in the last 10 minutes before each unique start time.
    close_slot=eligible_poll_slot(date_time("2026-11-03T23:52:00Z").astimezone(TZ),slate,set())
    assert close_slot and close_slot.startswith("CLOSE:")
    assert eligible_poll_slot(date_time("2026-11-03T23:52:00Z").astimezone(TZ),slate,{close_slot}) is None
    # A missed fixed check remains due until the next checkpoint.
    assert eligible_poll_slot(date_time("2026-11-03T20:24:00Z").astimezone(TZ),slate,set()).startswith("FIXED:")
    # After that fixed checkpoint is captured, it does not spend again.
    three=eligible_poll_slot(date_time("2026-11-03T20:24:00Z").astimezone(TZ),slate,set())
    assert eligible_poll_slot(date_time("2026-11-03T20:25:00Z").astimezone(TZ),slate,{three}) is None
    # In daylight time the UTC offset moves but 9 AM ET remains a valid fixed window.
    summer={"date":"2027-04-03","games":[]}
    assert eligible_poll_slot(date_time("2027-04-03T13:00:00Z").astimezone(TZ),summer,set()) is not None
    print("Market benchmark self-test PASS (no-vig, matchup, timestamp, freeze, schedule, no leakage)")

def eligible_poll_slot(local,schedule,captured):
    """Return the single market slot that should be captured now, or None.

    Fixed checkpoints are 9 AM, noon, 3 PM, 5 PM and 6 PM ET. A fixed
    checkpoint remains recoverable until the next fixed checkpoint. The final
    6 PM checkpoint remains recoverable until the first game's closing window.

    Closing checks take priority and occur once per unique puck-drop time in
    the final 10 minutes. Persisted slot keys make retries idempotent.
    """
    # Closing window has priority over a late fixed checkpoint.
    starts=[]
    for game in schedule.get("games",[]):
        start=date_time(game.get("start_time_et"))
        if start:
            starts.append(start.astimezone(TZ))
    unique_starts=sorted(set(starts))
    for start in unique_starts:
        seconds=(start-local).total_seconds()
        key="CLOSE:"+start.isoformat()
        if 0 < seconds <= 10*60 and key not in captured:
            return key

    fixed_times=[(9,0),(12,0),(15,0),(17,0),(18,0)]
    fixed_targets=[
        local.replace(hour=h,minute=m,second=0,microsecond=0)
        for h,m in fixed_times
    ]
    for idx in range(len(fixed_targets)-1,-1,-1):
        target=fixed_targets[idx]
        if local < target:
            continue
        key="FIXED:"+target.isoformat()
        if key in captured:
            return None

        # The checkpoint expires when the next fixed checkpoint begins.
        if idx+1 < len(fixed_targets):
            expiry=fixed_targets[idx+1]
        else:
            # After 6 PM, stop trying to backfill the 6 PM checkpoint once
            # the earliest game's closing window begins.
            future_starts=[s for s in unique_starts if s>target]
            expiry=(min(future_starts)-timedelta(minutes=10)) if future_starts else target+timedelta(hours=2)

        if local < expiry:
            return key
        return None
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--force",action="store_true",help="Allow manually triggered collection outside daytime.")
    ap.add_argument("--check-due",action="store_true",help="Print the currently due slot without calling the odds provider.")
    ap.add_argument("--fixture",help="Offline feed JSON for testing")
    args=ap.parse_args()
    if args.self_test:
        self_test()
        return
    now=now_utc()
    local=now.astimezone(TZ)
    schedule=load_json(SCHEDULE,{}) or {}
    if schedule.get("date")!=local.date().isoformat():
        print("SKIP stale slate; no market refresh")
        return
    old=load_json(OUTPUT,{}) or {}
    captured=set(old.get("captured_poll_slots") or []) if old.get("date")==schedule.get("date") else set()
    slot=eligible_poll_slot(local,schedule,captured)
    if args.check_due:
        print(slot or "NONE")
        return
    if not args.force and not slot:
        print("SKIP outside approved fixed/closing market collection windows")
        return
    remaining=any((start:=date_time(x.get("start_time_et"))) and start>now for x in schedule.get("games",[]))
    if not remaining:
        # Finalize last saved pregame quotes without spending API credits after the games begin.
        if old.get("date")==schedule.get("date") and old.get("games"):
            output=compose(schedule,[],old,now)
            if output.get("games")!=old.get("games"):
                save(OUTPUT,output)
                save(ARCHIVE/(output["date"]+".json"),output)
                print("FROZEN all final pregame market snapshots")
            else:
                print("No new market snapshots to freeze")
        return

    # Idempotency: never spend twice in the same fixed or game-close slot.
    if not args.force and slot in captured:
        print("SKIP already captured this market window:",slot)
        return
    if args.fixture:
        events=load_json(Path(args.fixture),[])
    else:
        key=os.getenv("THE_ODDS_API_KEY","")
        if not key:
            print("SKIP THE_ODDS_API_KEY not configured; existing data left untouched")
            return
        events=fetch_feed(key)
    if not isinstance(events,list):
        raise ValueError("Odds provider did not return an event list")
    output=compose(schedule,events,old,now)
    if not output["games"]:
        print("No usable odds; preserving existing benchmark file")
        return
    slot_key=slot if slot else "FORCED:"+stamp(now)
    captured.add(slot_key)
    output["last_poll_slot"]=slot_key
    output["captured_poll_slots"]=sorted(captured)
    save(OUTPUT,output)
    save(ARCHIVE/(output["date"]+".json"),output)
    print("Updated independently sourced NHL market benchmark:",len(output["games"]),"games at",stamp(now))

if __name__=="__main__":
    main()
