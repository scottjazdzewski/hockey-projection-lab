#!/usr/bin/env python3
"""Fetch timestamped PuckAPI full-game moneylines for canonical NHL game IDs.

The command is a dry-run by default. Historical requests are chargeable, so an
explicit credit cap is required before any API call is made.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://mcp.puckapi.com/v1/get_game_detail"
CREDITS_PER_GAME = 10


def _schedule(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    id_col = "game_id" if "game_id" in frame else "nhl_game_id"
    date_col = "game_date" if "game_date" in frame else "date"
    start_col = next((c for c in ["scheduled_start_utc", "scheduled_start", "start_time_utc", "game_start_utc"] if c in frame), None)
    if start_col is None:
        raise ValueError("Canonical games file must include scheduled_start_utc")
    home_col = next((c for c in ["home_team_abbr", "home_abbr"] if c in frame), None)
    away_col = next((c for c in ["away_team_abbr", "away_abbr"] if c in frame), None)
    if not home_col or not away_col:
        raise ValueError("Canonical games file must include home and away team abbreviations")
    frame = frame.rename(columns={id_col: "game_id", date_col: "game_date", start_col: "scheduled_start_utc",
                                  home_col: "home_abbr", away_col: "away_abbr"})
    frame["game_id"] = frame.game_id.astype("string")
    frame["scheduled_start_utc"] = pd.to_datetime(frame.scheduled_start_utc, utc=True, errors="coerce")
    frame["game_date"] = pd.to_datetime(frame.game_date, errors="coerce").dt.date.astype("string")
    if frame.game_id.duplicated().any():
        raise ValueError("Canonical NHL games must be unique by game_id")
    return frame.dropna(subset=["scheduled_start_utc"])


def _post(game_id: str, api_key: str) -> dict:
    body = json.dumps({"game_id": str(game_id)}).encode("utf-8")
    request = Request(API_URL, data=body, headers={
        "Content-Type": "application/json", "x-api-key": api_key,
        "User-Agent": "HockeyProjectionLab/1.0 market-benchmark",
    }, method="POST")
    with urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def _payload(response: dict) -> dict:
    return response.get("data", response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", required=True, help="Canonical NHL games parquet")
    parser.add_argument("--out", default=str(ROOT / "artifacts/market/staging/puckapi-closing-lines.parquet"))
    parser.add_argument("--start-season", type=int, default=2019, help="First source_season start year to include")
    parser.add_argument("--end-season", type=int, default=2026, help="Last source_season start year to include")
    parser.add_argument("--max-credits", type=int, default=0,
                        help="Maximum credits explicitly approved for this run; omit to remain dry-run")
    parser.add_argument("--sleep-seconds", type=float, default=1.05,
                        help="Delay between requests; respect provider rate limits")
    args = parser.parse_args()

    games = _schedule(Path(args.games))
    if "source_season" in games:
        seasons = pd.to_numeric(games.source_season, errors="coerce")
        games = games[seasons.between(args.start_season, args.end_season)]
    # Historical closing prices are only queried for games with an authoritative
    # result already in the canonical spine; scheduled future games are not part
    # of a historical backfill.
    if "result_available" in games:
        games = games[games.result_available.fillna(False).astype(bool)]
    elif "winner_abbr" in games:
        games = games[games.winner_abbr.notna()]
    elif "home_win" in games:
        games = games[pd.to_numeric(games.home_win, errors="coerce").isin([0, 1])]
    output = Path(args.out)
    if output.exists():
        previous = pd.read_parquet(output)
        completed = set(previous.game_id.astype(str))
    else:
        previous = pd.DataFrame()
        completed = set()
    todo = games[~games.game_id.astype(str).isin(completed)].copy()
    estimated_credits = int(len(todo) * CREDITS_PER_GAME)
    estimate = {
        "provider": "PuckAPI", "games_to_request": int(len(todo)),
        "estimated_credits": estimated_credits,
        "credits_per_game_documented": CREDITS_PER_GAME,
        "seasons_requested": [args.start_season, args.end_season],
        "completed_game_ids_already_in_output": len(completed),
        "dry_run": args.max_credits <= 0,
    }
    if args.max_credits <= 0:
        print(json.dumps(estimate, indent=2))
        return
    if args.max_credits < estimated_credits:
        raise SystemExit(f"Credit cap {args.max_credits} is below estimated request cost {estimated_credits}; no calls made")
    api_key = os.environ.get("PUCKAPI_API_KEY")
    if not api_key:
        raise SystemExit("PUCKAPI_API_KEY is required for --max-credits; no calls made")

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    audit = []
    attempts = 0
    successful_calls = 0
    for game in todo.itertuples(index=False):
        if (attempts + 1) * CREDITS_PER_GAME > args.max_credits:
            break
        attempts += 1
        try:
            response = _payload(_post(str(game.game_id), api_key))
        except (HTTPError, URLError, TimeoutError, ValueError, TypeError) as exc:
            audit.append({"game_id": str(game.game_id), "status": "REQUEST_ERROR", "detail": str(exc)})
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            continue
        successful_calls += 1
        response_game_id = response.get("game_id") or (response.get("game") or {}).get("id")
        if response_game_id is not None and str(response_game_id) != str(game.game_id):
            audit.append({"game_id": str(game.game_id), "status": "GAME_ID_MISMATCH", "provider_game_id": str(response_game_id)})
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            continue
        provider_home = response.get("home_team")
        provider_away = response.get("away_team")
        if isinstance(provider_home, dict):
            provider_home = provider_home.get("abbrev")
        if isinstance(provider_away, dict):
            provider_away = provider_away.get("abbrev")
        odds = response.get("odds") or []
        found = 0
        for line in odds:
            stamp = line.get("capturedAt") or line.get("captured_at")
            if not stamp or line.get("mlAway") is None or line.get("mlHome") is None:
                audit.append({"game_id": str(game.game_id), "sportsbook": line.get("bookmaker"), "status": "MISSING_PRICE_OR_TIMESTAMP"})
                continue
            timestamp = pd.to_datetime(stamp, utc=True, errors="coerce")
            if pd.isna(timestamp):
                audit.append({"game_id": str(game.game_id), "sportsbook": line.get("bookmaker"), "status": "INVALID_ODDS_TIMESTAMP", "odds_timestamp": str(stamp)})
                continue
            if timestamp >= game.scheduled_start_utc:
                audit.append({"game_id": str(game.game_id), "sportsbook": line.get("bookmaker"), "status": "TIMESTAMP_AT_OR_AFTER_START", "odds_timestamp": str(stamp), "scheduled_start_utc": game.scheduled_start_utc.isoformat()})
                continue
            rows.append({
                "game_id": str(game.game_id), "game_date": str(game.game_date),
                "scheduled_start_utc": game.scheduled_start_utc.isoformat(),
                "sportsbook": str(line.get("bookmaker", "")).strip().lower(),
                "away_moneyline": line.get("mlAway"), "home_moneyline": line.get("mlHome"),
                "odds_timestamp": stamp, "source": "PuckAPI",
                "provider_event_id": str(response_game_id or game.game_id),
                "snapshot_type": line.get("snapshotType"),
                "provider_home_team": provider_home,
                "provider_away_team": provider_away,
            })
            found += 1
        if found == 0:
            audit.append({"game_id": str(game.game_id), "status": "NO_VALID_PREGAME_MONEYLINE"})
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    incoming = pd.DataFrame(rows)
    if not previous.empty and not incoming.empty:
        combined = pd.concat([previous, incoming], ignore_index=True)
    elif not previous.empty:
        combined = previous
    else:
        combined = incoming
    if not combined.empty:
        combined.to_parquet(output, index=False)
    audit_path = output.with_name("puckapi-fetch-audit.json")
    prior_audit = []
    if audit_path.exists():
        prior_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_path.write_text(json.dumps(prior_audit + audit, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({**estimate, "dry_run": False, "calls_attempted": attempts,
                      "successful_calls": successful_calls,
                      "book_line_rows_received": len(rows), "games_with_no_valid_pregame_line_or_issue": len(audit),
                      "output": str(output), "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
