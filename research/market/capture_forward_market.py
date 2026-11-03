#!/usr/bin/env python3
"""Attach a pregame PuckAPI market snapshot to frozen forward-ledger records.

Only forecast.market_benchmark is changed. Forecast probabilities and model
metadata are never read as inputs to this collection step or modified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from market_math import PRIMARY_BOOKS, canonical_sportsbook, no_vig_probabilities  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LEDGER_DIR = ROOT / "artifacts/forward/games"
API_URL = "https://mcp.puckapi.com/v1/get_odds"
ET = ZoneInfo("America/New_York")
CREDITS_PER_GAME = 10


def parse_dt(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=ET)
    except Exception:
        return None


def fetch(game_id: str, api_key: str) -> dict:
    body = json.dumps({"game_id": str(game_id)}).encode("utf-8")
    req = Request(API_URL, data=body, headers={
        "Content-Type": "application/json", "x-api-key": api_key,
        "User-Agent": "HockeyProjectionLab/1.0 market-benchmark",
    }, method="POST")
    with urlopen(req, timeout=45) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("data", data)


def normalize(game_id: str, start, response: dict, requested_at: datetime) -> dict:
    response_game_id = response.get("game_id")
    if response_game_id is not None and str(response_game_id) != str(game_id):
        raise ValueError(f"PuckAPI game_id mismatch: requested {game_id}, received {response_game_id}")
    books = {}
    rejected_after_start = 0
    rejected_after_capture = 0
    for line in response.get("odds") or []:
        book = canonical_sportsbook(line.get("bookmaker"))
        source_name = str(line.get("bookmaker") or "").strip()
        stamp = parse_dt(line.get("capturedAt") or line.get("captured_at"))
        away_ml = line.get("mlAway", line.get("ml_away"))
        home_ml = line.get("mlHome", line.get("ml_home"))
        if not book or not stamp or away_ml is None or home_ml is None:
            continue
        if stamp.astimezone(timezone.utc) >= start.astimezone(timezone.utc):
            rejected_after_start += 1
            continue
        if stamp.astimezone(timezone.utc) > requested_at.astimezone(timezone.utc):
            rejected_after_capture += 1
            continue
        try:
            away_p, home_p = no_vig_probabilities(away_ml, home_ml)
        except (ValueError, TypeError, OverflowError):
            continue
        row = {
            "sportsbook": book, "sportsbook_source_name": source_name,
            "consensus_eligible": book in PRIMARY_BOOKS,
            "away_moneyline": int(away_ml), "home_moneyline": int(home_ml),
            "odds_timestamp": stamp.astimezone(timezone.utc).isoformat(),
            "minutes_before_puck_drop": (start.astimezone(timezone.utc)-stamp.astimezone(timezone.utc)).total_seconds()/60,
            "snapshot_type": line.get("snapshotType", line.get("snapshot_type")),
            "away_no_vig_probability": away_p, "home_no_vig_probability": home_p,
            "source": "PuckAPI",
        }
        prior = books.get(book)
        if prior is None or row["odds_timestamp"] > prior["odds_timestamp"]:
            books[book] = row
    rows = list(books.values())
    eligible_rows = [r for r in rows if r["consensus_eligible"]]
    probs = [r["home_no_vig_probability"] for r in eligible_rows]
    count = len(eligible_rows)
    status = "CONSENSUS" if count >= 3 else ("LIMITED_CONSENSUS" if count == 2 else ("SINGLE_BOOK" if count == 1 else "RETRYABLE_UNAVAILABLE"))
    median_home = float(median(probs)) if count else None
    return {
        "status": status,
        "source": "PuckAPI",
        "capture_requested_at": requested_at.astimezone(timezone.utc).isoformat(),
        "captured_at": requested_at.astimezone(timezone.utc).isoformat(),
        "minimum_books_required": 3,
        "book_count": count,
        "total_book_count": len(rows),
        "away_implied_probability": None if median_home is None else 1.0-median_home,
        "home_implied_probability": median_home,
        "home_probability_min": min(probs) if probs else None,
        "home_probability_max": max(probs) if probs else None,
        "home_probability_std": (sum((p-sum(probs)/count)**2 for p in probs)/count)**0.5 if count else None,
        "earliest_closing_timestamp": min((r["odds_timestamp"] for r in eligible_rows), default=None),
        "latest_closing_timestamp": max((r["odds_timestamp"] for r in eligible_rows), default=None),
        "books": sorted(rows, key=lambda r: r["sportsbook"]),
        "rejected_at_or_after_start_rows": rejected_after_start,
        "rejected_after_capture_rows": rejected_after_capture,
        "source_game_id": str(response_game_id or game_id),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", default=str(LEDGER_DIR))
    parser.add_argument("--max-minutes-before-start", type=float, default=25.0)
    parser.add_argument("--max-credits", type=int, default=0,
                        help="Explicit per-run API credit cap; omit to make no network calls")
    parser.add_argument("--as-of", default=None, help="ISO datetime override for dry-run/replay")
    args = parser.parse_args()
    now = parse_dt(args.as_of) if args.as_of else datetime.now(ET)
    api_key = os.environ.get("PUCKAPI_API_KEY")
    files = sorted(Path(args.ledger_dir).glob("*.json"))
    candidates = []
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        start = parse_dt(record.get("start_time_et"))
        if not start or start.astimezone(timezone.utc) <= now.astimezone(timezone.utc):
            continue
        minutes = (start.astimezone(timezone.utc)-now.astimezone(timezone.utc)).total_seconds()/60
        if minutes > args.max_minutes_before_start:
            continue
        market = (record.get("forecast") or {}).get("market_benchmark") or {}
        books = market.get("books") or []
        # Keep collecting until the record reaches a 3-book consensus, while
        # merging snapshots by sportsbook and retaining each book's latest time.
        if market.get("status") == "CONSENSUS" and int(market.get("book_count") or 0) >= 3:
            continue
        candidates.append((path, record, start))
    print(json.dumps({"eligible_games": len(candidates), "estimated_credits_at_10_each": len(candidates)*CREDITS_PER_GAME,
                      "api_calls_enabled": args.max_credits > 0, "as_of": now.isoformat()}, indent=2))
    if args.max_credits <= 0:
        return
    if not api_key:
        raise SystemExit("PUCKAPI_API_KEY is missing; no market requests were sent")
    if len(candidates)*CREDITS_PER_GAME > args.max_credits:
        raise SystemExit(f"Eligible request estimate exceeds max-credits={args.max_credits}; no market requests were sent")

    updated = []
    for path, record, start in candidates:
        try:
            payload = fetch(str(record.get("game_id")), api_key)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            market = (record.get("forecast") or {}).get("market_benchmark") or {
                "status": "RETRYABLE_UNAVAILABLE", "source": "PuckAPI", "book_count": 0, "books": [],
            }
            market["last_request_error_type"] = type(exc).__name__
            market["last_request_at"] = now.astimezone(timezone.utc).isoformat()
            record.setdefault("forecast", {})["market_benchmark"] = market
            path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            updated.append({"game_id": record.get("game_id"), "status": "RETRYABLE_UNAVAILABLE"})
            continue
        previous = ((record.get("forecast") or {}).get("market_benchmark") or {}).get("books") or []
        previous = [b for b in previous if b.get("sportsbook") and parse_dt(b.get("odds_timestamp")) and
                    parse_dt(b.get("odds_timestamp")).astimezone(timezone.utc) < start.astimezone(timezone.utc) and
                    parse_dt(b.get("odds_timestamp")).astimezone(timezone.utc) <= now.astimezone(timezone.utc)]
        market = normalize(str(record.get("game_id")), start, payload, now)
        merged_books = {str(b["sportsbook"]): b for b in previous if b.get("sportsbook")}
        for book in market["books"]:
            old = merged_books.get(book["sportsbook"])
            if old is None or book["odds_timestamp"] > old.get("odds_timestamp", ""):
                merged_books[book["sportsbook"]] = book
        market["books"] = sorted(merged_books.values(), key=lambda b: b["sportsbook"])
        market["books"] = [
            {**b, "sportsbook": canonical_sportsbook(b.get("sportsbook")),
             "consensus_eligible": canonical_sportsbook(b.get("sportsbook")) in PRIMARY_BOOKS}
            for b in market["books"]
        ]
        probabilities = [b["home_no_vig_probability"] for b in market["books"] if b.get("consensus_eligible")]
        count = len(probabilities)
        median_home = float(median(probabilities)) if count else None
        market.update({
            "book_count": count,
            "total_book_count": len(market["books"]),
            "status": "CONSENSUS" if count >= 3 else ("LIMITED_CONSENSUS" if count == 2 else ("SINGLE_BOOK" if count == 1 else "RETRYABLE_UNAVAILABLE")),
            "home_implied_probability": median_home,
            "away_implied_probability": None if median_home is None else 1.0-median_home,
            "home_probability_min": min(probabilities) if probabilities else None,
            "home_probability_max": max(probabilities) if probabilities else None,
            "home_probability_std": (sum((p-sum(probabilities)/count)**2 for p in probabilities)/count)**0.5 if count else None,
            "earliest_closing_timestamp": min((b["odds_timestamp"] for b in market["books"] if b.get("consensus_eligible")), default=None),
            "latest_closing_timestamp": max((b["odds_timestamp"] for b in market["books"] if b.get("consensus_eligible")), default=None),
        })
        # A final gate prevents even a source-side timestamp from crossing puck drop.
        start_utc = start.astimezone(timezone.utc)
        market["books"] = [b for b in market["books"] if parse_dt(b["odds_timestamp"]).astimezone(timezone.utc) < start_utc]
        probabilities = [b["home_no_vig_probability"] for b in market["books"] if b.get("consensus_eligible")]
        count = len(probabilities)
        median_home = float(median(probabilities)) if count else None
        market.update({
            "book_count": count,
            "total_book_count": len(market["books"]),
            "status": "CONSENSUS" if count >= 3 else ("LIMITED_CONSENSUS" if count == 2 else ("SINGLE_BOOK" if count == 1 else "RETRYABLE_UNAVAILABLE")),
            "home_implied_probability": median_home,
            "away_implied_probability": None if median_home is None else 1.0-median_home,
            "home_probability_min": min(probabilities) if probabilities else None,
            "home_probability_max": max(probabilities) if probabilities else None,
            "home_probability_std": (sum((p-sum(probabilities)/count)**2 for p in probabilities)/count)**0.5 if count else None,
            "earliest_closing_timestamp": min((b["odds_timestamp"] for b in market["books"] if b.get("consensus_eligible")), default=None),
            "latest_closing_timestamp": max((b["odds_timestamp"] for b in market["books"] if b.get("consensus_eligible")), default=None),
        })
        record.setdefault("forecast", {})["market_benchmark"] = market
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        updated.append({"game_id": record.get("game_id"), "status": market["status"], "book_count": market["book_count"]})
    print(json.dumps({"updated": updated}, indent=2))


if __name__ == "__main__":
    main()
