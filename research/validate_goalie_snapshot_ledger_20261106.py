"""Validate an append-only JSONL pregame goalie snapshot ledger using stdlib only."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = {
    "schema_version", "game_id", "team_abbr", "scheduled_start_utc",
    "captured_at_utc", "assignment_state", "goalie", "source",
    "source_observed_at_utc", "capture_run_id", "forecast_version",
}
VALID_STATES = {"projected", "confirmed", "unassigned", "unavailable"}
FORBIDDEN = {
    "actual_starter", "final_score", "reg_home_goals", "reg_away_goals",
    "saves", "goals_allowed", "result", "closing_odds", "market_probability",
}

def parse_utc(value: str, field: str, row_no: int) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception as exc:
        raise ValueError(f"row {row_no}: {field} is not ISO-8601") from exc
    if dt.tzinfo is None or dt.utcoffset() != timezone.utc.utcoffset(dt):
        raise ValueError(f"row {row_no}: {field} must carry UTC timezone")
    return dt.astimezone(timezone.utc)

def validate(path: Path) -> int:
    latest: dict[tuple[int, str], tuple[datetime, str, str | None]] = {}
    rows = 0
    for row_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"row {row_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"row {row_no}: expected JSON object")
        missing = REQUIRED - row.keys()
        extra = row.keys() - REQUIRED - {"source_confidence"}
        if missing:
            raise ValueError(f"row {row_no}: missing fields {sorted(missing)}")
        if extra:
            raise ValueError(f"row {row_no}: unsupported fields {sorted(extra)}")
        forbidden = FORBIDDEN.intersection(row)
        if forbidden:
            raise ValueError(f"row {row_no}: outcome/market leakage fields {sorted(forbidden)}")
        if row["schema_version"] != 1:
            raise ValueError(f"row {row_no}: unsupported schema_version")
        if row["assignment_state"] not in VALID_STATES:
            raise ValueError(f"row {row_no}: invalid assignment_state")
        start = parse_utc(row["scheduled_start_utc"], "scheduled_start_utc", row_no)
        captured = parse_utc(row["captured_at_utc"], "captured_at_utc", row_no)
        if captured >= start:
            raise ValueError(f"row {row_no}: snapshot was not captured before puck drop")
        observed = row["source_observed_at_utc"]
        if observed is not None:
            parse_utc(observed, "source_observed_at_utc", row_no)
        state, goalie = row["assignment_state"], row["goalie"]
        if state in {"projected", "confirmed"}:
            if not isinstance(goalie, dict) or not goalie.get("player_id") or not goalie.get("name"):
                raise ValueError(f"row {row_no}: {state} requires goalie player_id and name")
        elif goalie is not None:
            raise ValueError(f"row {row_no}: {state} requires goalie=null")
        key = (int(row["game_id"]), str(row["team_abbr"]))
        prior = latest.get(key)
        if prior and captured <= prior[0]:
            raise ValueError(f"row {row_no}: duplicate or out-of-order snapshot for {key}")
        signature = None if goalie is None else str(goalie["player_id"])
        if prior and prior[1] == state and prior[2] == signature:
            raise ValueError(f"row {row_no}: duplicate unchanged state; do not append")
        latest[key] = (captured, state, signature)
        rows += 1
    if rows == 0:
        raise ValueError("ledger is empty")
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path, help="Append-only JSONL snapshot ledger")
    args = parser.parse_args()
    count = validate(args.ledger)
    print(f"VALID: {count} pregame snapshots across {len(set_key(args.ledger))} game-team pairs")

def set_key(path: Path) -> set[tuple[int, str]]:
    keys=set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row=json.loads(line)
            keys.add((int(row["game_id"]), str(row["team_abbr"])))
    return keys

if __name__ == "__main__":
    main()
