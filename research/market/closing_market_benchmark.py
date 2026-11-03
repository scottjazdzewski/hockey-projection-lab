#!/usr/bin/env python3
"""Build an isolated Hockey Lab vs. closing-market research benchmark.

This module consumes frozen/chronological out-of-sample Hockey Lab forecasts,
canonical NHL games/results, and timestamped sportsbook prices. It never writes
to production model inputs or outputs.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    from .market_math import PRIMARY_BOOKS, canonical_sportsbook, no_vig_probabilities
except ImportError:  # direct CLI execution
    from market_math import PRIMARY_BOOKS, canonical_sportsbook, no_vig_probabilities


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "market"
LINE_COLUMNS = [
    "game_id", "game_date", "scheduled_start_utc", "sportsbook",
    "away_moneyline", "home_moneyline", "odds_timestamp",
    "minutes_before_puck_drop", "source", "provider_event_id",
    "snapshot_type", "away_no_vig_probability", "home_no_vig_probability",
    "consensus_eligible", "sportsbook_source_name",
]
TEAM_ALIASES = {
    "ANAHEIM DUCKS": "ANA", "ARIZONA COYOTES": "ARI", "BOSTON BRUINS": "BOS",
    "BUFFALO SABRES": "BUF", "CALGARY FLAMES": "CGY", "CAROLINA HURRICANES": "CAR",
    "CHICAGO BLACKHAWKS": "CHI", "COLORADO AVALANCHE": "COL", "COLUMBUS BLUE JACKETS": "CBJ",
    "DALLAS STARS": "DAL", "DETROIT RED WINGS": "DET", "EDMONTON OILERS": "EDM",
    "FLORIDA PANTHERS": "FLA", "LOS ANGELES KINGS": "LAK", "LA KINGS": "LAK", "MINNESOTA WILD": "MIN",
    "MONTREAL CANADIENS": "MTL", "NASHVILLE PREDATORS": "NSH", "NEW JERSEY DEVILS": "NJD",
    "NEW YORK ISLANDERS": "NYI", "NEW YORK RANGERS": "NYR", "OTTAWA SENATORS": "OTT",
    "PHILADELPHIA FLYERS": "PHI", "PITTSBURGH PENGUINS": "PIT", "SAN JOSE SHARKS": "SJS",
    "SEATTLE KRAKEN": "SEA", "ST LOUIS BLUES": "STL", "ST. LOUIS BLUES": "STL",
    "TAMPA BAY LIGHTNING": "TBL", "TORONTO MAPLE LEAFS": "TOR", "UTAH HOCKEY CLUB": "UTA",
    "UTAH MAMMOTH": "UTA", "VANCOUVER CANUCKS": "VAN", "VEGAS GOLDEN KNIGHTS": "VGK",
    "WASHINGTON CAPITALS": "WSH", "WINNIPEG JETS": "WPG",
}
TEAM_CODES = {
    "ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL",
    "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR",
    "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN",
    "VGK", "WPG", "WSH",
}
PROBABILITY_BUCKETS = np.linspace(0.0, 1.0, 21)


def _first_col(frame: pd.DataFrame, names: list[str], required: bool = True) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(f"Required column missing; expected one of {names}")
    return None


def _read_frame(path: Path) -> pd.DataFrame:
    if path.is_dir():
        rows = []
        for file in sorted(path.glob("*.json")):
            record = json.loads(file.read_text(encoding="utf-8"))
            forecast = record.get("forecast") or {}
            matchup = record.get("matchup") or {}
            model = forecast.get("model") or {}
            away = forecast.get("away") or {}
            home = forecast.get("home") or {}
            rest = forecast.get("rest_state") or {}
            goalie = forecast.get("goalie_context") or {}
            goalie_statuses = [
                (goalie.get(side) or {}).get("status") for side in ("away", "home")
                if isinstance(goalie.get(side), dict)
            ]
            flags = {
                side: {str(v).upper() for v in (rest.get(f"{side}_flags") or [])}
                for side in ("away", "home")
            }
            def has_load(side, label):
                return label in flags[side]
            def is_b2b(side):
                severity = str(rest.get(f"{side}_severity") or "").upper()
                return severity == "B2B" or "B2B" in flags[side]
            rows.append({
                "game_id": record.get("game_id"),
                "model_id": model.get("model_id"),
                "model_version": model.get("version"),
                "feature_version": model.get("feature_version"),
                "away_win_probability": away.get("win_probability"),
                "home_win_probability": home.get("win_probability"),
                "frozen_at": record.get("frozen_at"),
                "forecast_origin": "FROZEN_PREGAME",
                "home_abbr": matchup.get("home_abbr"),
                "away_abbr": matchup.get("away_abbr"),
                "goalie_state": "CONFIRMED" if len(goalie_statuses) == 2 and all(str(v).upper() == "CONFIRMED" for v in goalie_statuses) else ("PARTIAL" if any(str(v).upper() == "CONFIRMED" for v in goalie_statuses) else "UNKNOWN"),
                "rest_state": rest,
                "rest_home_severity": rest.get("home_severity"),
                "rest_away_severity": rest.get("away_severity"),
                "home_b2b": is_b2b("home"), "away_b2b": is_b2b("away"),
                "any_b2b": is_b2b("home") or is_b2b("away"),
                "home_3in4": has_load("home", "3-IN-4"), "away_3in4": has_load("away", "3-IN-4"),
                "any_3in4": has_load("home", "3-IN-4") or has_load("away", "3-IN-4"),
                "home_4in6": has_load("home", "4-IN-6"), "away_4in6": has_load("away", "4-IN-6"),
                "any_4in6": has_load("home", "4-IN-6") or has_load("away", "4-IN-6"),
                "home_5in8": has_load("home", "5-IN-8"), "away_5in8": has_load("away", "5-IN-8"),
                "any_5in8": has_load("home", "5-IN-8") or has_load("away", "5-IN-8"),
                "game_date": record.get("game_date"),
            })
        return pd.DataFrame(rows)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
                return pd.DataFrame(obj["rows"])
            return pd.json_normalize(obj)
        except json.JSONDecodeError:
            return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported input format: {path}")


def _utc(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def _team_key(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()
    text = re.sub(r"\s+", " ", text)
    if text in TEAM_CODES:
        return text
    direct = TEAM_ALIASES.get(text)
    if direct:
        return direct
    # Current short-name variants used by odds feeds.
    short = {
        "ANAHEIM": "ANA", "ARIZONA": "ARI", "BOSTON": "BOS", "BUFFALO": "BUF",
        "CALGARY": "CGY", "CAROLINA": "CAR", "CHICAGO": "CHI", "COLORADO": "COL",
        "COLUMBUS": "CBJ", "DALLAS": "DAL", "DETROIT": "DET", "EDMONTON": "EDM",
        "FLORIDA": "FLA", "LOS ANGELES": "LAK", "LA KINGS": "LAK", "MINNESOTA": "MIN", "MONTREAL": "MTL",
        "NASHVILLE": "NSH", "NEW JERSEY": "NJD", "NY ISLANDERS": "NYI", "NY RANGERS": "NYR",
        "OTTAWA": "OTT", "PHILADELPHIA": "PHI", "PITTSBURGH": "PIT", "SAN JOSE": "SJS",
        "SEATTLE": "SEA", "ST LOUIS": "STL", "TAMPA BAY": "TBL", "TORONTO": "TOR",
        "UTAH": "UTA", "VANCOUVER": "VAN", "VEGAS": "VGK", "WASHINGTON": "WSH",
        "WINNIPEG": "WPG",
    }
    return short.get(text)


def _canonical_games(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    gid = _first_col(x, ["game_id", "nhl_game_id"])
    date = _first_col(x, ["game_date", "date"])
    start = _first_col(x, ["scheduled_start_utc", "scheduled_start", "start_time_utc", "game_start_utc"])
    home = _first_col(x, ["home_team_abbr", "home_abbr", "home_team"])
    away = _first_col(x, ["away_team_abbr", "away_abbr", "away_team"])
    x["game_id"] = x[gid].astype("string")
    x["game_date"] = pd.to_datetime(x[date], errors="coerce").dt.date.astype("string")
    x["scheduled_start_utc"] = x[start].map(_utc)
    x["home_abbr"] = x[home].map(_team_key)
    x["away_abbr"] = x[away].map(_team_key)
    if x[["game_id", "game_date", "scheduled_start_utc", "home_abbr", "away_abbr"]].isna().any().any():
        raise ValueError("Canonical NHL games require unique IDs, exact dates, scheduled starts, and recognized home/away teams")
    if x["game_id"].duplicated().any():
        dupes = x.loc[x["game_id"].duplicated(False), "game_id"].head(10).tolist()
        raise ValueError(f"Canonical games must be unique by game_id; duplicates include {dupes}")
    win = _first_col(x, ["winner_abbr", "winner_team_abbr"], required=False)
    home_win = _first_col(x, ["home_win", "home_team_win"], required=False)
    if win:
        x["winner_abbr"] = x[win].map(_team_key)
    elif home_win:
        flag = pd.to_numeric(x[home_win], errors="coerce")
        x["winner_abbr"] = np.where(flag.eq(1), x["home_abbr"], np.where(flag.eq(0), x["away_abbr"], None))
    else:
        x["winner_abbr"] = None
    if "source_season" not in x:
        x["source_season"] = pd.NA
    return x


def _canonical_forecasts(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["game_id"] = x[_first_col(x, ["game_id", "nhl_game_id"])].astype("string")
    away = _first_col(x, ["away_win_probability", "p_away", "away_probability"])
    home = _first_col(x, ["home_win_probability", "p_home", "home_probability"])
    x["hl_away_probability"] = pd.to_numeric(x[away], errors="coerce")
    x["hl_home_probability"] = pd.to_numeric(x[home], errors="coerce")
    for target, options in {
        "model_id": ["model_id"], "model_version": ["model_version", "version"],
        "feature_version": ["feature_version"], "frozen_at": ["frozen_at"],
        "forecast_origin": ["forecast_origin"],
    }.items():
        src = _first_col(x, options, required=False)
        if src is None:
            x[target] = None
        elif src != target:
            x[target] = x[src]
    cutoff_col = _first_col(x, ["training_cutoff_utc", "max_training_date", "training_cutoff"], required=False)
    max_season_col = _first_col(x, ["training_max_source_season", "max_training_season", "trained_through_source_season"], required=False)
    x["training_cutoff_utc"] = x[cutoff_col].map(_utc) if cutoff_col else pd.NaT
    x["training_max_source_season"] = pd.to_numeric(x[max_season_col], errors="coerce") if max_season_col else np.nan
    x["frozen_at_utc"] = x["frozen_at"].map(_utc)
    inferred_origin = pd.Series("UNVERIFIED", index=x.index, dtype="string")
    x["forecast_origin"] = x["forecast_origin"].where(x["forecast_origin"].notna(), inferred_origin)
    return x


def _line_match(lines: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    x = lines.copy()
    audits: list[dict] = []
    gid_col = _first_col(x, ["game_id", "nhl_game_id"], required=False)
    if gid_col:
        x["game_id"] = x[gid_col].astype("string")
        known = set(games["game_id"].dropna().astype(str))
        direct = x[x["game_id"].astype(str).isin(known)].copy()
        unknown = x[~x["game_id"].astype(str).isin(known)].copy()
        if unknown.empty:
            return direct, audits
        # Some feeds use a generic event UUID in a column called game_id. If
        # the source also supplies exact schedule/team identity, use that
        # conservative join rather than mistaking the UUID for an NHL ID.
        residual = unknown.drop(columns=["game_id", "nhl_game_id"], errors="ignore")
        has_exact_identity = (
            _first_col(residual, ["scheduled_start_utc", "scheduled_start", "commence_time"], required=False)
            and _first_col(residual, ["home_team", "home_team_name", "home_abbr"], required=False)
            and _first_col(residual, ["away_team", "away_team_name", "away_abbr"], required=False)
        )
        if has_exact_identity:
            fallback, fallback_audit = _line_match(residual, games)
            audits.extend(fallback_audit)
            matched_indexes = set(fallback.index)
            for idx, row in unknown.iterrows():
                if idx not in matched_indexes and not any(a.get("row") == int(idx) for a in fallback_audit):
                    audits.append({"row": int(idx), "reason": "unknown_nhl_game_id", "source_game_id": str(row["game_id"])})
            return pd.concat([direct, fallback], axis=0).sort_index(), audits
        for idx, row in unknown.iterrows():
            audits.append({"row": int(idx), "reason": "unknown_nhl_game_id", "source_game_id": str(row["game_id"])})
        return direct, audits

    date_col = _first_col(x, ["game_date", "date"], required=False)
    start_col = _first_col(x, ["scheduled_start_utc", "scheduled_start", "commence_time"])
    home_col = _first_col(x, ["home_team", "home_team_name", "home_abbr"])
    away_col = _first_col(x, ["away_team", "away_team_name", "away_abbr"])
    x["_start"] = x[start_col].map(_utc)
    if date_col:
        x["_date"] = pd.to_datetime(x[date_col], errors="coerce").dt.date.astype("string")
        missing_date = x["_date"].isna()
        local_date = x["_start"].dt.tz_convert("America/New_York").dt.date.astype("string")
        x.loc[missing_date, "_date"] = local_date.loc[missing_date]
    else:
        # Derive the source event date from its UTC scheduled timestamp in
        # NHL's primary game-date timezone when the provider omits a date field.
        x["_date"] = x["_start"].dt.tz_convert("America/New_York").dt.date.astype("string")
    x["_home"] = x[home_col].map(_team_key)
    x["_away"] = x[away_col].map(_team_key)
    lookup: dict[tuple, list[str]] = {}
    for g in games.itertuples(index=False):
        key = (str(g.game_date), g.scheduled_start_utc, g.home_abbr, g.away_abbr)
        lookup.setdefault(key, []).append(str(g.game_id))
    matched = []
    for idx, row in x.iterrows():
        key = (str(row["_date"]), row["_start"], row["_home"], row["_away"])
        ids = lookup.get(key, [])
        if len(ids) == 1:
            matched.append((idx, ids[0]))
        else:
            audits.append({
                "row": int(idx),
                "reason": "no_exact_match" if not ids else "ambiguous_exact_match",
                "source_event_id": row.get("source_event_id", row.get("event_id")),
                "game_date": row["_date"], "scheduled_start_utc": str(row["_start"]),
                "home_team": row.get(home_col), "away_team": row.get(away_col),
                "candidate_game_ids": ids,
            })
    id_map = {idx: gid for idx, gid in matched}
    x["game_id"] = [id_map.get(idx) for idx in x.index]
    return x[x.game_id.notna()].copy(), audits


def _line_rows(lines: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    x, match_audit = _line_match(lines, games)
    required = {
        "sportsbook": _first_col(x, ["sportsbook", "bookmaker", "book"]),
        "away_ml": _first_col(x, ["away_moneyline", "ml_away", "mlAway", "away_ml"]),
        "home_ml": _first_col(x, ["home_moneyline", "ml_home", "mlHome", "home_ml"]),
        "timestamp": _first_col(x, ["odds_timestamp", "captured_at", "capturedAt", "timestamp"]),
        "source": _first_col(x, ["source", "provider"], required=False),
    }
    x["sportsbook_source_name"] = x[required["sportsbook"]].astype("string").str.strip()
    x["sportsbook"] = x["sportsbook_source_name"].map(canonical_sportsbook)
    x["away_moneyline"] = pd.to_numeric(x[required["away_ml"]], errors="coerce")
    x["home_moneyline"] = pd.to_numeric(x[required["home_ml"]], errors="coerce")
    x["odds_timestamp"] = x[required["timestamp"]].map(_utc)
    x["source"] = x[required["source"]] if required["source"] else "UNSPECIFIED"
    start_lookup = games.set_index("game_id")["scheduled_start_utc"]
    date_lookup = games.set_index("game_id")["game_date"]
    x["scheduled_start_utc"] = x["game_id"].map(start_lookup)
    x["game_date"] = x["game_id"].map(date_lookup)
    x["minutes_before_puck_drop"] = (x["scheduled_start_utc"] - x["odds_timestamp"]).dt.total_seconds() / 60.0
    x["snapshot_type"] = x.get("snapshot_type", x.get("snapshotType"))
    if "provider_event_id" not in x:
        x["provider_event_id"] = x.get("source_event_id", x.get("event_id"))

    rejected: list[dict] = []
    good = []
    for idx, row in x.iterrows():
        reason = None
        if not row["sportsbook"]:
            reason = "missing_sportsbook"
        elif pd.isna(row["scheduled_start_utc"]) or pd.isna(row["odds_timestamp"]):
            reason = "missing_schedule_or_odds_timestamp"
        elif row["odds_timestamp"] >= row["scheduled_start_utc"]:
            reason = "timestamp_at_or_after_scheduled_start"
        elif pd.isna(row["away_moneyline"]) or pd.isna(row["home_moneyline"]):
            reason = "missing_two_way_moneyline"
        elif row["away_moneyline"] == 0 or row["home_moneyline"] == 0:
            reason = "invalid_zero_moneyline"
        if reason:
            rejected.append({
                "row": int(idx), "game_id": str(row["game_id"]), "sportsbook": row["sportsbook"],
                "odds_timestamp": str(row["odds_timestamp"]),
                "scheduled_start_utc": str(row["scheduled_start_utc"]), "reason": reason,
            })
            continue
        try:
            away_p, home_p = no_vig_probabilities(row["away_moneyline"], row["home_moneyline"])
        except (ValueError, TypeError, OverflowError):
            rejected.append({"row": int(idx), "game_id": str(row["game_id"]), "sportsbook": row["sportsbook"], "reason": "invalid_moneyline"})
            continue
        out = row.to_dict()
        out["away_no_vig_probability"] = away_p
        out["home_no_vig_probability"] = home_p
        out["consensus_eligible"] = row["sportsbook"] in PRIMARY_BOOKS
        good.append(out)

    valid = pd.DataFrame(good)
    if valid.empty:
        valid = pd.DataFrame(columns=LINE_COLUMNS)
        return valid, match_audit + rejected, {"input_rows": int(len(lines)), "valid_rows": 0, "rejected_rows": len(rejected)}

    # If duplicate copies of a closing capture exist, retain the latest timestamp.
    # Conflicting duplicates at that timestamp are quarantined rather than guessed.
    selected = []
    key_cols = ["game_id", "sportsbook"]
    for (gid, book), group in valid.groupby(key_cols, dropna=False):
        latest = group["odds_timestamp"].max()
        latest_rows = group[group["odds_timestamp"].eq(latest)]
        pairs = latest_rows[["away_moneyline", "home_moneyline"]].drop_duplicates()
        if len(pairs) != 1:
            rejected.append({"game_id": str(gid), "sportsbook": str(book), "odds_timestamp": str(latest), "reason": "conflicting_same_time_lines"})
            continue
        selected.append(latest_rows.iloc[0].to_dict())
    valid = pd.DataFrame(selected)
    for col in LINE_COLUMNS:
        if col not in valid:
            valid[col] = None
    valid = valid[LINE_COLUMNS].sort_values(["game_date", "game_id", "sportsbook"]).reset_index(drop=True)
    by_book = {}
    for book, group in valid.groupby("sportsbook", dropna=False):
        by_book[str(book)] = {
            "games": int(group.game_id.nunique()), "lines": int(len(group)),
            "consensus_eligible": bool(group.consensus_eligible.fillna(False).any()),
            "first_odds_timestamp": group.odds_timestamp.min().isoformat(),
            "last_odds_timestamp": group.odds_timestamp.max().isoformat(),
        }
    by_source = {}
    for source, group in valid.groupby("source", dropna=False):
        by_source[str(source)] = {"games": int(group.game_id.nunique()), "lines": int(len(group))}
    summary = {
        "input_rows": int(len(lines)), "valid_rows": int(len(valid)),
        "rejected_rows": int(len(rejected)),
        "sportsbooks": sorted(valid.sportsbook.dropna().astype(str).unique().tolist()),
        "sportsbook_coverage": by_book,
        "source_coverage": by_source,
        "first_odds_timestamp": None if valid.empty else valid.odds_timestamp.min().isoformat(),
        "last_odds_timestamp": None if valid.empty else valid.odds_timestamp.max().isoformat(),
    }
    return valid, match_audit + rejected, summary


def _consensus(lines: pd.DataFrame, games: pd.DataFrame, min_books: int) -> pd.DataFrame:
    by_game = {}
    if not lines.empty:
        included = lines[lines.consensus_eligible.fillna(False)].copy() if "consensus_eligible" in lines else lines
        for gid, group in included.groupby("game_id"):
            ps = group["home_no_vig_probability"].astype(float).to_numpy()
            count = int(len(ps))
            status = "CONSENSUS" if count >= min_books else ("LIMITED_CONSENSUS" if count == 2 else ("SINGLE_BOOK" if count == 1 else "UNAVAILABLE"))
            by_game[str(gid)] = {
                "market_home_probability": float(np.median(ps)),
                "market_away_probability": float(1.0 - np.median(ps)),
                "book_count": count,
                "consensus_status": status,
                "home_probability_min": float(np.min(ps)),
                "home_probability_max": float(np.max(ps)),
                "home_probability_std": float(np.std(ps, ddof=0)),
                "earliest_closing_timestamp": group.odds_timestamp.min().isoformat(),
                "latest_closing_timestamp": group.odds_timestamp.max().isoformat(),
                "sportsbooks": sorted(group.sportsbook.astype(str).tolist()),
                "source_list": sorted(set(map(str, group.source.dropna().tolist()))),
            }
    rows = []
    for r in games.itertuples(index=False):
        item = by_game.get(str(r.game_id), {})
        rows.append({
            "game_id": str(r.game_id), "game_date": r.game_date,
            "scheduled_start_utc": r.scheduled_start_utc,
            "home_abbr": r.home_abbr, "away_abbr": r.away_abbr,
            "source_season": getattr(r, "source_season", None),
            "market_home_probability": item.get("market_home_probability"),
            "market_away_probability": item.get("market_away_probability"),
            "book_count": int(item.get("book_count", 0)),
            "consensus_status": item.get("consensus_status", "UNAVAILABLE"),
            "home_probability_min": item.get("home_probability_min"),
            "home_probability_max": item.get("home_probability_max"),
            "home_probability_std": item.get("home_probability_std"),
            "earliest_closing_timestamp": item.get("earliest_closing_timestamp"),
            "latest_closing_timestamp": item.get("latest_closing_timestamp"),
            "sportsbooks": item.get("sportsbooks", []),
            "source_list": item.get("source_list", []),
        })
    return pd.DataFrame(rows)


def _safe_mean(values) -> float | None:
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return None if s.empty else float(s.mean())


def _score(frame: pd.DataFrame, p_home_col: str, y_home_col: str = "home_win_actual") -> dict:
    d = frame[[p_home_col, y_home_col]].copy()
    d[p_home_col] = pd.to_numeric(d[p_home_col], errors="coerce")
    d[y_home_col] = pd.to_numeric(d[y_home_col], errors="coerce")
    d = d[d[p_home_col].between(0, 1) & d[y_home_col].isin([0, 1])]
    if d.empty:
        return {"games": 0, "accuracy": None, "brier": None, "log_loss": None,
                "calibration": {"mean_predicted_home_win_rate": None, "actual_home_win_rate": None, "ece": None},
                "average_probability_assigned_to_winner": None}
    p = d[p_home_col].to_numpy(float)
    y = d[y_home_col].to_numpy(float)
    eps = 1e-15
    ll = -(y * np.log(np.clip(p, eps, 1-eps)) + (1-y) * np.log(np.clip(1-p, eps, 1-eps)))
    acc = ((p >= .5) == (y == 1)).mean()
    bins = pd.cut(p, PROBABILITY_BUCKETS, right=False, include_lowest=True, labels=False)
    # Include p=1.0 in the final bucket.
    bins = np.where(np.isnan(bins), len(PROBABILITY_BUCKETS)-2, bins).astype(int)
    cal_rows = []
    ece = 0.0
    for b in sorted(set(bins.tolist())):
        ix = bins == b
        pred = float(p[ix].mean())
        actual = float(y[ix].mean())
        n = int(ix.sum())
        ece += n / len(y) * abs(pred - actual)
        cal_rows.append({"bucket": f"{PROBABILITY_BUCKETS[b]:.2f}-{PROBABILITY_BUCKETS[b+1]:.2f}",
                         "games": n, "mean_predicted_home_probability": pred,
                         "actual_home_win_rate": actual, "calibration_error": pred-actual})
    winner_p = np.where(y == 1, p, 1-p)
    return {
        "games": int(len(d)), "accuracy": float(acc), "brier": float(np.mean((p-y)**2)),
        "log_loss": float(np.mean(ll)),
        "calibration": {"mean_predicted_home_win_rate": float(p.mean()),
                        "actual_home_win_rate": float(y.mean()), "ece": float(ece),
                        "home_probability_bins": cal_rows},
        "average_probability_assigned_to_winner": float(winner_p.mean()),
    }


def _breakdown(frame: pd.DataFrame, key: str, label: str | None = None) -> dict:
    if key not in frame.columns:
        return {}
    out = {}
    for value, g in frame.groupby(key, dropna=False, sort=True):
        k = "UNKNOWN" if pd.isna(value) else str(value)
        out[k] = {
            "games": int(len(g)),
            "hockey_lab": _score(g, "hl_home_probability"),
            "market": _score(g, "market_home_probability"),
        }
    return out


def _disagreement_report(frame: pd.DataFrame) -> dict:
    d = frame[frame.consensus_status.eq("CONSENSUS") & frame.home_win_actual.notna()].copy()
    d["hl_pick_home"] = d.hl_home_probability.ge(.5)
    d["market_pick_home"] = d.market_home_probability.ge(.5)
    disagree = d[d.hl_pick_home.ne(d.market_pick_home)].copy()
    disagree["magnitude_pp"] = (disagree.hl_home_probability - disagree.market_home_probability).abs() * 100
    edges = [0, 3, 5, 10, 15, np.inf]
    labels = ["<3 pp", "3-5 pp", "5-10 pp", "10-15 pp", "15+ pp"]
    disagree["magnitude_bucket"] = pd.cut(disagree.magnitude_pp, edges, right=False, labels=labels, include_lowest=True)
    if len(disagree):
        home_actual = disagree.home_win_actual.eq(1)
        disagree["hl_correct"] = disagree.hl_pick_home.eq(home_actual)
        disagree["market_correct"] = disagree.market_pick_home.eq(home_actual)
    by_mag = {}
    for label, g in disagree.groupby("magnitude_bucket", observed=False):
        if len(g):
            by_mag[str(label)] = {"games": int(len(g)), "hockey_lab_wins": int(g.hl_correct.sum()),
                                  "market_wins": int(g.market_correct.sum()),
                                  "hockey_lab_accuracy": float(g.hl_correct.mean()),
                                  "market_accuracy": float(g.market_correct.mean())}
    home_bull = d[(d.hl_home_probability - d.market_home_probability) > 0].copy()
    road_bull = d[(d.hl_home_probability - d.market_home_probability) < 0].copy()
    conf = d.assign(confidence_diff=(d[["hl_home_probability", "hl_away_probability"]].max(axis=1) -
                                    d[["market_home_probability", "market_away_probability"]].max(axis=1)))
    def side_summary(g):
        if g.empty:
            return {"games": 0, "home_probability_difference_pp_mean": None,
                    "hockey_lab_accuracy": None, "market_accuracy": None}
        actual_home = g.home_win_actual.eq(1)
        return {"games": int(len(g)),
                "home_probability_difference_pp_mean": float((g.hl_home_probability-g.market_home_probability).mean()*100),
                "hockey_lab_accuracy": float((g.hl_home_probability.ge(.5).eq(actual_home)).mean()),
                "market_accuracy": float((g.market_home_probability.ge(.5).eq(actual_home)).mean())}
    return {
        "games": int(len(disagree)),
        "hockey_lab_wins": int(disagree.hl_correct.sum()) if len(disagree) else 0,
        "market_wins": int(disagree.market_correct.sum()) if len(disagree) else 0,
        "hockey_lab_accuracy": float(disagree.hl_correct.mean()) if len(disagree) else None,
        "market_accuracy": float(disagree.market_correct.mean()) if len(disagree) else None,
        "by_probability_difference": by_mag,
        "hockey_lab_more_bullish_on_home": side_summary(home_bull),
        "hockey_lab_more_bullish_on_road": side_summary(road_bull),
        "material_confidence_difference": {
            "thresholds_pp": [5, 10],
            "more_confident": {str(t): {"games": int((conf.confidence_diff >= t/100).sum()),
                                         "average_confidence_difference_pp": _safe_mean(conf.loc[conf.confidence_diff >= t/100, "confidence_diff"])*100 if (conf.confidence_diff >= t/100).any() else None}
                                for t in [5, 10]},
            "less_confident": {str(t): {"games": int((conf.confidence_diff <= -t/100).sum()),
                                        "average_confidence_difference_pp": _safe_mean(conf.loc[conf.confidence_diff <= -t/100, "confidence_diff"])*100 if (conf.confidence_diff <= -t/100).any() else None}
                               for t in [5, 10]},
        },
        "road_pick_difference_vs_market": {
            "games": int(d.hl_pick_side.eq("away").sum()),
            "mean_hockey_lab_minus_market_away_probability_pp": _safe_mean(
                (d.loc[d.hl_pick_side.eq("away"), "hl_away_probability"] -
                 d.loc[d.hl_pick_side.eq("away"), "market_away_probability"]) * 100
            ),
        },
    }


def _road_tendency(frame: pd.DataFrame) -> dict:
    d = frame[frame.consensus_status.eq("CONSENSUS") & frame.home_win_actual.notna()].copy()
    if d.empty:
        return {"games": 0, "hockey_lab_away_pick_rate": None,
                "market_away_pick_rate": None, "away_pick_rate_difference_pp": None,
                "mean_away_probability_difference_pp": None,
                "hockey_lab_accuracy_when_picking_away": None,
                "market_accuracy_on_same_away_picks": None}
    hl_away = d.hl_away_probability.ge(.5)
    market_away = d.market_away_probability.ge(.5)
    actual_away = d.home_win_actual.eq(0)
    return {
        "games": int(len(d)),
        "hockey_lab_away_pick_rate": float(hl_away.mean()),
        "market_away_pick_rate": float(market_away.mean()),
        "away_pick_rate_difference_pp": float((hl_away.mean() - market_away.mean()) * 100),
        "mean_away_probability_difference_pp": float((d.hl_away_probability - d.market_away_probability).mean() * 100),
        "hockey_lab_accuracy_when_picking_away": float((hl_away[hl_away].eq(actual_away[hl_away])).mean()) if hl_away.any() else None,
        "market_accuracy_on_same_away_picks": float((market_away[hl_away].eq(actual_away[hl_away])).mean()) if hl_away.any() else None,
    }


def build(args) -> dict:
    games = _canonical_games(_read_frame(Path(args.games)))
    forecasts = _canonical_forecasts(_read_frame(Path(args.forecasts)))
    raw_lines = _read_frame(Path(args.lines))
    if forecasts.game_id.duplicated().any():
        dupes = forecasts.loc[forecasts.game_id.duplicated(False), "game_id"].head(10).tolist()
        raise ValueError(f"Forecast input must contain one chosen forecast per game; duplicates include {dupes}")
    start_by_game = games.set_index("game_id")["scheduled_start_utc"]
    forecasts["scheduled_start_utc"] = forecasts.game_id.map(start_by_game)
    season_by_game = games.set_index("game_id")["source_season"]
    if "source_season" not in forecasts:
        forecasts["source_season"] = forecasts.game_id.map(season_by_game)
    else:
        forecasts["source_season"] = forecasts["source_season"].fillna(forecasts.game_id.map(season_by_game))
    origin = forecasts.forecast_origin.fillna("").astype("string").str.upper()
    frozen_origin = origin.eq("FROZEN_PREGAME")
    frozen_valid = forecasts.frozen_at_utc.notna() & (forecasts.frozen_at_utc < forecasts.scheduled_start_utc)
    oos_origin = origin.eq("CHRONOLOGICAL_OOS")
    season_cutoff_valid = pd.to_numeric(forecasts.training_max_source_season, errors="coerce") < pd.to_numeric(forecasts.source_season, errors="coerce")
    time_cutoff_valid = forecasts.training_cutoff_utc.notna() & (forecasts.training_cutoff_utc < forecasts.scheduled_start_utc)
    oos_valid = season_cutoff_valid | time_cutoff_valid
    bad_mask = (
        forecasts.scheduled_start_utc.isna() |
        (frozen_origin & ~frozen_valid) |
        (oos_origin & ~oos_valid) |
        ~(frozen_origin | oos_origin) |
        ~forecasts.hl_home_probability.between(0, 1) |
        ~forecasts.hl_away_probability.between(0, 1) |
        ((forecasts.hl_home_probability + forecasts.hl_away_probability - 1).abs() > .001)
    )
    # Historical rows without a capture timestamp are accepted only when their
    # source explicitly labels them as chronological out-of-sample forecasts.
    bad_forecast = forecasts.loc[bad_mask].copy()
    forecast_audit = []
    for idx, row in bad_forecast.iterrows():
        if pd.isna(row.scheduled_start_utc):
            reason = "unknown_game_or_missing_scheduled_start"
        elif str(row.forecast_origin).upper() == "FROZEN_PREGAME" and not (
            pd.notna(row.frozen_at_utc) and row.frozen_at_utc < row.scheduled_start_utc
        ):
            reason = "frozen_forecast_not_timestamped_before_start"
        elif str(row.forecast_origin).upper() == "CHRONOLOGICAL_OOS" and not bool(oos_valid.loc[idx]):
            reason = "chronological_oos_cutoff_not_verified_before_game"
        elif str(row.forecast_origin).upper() not in {"FROZEN_PREGAME", "CHRONOLOGICAL_OOS"}:
            reason = "forecast_origin_unverified"
        else:
            reason = "invalid_or_non_complementary_probabilities"
        forecast_audit.append({"row": int(idx), "game_id": str(row.game_id), "reason": reason})
    forecasts = forecasts.loc[~bad_mask].copy()

    lines, line_audit, source_summary = _line_rows(raw_lines, games)
    consensus = _consensus(lines, games, args.minimum_books)
    select_game_cols = ["game_id", "game_date", "scheduled_start_utc", "home_abbr", "away_abbr",
                        "winner_abbr", "source_season"]
    game_cols = games[select_game_cols].copy()
    if "game_type" in games:
        game_cols["game_type"] = games["game_type"]
    merged = forecasts.merge(game_cols, on="game_id", how="left", suffixes=("", "_game"))
    merged = merged.merge(consensus, on="game_id", how="left", suffixes=("", "_market"))
    # Resolve duplicate fields from the canonical game and consensus records.
    for col in ["game_date", "scheduled_start_utc", "home_abbr", "away_abbr", "source_season"]:
        other = f"{col}_market"
        if other in merged:
            if col not in merged:
                merged[col] = merged[other]
            merged = merged.drop(columns=[other])
        game_other = f"{col}_game"
        if game_other in merged:
            merged = merged.drop(columns=[game_other])
    merged = merged.merge(games[["game_id", "winner_abbr"]], on="game_id", how="left", suffixes=("", "_result"))
    if "winner_abbr_result" in merged:
        merged["winner_abbr"] = merged["winner_abbr"].fillna(merged["winner_abbr_result"])
        merged = merged.drop(columns=["winner_abbr_result"])
    merged["home_win_actual"] = np.where(merged.winner_abbr.eq(merged.home_abbr), 1.0,
                                         np.where(merged.winner_abbr.eq(merged.away_abbr), 0.0, np.nan))
    merged["hl_pick"] = np.where(merged.hl_home_probability.ge(.5), merged.home_abbr, merged.away_abbr)
    merged["market_pick"] = np.where(merged.market_home_probability.notna(),
                                     np.where(merged.market_home_probability.ge(.5), merged.home_abbr, merged.away_abbr), None)
    merged["probability_difference_home"] = merged.hl_home_probability - merged.market_home_probability
    merged["probability_difference_away"] = merged.hl_away_probability - merged.market_away_probability
    merged["disagreement_magnitude_pp"] = (merged.probability_difference_home.abs() * 100)
    merged["hl_probability_assigned_to_winner"] = np.where(merged.home_win_actual.eq(1), merged.hl_home_probability,
                                                           np.where(merged.home_win_actual.eq(0), merged.hl_away_probability, np.nan))
    merged["market_probability_assigned_to_winner"] = np.where(merged.home_win_actual.eq(1), merged.market_home_probability,
                                                               np.where(merged.home_win_actual.eq(0), merged.market_away_probability, np.nan))
    merged["hl_correct"] = merged.hl_pick.eq(merged.winner_abbr).where(merged.winner_abbr.notna())
    merged["market_correct"] = merged.market_pick.eq(merged.winner_abbr).where(merged.winner_abbr.notna() & merged.market_home_probability.notna())
    merged["hl_brier"] = (merged.hl_home_probability - merged.home_win_actual) ** 2
    merged["market_brier"] = (merged.market_home_probability - merged.home_win_actual) ** 2
    merged["hl_log_loss"] = -np.log(merged.hl_probability_assigned_to_winner.clip(1e-15, 1-1e-15))
    merged["market_log_loss"] = -np.log(merged.market_probability_assigned_to_winner.clip(1e-15, 1-1e-15))

    # Rest/load/goalie metadata may be scalar columns or nested JSON from the
    # forward ledger. Keep unknowns visible instead of dropping those games.
    if "rest_state" in merged:
        def rest_field(v, field):
            return v.get(field) if isinstance(v, dict) else None
        for field in ["away", "home", "away_severity", "home_severity", "away_flags", "home_flags"]:
            col = f"rest_{field}"
            if col not in merged:
                merged[col] = merged.rest_state.map(lambda v: rest_field(v, field))
    merged["hl_pick_side"] = np.where(merged.hl_pick.eq(merged.home_abbr), "home", "away")
    merged["market_pick_side"] = np.where(merged.market_pick.eq(merged.home_abbr), "home", "away")
    merged["hl_picked_team_probability"] = merged[["hl_home_probability", "hl_away_probability"]].max(axis=1)
    merged["market_favorite_probability"] = merged[["market_home_probability", "market_away_probability"]].max(axis=1)
    merged["confidence_difference_pp"] = (merged.hl_picked_team_probability - merged.market_favorite_probability) * 100
    merged["hl_underdog_probability"] = merged[["hl_home_probability", "hl_away_probability"]].min(axis=1)
    merged["market_underdog_probability"] = merged[["market_home_probability", "market_away_probability"]].min(axis=1)
    merged["hl_favorite_probability"] = merged[["hl_home_probability", "hl_away_probability"]].max(axis=1)

    for col in ["hl_home_probability", "market_home_probability", "market_favorite_probability"]:
        bucket = {
            "hl_home_probability": "hl_home_probability_bucket",
            "market_home_probability": "market_home_probability_bucket",
            "market_favorite_probability": "market_favorite_bucket",
        }[col]
        merged[bucket] = pd.cut(merged[col], PROBABILITY_BUCKETS, right=False, include_lowest=True).astype("string")
    merged["hl_underdog_probability_bucket"] = pd.cut(
        merged["hl_underdog_probability"], PROBABILITY_BUCKETS, right=False, include_lowest=True
    ).astype("string")
    merged["market_underdog_probability_bucket"] = pd.cut(
        merged["market_underdog_probability"], PROBABILITY_BUCKETS, right=False, include_lowest=True
    ).astype("string")
    merged["hl_favorite_probability_bucket"] = pd.cut(
        merged["hl_favorite_probability"], PROBABILITY_BUCKETS, right=False, include_lowest=True
    ).astype("string")
    primary = merged[merged.consensus_status.eq("CONSENSUS") & merged.home_win_actual.notna()].copy()

    all_available = merged[merged.book_count.fillna(0).gt(0) & merged.home_win_actual.notna()].copy()
    report = {
        "schema_version": "1.0",
        "benchmark_only": True,
        "market_is_model_input": False,
        "market_used_for_fit_or_recalibration": False,
        "historical_forecast_policy": "Only timestamped frozen pregame forecasts or explicitly labeled chronological out-of-sample forecasts are admitted.",
        "quality_rule": {"consensus_minimum_books": args.minimum_books,
                          "two_books": "LIMITED_CONSENSUS; secondary analysis only",
                          "one_book": "SINGLE_BOOK; not primary consensus",
                          "zero_books": "UNAVAILABLE"},
        "sample": {
            "canonical_games": int(len(games)),
            "forecast_rows_input": int(len(forecasts) + len(bad_forecast)),
            "valid_forecasts": int(len(forecasts)),
            "invalid_forecasts": int(len(bad_forecast)),
            "forecast_games_with_actual_result": int(merged.home_win_actual.notna().sum()),
            "matched_forecast_games_with_any_market": int(len(all_available)),
            "primary_consensus_games": int(len(primary)),
            "limited_consensus_games": int(merged.consensus_status.eq("LIMITED_CONSENSUS").sum()),
            "single_book_games": int(merged.consensus_status.eq("SINGLE_BOOK").sum()),
            "market_unavailable_games": int(merged.consensus_status.eq("UNAVAILABLE").sum()),
            "games_without_model_forecast": int(len(set(games.game_id.astype(str))-set(forecasts.game_id.astype(str)))),
        },
        "source_coverage": source_summary,
        "market_quality_by_season": {},
        "primary_consensus_metrics": {
            "hockey_lab": _score(primary, "hl_home_probability"),
            "market": _score(primary, "market_home_probability"),
            "disagreement": _disagreement_report(primary),
        },
        "all_available_market_metrics_secondary": {
            "games": int(len(all_available)),
            "hockey_lab": _score(all_available, "hl_home_probability"),
            "market": _score(all_available, "market_home_probability"),
        },
        "by_season": _breakdown(primary, "source_season"),
        "road_pick_tendency": _road_tendency(primary),
        "road_pick_tendency_by_season": {},
        "by_hockey_lab_pick_side": _breakdown(primary, "hl_pick_side"),
        "by_market_pick_side": _breakdown(primary, "market_pick_side"),
        "by_hockey_lab_home_probability_bucket": _breakdown(primary, "hl_home_probability_bucket"),
        "by_market_home_probability_bucket": _breakdown(primary, "market_home_probability_bucket"),
        "by_favorite_probability_bucket": _breakdown(primary, "market_favorite_bucket"),
        "by_hockey_lab_favorite_probability_bucket": _breakdown(primary, "hl_favorite_probability_bucket"),
        "by_hockey_lab_underdog_probability_bucket": _breakdown(primary, "hl_underdog_probability_bucket"),
        "by_market_underdog_probability_bucket": _breakdown(primary, "market_underdog_probability_bucket"),
        "by_model_version": {},
        "by_feature_version": _breakdown(primary, "feature_version"),
        "by_goalie_confirmation_state": _breakdown(primary, "goalie_state"),
        "by_rest_state": _breakdown(primary, "rest_home_severity"),
        "by_home_rest_state": _breakdown(primary, "rest_home_severity"),
        "by_away_rest_state": _breakdown(primary, "rest_away_severity"),
        "by_b2b": _breakdown(primary, "any_b2b"),
        "by_3in4": _breakdown(primary, "any_3in4"),
        "by_4in6": _breakdown(primary, "any_4in6"),
        "by_5in8": _breakdown(primary, "any_5in8"),
        "matching_audit": {"unmatched_or_ambiguous_source_rows": sum(1 for r in line_audit if r.get("reason") in {"unknown_nhl_game_id", "no_exact_match", "ambiguous_exact_match"}),
                           "rejected_or_quarantined_line_rows": sum(1 for r in line_audit if r.get("reason") not in {"unknown_nhl_game_id", "no_exact_match", "ambiguous_exact_match"})},
    }
    for key, group in primary.groupby(["model_id", "model_version"], dropna=False):
        report["by_model_version"]["/".join("UNKNOWN" if pd.isna(v) else str(v) for v in key)] = {
            "games": int(len(group)), "hockey_lab": _score(group, "hl_home_probability"),
            "market": _score(group, "market_home_probability"),
        }

    quality = consensus.merge(games[["game_id", "source_season"]], on="game_id", how="left", suffixes=("", "_game"))
    season_col = "source_season" if "source_season" in quality else "source_season_game"
    for season, group in quality.groupby(season_col, dropna=False):
        report["market_quality_by_season"]["UNKNOWN" if pd.isna(season) else str(season)] = {
            "games": int(len(group)),
            "consensus": int(group.consensus_status.eq("CONSENSUS").sum()),
            "limited_consensus": int(group.consensus_status.eq("LIMITED_CONSENSUS").sum()),
            "single_book": int(group.consensus_status.eq("SINGLE_BOOK").sum()),
            "unavailable": int(group.consensus_status.eq("UNAVAILABLE").sum()),
        }
    for season, group in primary.groupby("source_season", dropna=False):
        report["road_pick_tendency_by_season"]["UNKNOWN" if pd.isna(season) else str(season)] = _road_tendency(group)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    lines.to_parquet(out / "closing-book-lines.parquet", index=False)
    consensus.to_parquet(out / "closing-consensus.parquet", index=False)
    merged.to_parquet(out / "hockey-lab-vs-market.parquet", index=False)
    (out / "market-benchmark-report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "unmatched-audit.json").write_text(json.dumps([r for r in line_audit if r.get("reason") in {"unknown_nhl_game_id", "no_exact_match", "ambiguous_exact_match"}], indent=2, default=str) + "\n", encoding="utf-8")
    (out / "line-rejections.json").write_text(json.dumps([r for r in line_audit if r.get("reason") not in {"unknown_nhl_game_id", "no_exact_match", "ambiguous_exact_match"}], indent=2, default=str) + "\n", encoding="utf-8")
    (out / "forecast-rejections.json").write_text(json.dumps(forecast_audit, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", required=True, help="Canonical NHL schedule/results parquet, with NHL game IDs")
    p.add_argument("--forecasts", required=True, help="Frozen forward-ledger directory or chronological OOS forecast table")
    p.add_argument("--lines", required=True, help="Timestamped sportsbook moneyline rows; CSV/JSON/Parquet")
    p.add_argument("--outdir", default=str(DEFAULT_OUT))
    p.add_argument("--minimum-books", type=int, default=3)
    args = p.parse_args()
    report = build(args)
    print(json.dumps({"sample": report["sample"], "primary_consensus_metrics": report["primary_consensus_metrics"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
