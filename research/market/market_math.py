"""Pure standard-library price and no-vig calculations shared by collectors."""
from __future__ import annotations

import math
import re
from typing import Any


PRIMARY_BOOKS = frozenset({
    "pinnacle", "circa", "draftkings", "fanduel", "betmgm", "caesars",
    "espnbet", "bet365",
})

BOOK_ALIASES = {
    "pinnacle sportsbook": "pinnacle", "circa sports": "circa",
    "draftkings sportsbook": "draftkings", "dk": "draftkings",
    "fan duel": "fanduel", "fanduel sportsbook": "fanduel",
    "bet mgm": "betmgm", "betmgm sportsbook": "betmgm",
    "caesars sportsbook": "caesars", "william hill": "caesars",
    "williamhill": "caesars", "espn bet": "espnbet",
    "espnbet sportsbook": "espnbet", "bet 365": "bet365",
}


def canonical_sportsbook(value: Any) -> str:
    """Normalize common display-name variants while retaining unknown book IDs."""
    raw = "" if value is None else str(value).strip().lower()
    if raw in {"nan", "<na>", "none"}:
        return ""
    if raw in BOOK_ALIASES:
        return BOOK_ALIASES[raw]
    normalized = re.sub(r"[^a-z0-9]+", "", raw)
    if normalized in BOOK_ALIASES:
        return BOOK_ALIASES[normalized]
    return normalized


def american_implied_probability(price: Any) -> float:
    value = float(price)
    if not math.isfinite(value) or value == 0:
        raise ValueError(f"Invalid American moneyline: {price!r}")
    if value < 0:
        return abs(value) / (abs(value) + 100.0)
    return 100.0 / (value + 100.0)


def no_vig_probabilities(away_ml: Any, home_ml: Any) -> tuple[float, float]:
    away_raw = american_implied_probability(away_ml)
    home_raw = american_implied_probability(home_ml)
    total = away_raw + home_raw
    return away_raw / total, home_raw / total
