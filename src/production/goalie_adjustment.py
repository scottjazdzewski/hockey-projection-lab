"""Calibrated expected-goalie adjustment for Hockey Lab win probabilities.

The team-only scorer remains authoritative unless a promoted calibration artifact
with the explicit goalie-adjustment feature marker is supplied. Historical
goalie rows are filtered strictly before the target date.
"""
from __future__ import annotations

from datetime import date, datetime
import math
import re
import unicodedata

FEATURE_MARKER = "goalie-adjustment-"


def normalize_player_id(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.fullmatch(r"(\d+)(?:\.0+)?", text)
    return str(int(match.group(1))) if match else text


def normalize_name(value):
    if value is None:
        return None
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return text or None


def _as_date(value):
    if value is None:
        return None
    if hasattr(value, "date") and not isinstance(value, date):
        try:
            return value.date()
        except Exception:
            pass
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def goalie_age(birth_date, as_of):
    birth = _as_date(birth_date)
    target = _as_date(as_of)
    if birth is None or target is None or birth >= target:
        return None
    return (target - birth).days / 365.2425


def age_curve_save_pct(age_years, calibration):
    """Evaluate the training-fitted quadratic age curve in standardized years."""
    if age_years is None or not math.isfinite(float(age_years)):
        return None
    curve = calibration.get("age_curve") or {}
    center = float(curve["center"])
    scale = float(curve["scale"])
    coefficients = curve["coefficients"]
    if scale <= 0 or len(coefficients) != 3:
        raise ValueError("age_curve requires center, positive scale, and 3 coefficients")
    z = (float(age_years) - center) / scale
    value = float(coefficients[0]) + float(coefficients[1]) * z + float(coefficients[2]) * z * z
    if not 0 < value < 1:
        raise ValueError("fitted age curve returned an invalid save percentage")
    return value


def _finite_sum(rows, column):
    total = 0.0
    for row in rows:
        value = row.get(column)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            total += number
    return total


def rate_goalie(goalie_games, player_id, birth_date, as_of, calibration):
    """Return age-adjusted, exposure-shrunk prior rating for one expected goalie."""
    player_id = normalize_player_id(player_id)
    age = goalie_age(birth_date, as_of)
    if player_id is None or age is None:
        return {"available": False, "reason": "unresolved_identity_or_age"}

    target = _as_date(as_of)
    rows = []
    for record in goalie_games:
        if normalize_player_id(record.get("goalie_id_key")) != player_id:
            continue
        if not bool(record.get("starter_bool", False)):
            continue
        game_date = _as_date(record.get("game_date"))
        if game_date is None or game_date >= target:
            continue
        rows.append((game_date, record))
    rows.sort(key=lambda item: item[0])

    current_prior = age_curve_save_pct(age, calibration)
    if current_prior is None:
        return {"available": False, "reason": "age_curve_unavailable"}

    prior_shots = _finite_sum([r for _, r in rows], "shots_against_pbp")
    prior_saves = _finite_sum([r for _, r in rows], "saves_pbp")
    prior_minutes = _finite_sum([r for _, r in rows], "toi_seconds_box") / 60.0
    prior_starts = len(rows)

    expected_saves = 0.0
    valid_saves = 0.0
    valid_shots = 0.0
    for _, record in rows:
        shots = record.get("shots_against_pbp")
        try:
            shots = float(shots)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(shots) or shots <= 0:
            continue
        prior_age = record.get("age_years")
        try:
            prior_age = float(prior_age)
        except (TypeError, ValueError):
            prior_age = None
        if prior_age is None or not math.isfinite(prior_age):
            continue
        baseline = age_curve_save_pct(prior_age, calibration)
        saves = record.get("saves_pbp")
        try:
            saves = float(saves)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(saves):
            continue
        expected_saves += shots * baseline
        valid_saves += saves
        valid_shots += shots

    if valid_shots > 0:
        residual_per_shot = (valid_saves - expected_saves) / valid_shots
    else:
        residual_per_shot = 0.0

    shrink = calibration.get("shrinkage") or {}
    shot_k = float(shrink["shots_k"])
    start_k = float(shrink["starts_k"])
    minute_k = float(shrink["minutes_k"])
    if min(shot_k, start_k, minute_k) < 0:
        raise ValueError("goalie exposure shrinkage values cannot be negative")

    def exposure_reliability(exposure, k):
        if exposure <= 0:
            return 0.0
        return 1.0 if k == 0 else exposure / (exposure + k)

    # Each reliability component is learned on historical folds. Their mean
    # avoids multiplying three correlated exposure measures into near-zero.
    reliability = sum((
        exposure_reliability(prior_shots, shot_k),
        exposure_reliability(prior_starts, start_k),
        exposure_reliability(prior_minutes, minute_k),
    )) / 3.0
    rating = current_prior + reliability * residual_per_shot

    return {
        "available": True,
        "rating": rating,
        "age_years": age,
        "age_prior_save_pct": current_prior,
        "prior_starts": prior_starts,
        "prior_shots": prior_shots,
        "prior_minutes": prior_minutes,
        "reliability": reliability,
        "personalized": prior_starts > 0 and valid_shots > 0,
        "skill_residual_per_shot": residual_per_shot,
    }


def adjust_home_probability(team_only_home_probability, home_rating, away_rating, calibration):
    """Apply the fitted goalie difference on the logit scale."""
    if home_rating is None or away_rating is None:
        return team_only_home_probability
    p = float(team_only_home_probability)
    if not 0 < p < 1:
        raise ValueError("team-only probability must be strictly between zero and one")
    beta = float(calibration["probability_beta"])
    logit = math.log(p / (1.0 - p)) + beta * (float(home_rating) - float(away_rating))
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1.0 + exp_value)


def is_promoted_model(model, expected_baseline_feature_version=None):
    model = model if isinstance(model, dict) else {}
    metrics = model.get("performance_metrics") or {}
    required_metrics = ("accuracy", "brier", "log_loss", "calibration")
    baseline_matches = (
        expected_baseline_feature_version is None
        or model.get("baseline_feature_version") == expected_baseline_feature_version
    )
    return (
        isinstance(model, dict)
        and str(model.get("status", "")).upper() == "PROMOTED"
        and FEATURE_MARKER in str(model.get("feature_version", "")).lower()
        and baseline_matches
        and isinstance(model.get("calibration"), dict)
        and all(isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key]) for key in required_metrics)
    )
