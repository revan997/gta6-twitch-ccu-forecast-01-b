"""Hour-by-hour empirical-quantile forecaster for launch-day Twitch viewers.

Model (written fresh for this competition):
  For a case, let p1 be hour 1's viewer print.  For each later hour h we
  forecast the log-ratio y_h = ln(viewers_h / p1) from the case's own
  comparable past launches.  The comparables' release-day curves give, per
  hour, a small empirical sample of y_h -- but the sample is NOT used raw:
  the ratio depends strongly on how big hour 1 already was (across the
  comparables, corr(ln p1, ln ratio at hour 19) is about -0.93: a launch
  that is already huge in hour 1 stays flat, one that is still tiny in
  hour 1 explodes later as time zones unlock).  So per hour we fit a
  weighted local-linear regression of y_h on x = ln(hour-1 viewers),
  weighting each comparable by a Gaussian kernel in x (bandwidth 1.5 in
  log units, hard trim beyond 3 bandwidths), then take weighted empirical
  quantiles of the regression residuals at the five scored levels, shrunk
  halfway toward a normal with the same residual scale, and narrowed by a
  factor 0.85 (tuned on leave-one-out validation).  The median curve is
  lightly smoothed across neighbouring hours (0.25/0.5/0.25) so a single
  noisy hour cannot spike the forecast; the same shift is applied to all
  five quantiles of that hour, so ordering is preserved.

  Comparables are also filtered by the tier tag in their unlock note:
  weak analogues and early-access-contaminated launches are dropped,
  because their hour-1 print does not mean the same thing (early access
  inflates hour 1, free-to-play launches behave differently).  A
  comparable with no tier tag is kept.  If the filter would leave too
  few curves, all of them are used instead.

  Everything is computed at run time from the case file alone: standard
  library only, no network, no randomness, deterministic.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles at the scored levels.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Tuned constants (leave-one-out + practice validation; see REPORT.md).
BANDWIDTH = 1.5      # Gaussian kernel bandwidth on ln(hour-1 viewers)
TRIM = 3.0           # drop comparables beyond this many bandwidths
SLOPE_CLAMP = 1.5    # local-linear slope is clamped to +/- this
SHRINK = 0.5         # residual quantiles shrunk this far toward normal
SPREAD = 0.85        # residual quantiles narrowed by this factor
MIN_WEIGHTED_POINTS = 5
FALLBACK_SIGMA = 0.35  # log-spread per sqrt(hour) if comparables unusable


def _parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hour_key(moment):
    """Canonical 'YYYY-MM-DDTHH:MM:SSZ' key for an hour start."""
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _tier(unlock):
    """Tier tag carried in a comparable's unlock note, or 'comp'."""
    text = unlock or ""
    if "Tier: weak" in text:
        return "weak"
    if "early-access-contaminated" in text:
        return "ea"
    if "target-history" in text:
        return "target"
    return "comp"


def _release_day_curve(comp):
    """Return the comparable's 24 release-day viewer counts, or None.

    The hourly list spans 14 days before to 3 days after the release
    day; the release day starts at comp['day_start'].  A missing or
    non-positive hour makes the curve unusable for ratio work.
    """
    try:
        start = _parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is not None and viewers > 0:
            try:
                key = _hour_key(_parse_time(point.get("from", "")))
            except (TypeError, ValueError):
                key = point.get("from")
            by_hour[key] = float(viewers)
    values = []
    for offset in range(24):
        key = _hour_key(start + timedelta(hours=offset))
        value = by_hour.get(key)
        if value is None:
            return None
        values.append(value)
    if values[0] <= 0:
        return None
    return values


def _collect_curves(case):
    """Usable comparable curves, tier-filtered with a safe fallback."""
    curves = []
    for comp in case.get("comparables") or []:
        values = _release_day_curve(comp)
        if values is None:
            continue
        curves.append((values, _tier(comp.get("unlock", ""))))
    kept = [values for values, tier in curves if tier == "comp"]
    if len(kept) >= 8:
        return kept
    return [values for values, _tier in curves]


def _weighted_quantile(pairs, tau):
    """Weighted empirical quantile with linear interpolation inside a bin.

    pairs is a list of (value, weight), unsorted.  Returns a value.
    """
    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = tau * total
    acc = 0.0
    for index, (value, weight) in enumerate(ordered):
        before = acc
        acc += weight
        if acc >= target:
            if index == 0 or weight <= 0:
                return value
            frac = (target - before) / weight
            return ordered[index - 1][0] + frac * (value - ordered[index - 1][0])
    return ordered[-1][0]


def _hour_log_quantiles(curves, p1, hour):
    """Five log-ratio quantiles for one hour (index 1..23 = hours 2..24)."""
    x0 = math.log(p1)
    points = []
    for values in curves:
        ratio = values[hour] / values[0]
        if ratio <= 0:
            continue
        dx = (math.log(values[0]) - x0) / BANDWIDTH
        if abs(dx) > TRIM:
            continue
        weight = math.exp(-0.5 * dx * dx)
        if weight < 1e-6:
            continue
        points.append((math.log(values[0]), math.log(ratio), weight))
    if len(points) < MIN_WEIGHTED_POINTS:
        # Too few neighbours: fall back to every curve, equally weighted.
        points = []
        for values in curves:
            ratio = values[hour] / values[0]
            if ratio > 0:
                points.append((math.log(values[0]), math.log(ratio), 1.0))
    if not points:
        return None

    # Weighted local-linear fit of log-ratio on ln(hour-1 viewers),
    # centred at the case's own x0 so the intercept is the prediction.
    sw = sum(w for _, _, w in points)
    sx = sum(w * (x - x0) for x, _, w in points)
    sy = sum(w * y for _, y, w in points)
    sxx = sum(w * (x - x0) ** 2 for x, _, w in points)
    sxy = sum(w * (x - x0) * y for x, y, w in points)
    denom = sw * sxx - sx * sx
    if denom > 1e-12:
        slope = (sw * sxy - sx * sy) / denom
        centre = (sy - slope * sx) / sw
    else:
        slope = 0.0
        centre = sy / sw
    slope = max(-SLOPE_CLAMP, min(SLOPE_CLAMP, slope))

    residuals = [(y - (centre + slope * (x - x0)), w) for x, y, w in points]
    variance = sum(w * e * e for e, w in residuals) / sw
    scale = math.sqrt(max(variance, 0.0))

    quantiles = []
    for tau, z in zip(LEVELS, Z):
        empirical = _weighted_quantile(residuals, tau)
        shrunk = (1.0 - SHRINK) * empirical + SHRINK * z * scale
        quantiles.append(centre + SPREAD * shrunk)
    return quantiles


def _round2(value):
    """Round to 2 decimals, half up; whole numbers come back as int."""
    scaled = value * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100.0
    return int(rounded) if float(rounded).is_integer() else rounded


def _fallback_quantiles(p1, hour):
    """Wide persistence forecast, used only if comparables are unusable."""
    sigma = FALLBACK_SIGMA * math.sqrt(hour - 1)
    return [p1 * math.exp(z * sigma) for z in Z]


def forecast_case(case):
    """Forecast one case: {'hours': [{'hour': h, 'q': [...]}, ...]}."""
    known = next((row for row in case.get("known") or []
                  if row.get("hour") == 1), None)
    if known is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(known["viewers"]))

    curves = _collect_curves(case)
    hours = sorted(row["hour"] for row in case.get("forecast") or [])
    if not hours:
        hours = list(range(2, 25))

    # Raw per-hour log-ratio quantiles.
    raw = {}
    for hour in hours:
        quantiles = _hour_log_quantiles(curves, p1, hour - 1) if curves else None
        if quantiles is None:
            quantiles = [math.log(q / p1) for q in _fallback_quantiles(p1, hour)]
        raw[hour] = quantiles

    # Smooth the median curve across neighbouring hours; shift every
    # quantile of an hour by the same amount so ordering is preserved.
    medians = [raw[hour][2] for hour in hours]
    smoothed = []
    for index in range(len(hours)):
        if len(hours) == 1:
            smoothed.append(medians[0])
        elif index == 0:
            smoothed.append(0.75 * medians[0] + 0.25 * medians[1])
        elif index == len(hours) - 1:
            smoothed.append(0.25 * medians[-2] + 0.75 * medians[-1])
        else:
            smoothed.append(0.25 * medians[index - 1]
                            + 0.5 * medians[index]
                            + 0.25 * medians[index + 1])
    shifts = [smoothed[i] - medians[i] for i in range(len(hours))]

    rows = []
    for index, hour in enumerate(hours):
        quantiles = [math.exp(q + shifts[index]) * p1 for q in raw[hour]]
        # Enforce non-decreasing order and the format bounds.
        cleaned = []
        previous = 0.0
        for value in quantiles:
            value = min(max(value, previous), 1e9)
            value = max(value, 0.01)
            cleaned.append(_round2(value))
            previous = cleaned[-1]
        # Rounding can only move a value down to the previous one's level
        # at most; re-enforce monotonicity after rounding.
        for j in range(1, 5):
            if cleaned[j] < cleaned[j - 1]:
                cleaned[j] = cleaned[j - 1]
        rows.append({"hour": hour, "q": cleaned})
    return {"hours": rows}


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        with open(path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        text = json.dumps(result, separators=(",", ":"), sort_keys=False)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
