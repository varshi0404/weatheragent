"""
weather_analysis.py
-------------------
LangChain tool: analyze_weather(parsed_json)

Applies rule-based analysis on the structured daily forecast:

Rules
-----
• rain_probability > 70  → rain_alert
• temp_max > 35          → heat_warning
• humidity_avg > 80      → discomfort_alert
• temp rising each day   → rising_temp_trend
• temp falling each day  → falling_temp_trend

Also determines the "best outdoor days" (low rain, moderate temp, low cloud).
"""

import json
try:
    from langchain.tools import tool
except ImportError:  # fallback when langchain isn't installed
    def tool(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func


# ── scoring helpers ───────────────────────────────────────────────────────────

def _outdoor_score(day: dict) -> float:
    """Higher is better for outdoor activities."""
    score  = 100.0
    score -= day.get("rain_probability", 0) * 0.5   # heavy penalty for rain
    score -= max(0, day.get("temp_max", 30) - 33)   * 3   # penalty above 33 °C
    score -= max(0, 25 - day.get("temp_min", 25))   * 2   # penalty below 25 °C
    score -= day.get("cloud_cover_avg", 50)          * 0.1
    score -= max(0, day.get("humidity_avg", 60) - 70) * 0.3
    return round(score, 1)


def _trend(values: list[float | None]) -> str:
    """Detect monotone rising / falling or mixed trend."""
    v = [x for x in values if x is not None]
    if len(v) < 3:
        return "stable"
    diffs = [v[i+1] - v[i] for i in range(len(v)-1)]
    rising  = sum(1 for d in diffs if d > 0.5)
    falling = sum(1 for d in diffs if d < -0.5)
    if rising   >= len(diffs) * 0.65:
        return "rising"
    if falling  >= len(diffs) * 0.65:
        return "falling"
    return "stable"


# ── main analysis ─────────────────────────────────────────────────────────────

def _analyse(city: str, forecast: list[dict]) -> dict:
    alerts: list[str]   = []
    rainy_days: list[str] = []
    best_days:  list[str] = []

    for day in forecast:
        date = day["date"]
        rp   = day.get("rain_probability", 0)
        tmax = day.get("temp_max") or 0
        hum  = day.get("humidity_avg", 0)

        if rp   > 70:  alerts.append(f"Rain alert on {date} ({rp}% chance)")
        if tmax > 35:  alerts.append(f"Heat warning on {date} ({tmax}°C)")
        if hum  > 80:  alerts.append(f"High-humidity discomfort on {date} ({hum}%)")
        if rp   > 50:  rainy_days.append(date)

    # Score each day for outdoor suitability
    scored = sorted(forecast, key=_outdoor_score, reverse=True)
    best_days = [d["date"] for d in scored[:2]]

    temp_trend     = _trend([d.get("temp_max")        for d in forecast])
    humidity_trend = _trend([d.get("humidity_avg")    for d in forecast])
    rain_trend     = _trend([d.get("rain_probability") for d in forecast])

    temp_values = [d.get("temp_max") for d in forecast if d.get("temp_max")]
    hum_values  = [d.get("humidity_avg") for d in forecast if d.get("humidity_avg")]

    return {
        "city":          city,
        "forecast":      forecast,
        "temp_min_week": min(temp_values)  if temp_values else None,
        "temp_max_week": max(temp_values)  if temp_values else None,
        "humidity_avg_week": round(sum(hum_values)/len(hum_values), 1) if hum_values else None,
        "temp_trend":     temp_trend,
        "humidity_trend": humidity_trend,
        "rain_trend":     rain_trend,
        "rainy_days":     rainy_days,
        "best_outdoor_days": best_days,
        "alerts":         alerts,
    }


# ── LangChain tool ────────────────────────────────────────────────────────────

@tool
def analyze_weather(parsed_json: str) -> str:
    """
    Analyse structured weather forecast data for trends, alerts, and outdoor suitability.

    Args:
        parsed_json: JSON string returned by parse_weather_data.

    Returns:
        JSON string with analysis results, or error dict.
    """
    try:
        data = json.loads(parsed_json)
        if "error" in data:
            return parsed_json
        result = _analyse(data["city"], data["forecast"])
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
