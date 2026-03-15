"""
parser.py
---------
LangChain tool: parse_weather_data(raw_json)

Transforms the raw Open-Meteo JSON into a clean list of daily records
with the exact fields required by the analysis engine:

    date, temp_max, temp_min, humidity_avg,
    rain_probability, wind_speed_max, cloud_cover_avg
"""

import json
from statistics import mean
try:
    from langchain.tools import tool
except ImportError:  # fallback when langchain isn't installed
    def tool(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func


# ── helper ────────────────────────────────────────────────────────────────────

def _safe_mean(values: list) -> float:
    """Return mean of non-None numeric values."""
    filtered = [v for v in values if v is not None]
    return round(mean(filtered), 1) if filtered else 0.0


def _parse(raw: dict) -> list[dict]:
    hourly = raw.get("hourly", {})
    daily  = raw.get("daily",  {})

    hours      = hourly.get("time", [])
    temps_h    = hourly.get("temperature_2m", [])
    humidity_h = hourly.get("relativehumidity_2m", [])
    cloud_h    = hourly.get("cloudcover", [])

    # Build a day → hourly-values map for fields not in daily
    day_humidity: dict[str, list] = {}
    day_cloud:    dict[str, list] = {}

    for i, ts in enumerate(hours):
        day = ts[:10]
        day_humidity.setdefault(day, []).append(humidity_h[i] if i < len(humidity_h) else None)
        day_cloud.setdefault(day,    []).append(cloud_h[i]    if i < len(cloud_h)    else None)

    dates     = daily.get("time", [])
    temp_max  = daily.get("temperature_2m_max", [])
    temp_min  = daily.get("temperature_2m_min", [])
    rain_prob = daily.get("precipitation_probability_max", [])
    wind_max  = daily.get("windspeed_10m_max", [])

    records = []
    for idx, date in enumerate(dates):
        records.append({
            "date":             date,
            "temp_max":         temp_max[idx]  if idx < len(temp_max)  else None,
            "temp_min":         temp_min[idx]  if idx < len(temp_min)  else None,
            "humidity_avg":     _safe_mean(day_humidity.get(date, [])),
            "rain_probability": rain_prob[idx] if idx < len(rain_prob) else 0,
            "wind_speed_max":   wind_max[idx]  if idx < len(wind_max)  else None,
            "cloud_cover_avg":  _safe_mean(day_cloud.get(date, [])),
        })
    return records


# ── LangChain tool ────────────────────────────────────────────────────────────

@tool
def parse_weather_data(raw_json: str) -> str:
    """
    Parse raw weather JSON (from scrape_weather) into structured daily records.

    Args:
        raw_json: JSON string returned by scrape_weather.

    Returns:
        JSON string: list of daily weather dicts, or error dict.
    """
    try:
        raw  = json.loads(raw_json)
        if "error" in raw:
            return raw_json          # pass the error through
        city    = raw.get("_city", "Unknown")
        records = _parse(raw)
        return json.dumps({"city": city, "forecast": records})
    except Exception as exc:
        return json.dumps({"error": str(exc)})
