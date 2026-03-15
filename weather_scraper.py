"""
weather_scraper.py
------------------
LangChain tool: scrape_weather(city)

Uses the Open-Meteo free API (no key required) to fetch a 7-day hourly forecast,
then returns a JSON string that the parser can consume.

Open-Meteo docs: https://open-meteo.com/en/docs
Geocoding API:   https://geocoding-api.open-meteo.com
"""

import json
import requests
try:
    from langchain.tools import tool
except ImportError:  # fallback when langchain isn't installed
    def tool(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func


# ── helpers ──────────────────────────────────────────────────────────────────

GEO_URL     = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

def _geocode(city: str) -> tuple[float, float]:
    """Return (latitude, longitude) for a city name."""
    resp = requests.get(GEO_URL, params={"name": city, "count": 1}, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"City '{city}' not found in geocoding API.")
    r = results[0]
    return r["latitude"], r["longitude"]


def _fetch_forecast(lat: float, lon: float) -> dict:
    """Fetch 7-day hourly forecast from Open-Meteo."""
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m",
            "relativehumidity_2m",
            "precipitation_probability",
            "windspeed_10m",
            "cloudcover",
        ]),
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "windspeed_10m_max",
        ]),
        "forecast_days": 7,
        "timezone": "auto",
    }
    resp = requests.get(WEATHER_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── LangChain tool ────────────────────────────────────────────────────────────

@tool
def scrape_weather(city: str) -> str:
    """
    Scrape / fetch weather forecast data for a given city.

    Args:
        city: Name of the city (e.g. 'Chennai', 'London').

    Returns:
        JSON string containing raw forecast data, or an error message.
    """
    try:
        lat, lon = _geocode(city)
        data = _fetch_forecast(lat, lon)
        data["_city"] = city
        data["_lat"]  = lat
        data["_lon"]  = lon
        return json.dumps(data)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
