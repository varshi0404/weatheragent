"""
test_mood_forecast.py
---------------------
Self-contained smoke test for mood_forecast.py.
Run from the weather_agent/ root:
    python test_mood_forecast.py
No API key or network access required.
"""

import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from insights.mood_forecast import build_mood_forecast, generate_mood_forecast

# ── Test 1: Direct Python import (no LangChain) ───────────────────────────────
RECORDS_PIPELINE = [
    {"date": "2026-03-16", "temp_max": 32, "temp_min": 24,
     "rain_probability": 10, "humidity_avg": 60, "wind_speed_max": 5,  "cloud_cover_avg": 10},
    {"date": "2026-03-17", "temp_max": 35, "temp_min": 26,
     "rain_probability": 70, "humidity_avg": 80, "wind_speed_max": 10, "cloud_cover_avg": 80},
    {"date": "2026-03-18", "temp_max": 28, "temp_min": 22,
     "rain_probability": 20, "humidity_avg": 55, "wind_speed_max": 18, "cloud_cover_avg": 45},
    {"date": "2026-03-19", "temp_max": 38, "temp_min": 27,
     "rain_probability": 5,  "humidity_avg": 50, "wind_speed_max": 3,  "cloud_cover_avg": 5},
    {"date": "2026-03-20", "temp_max": 25, "temp_min": 18,
     "rain_probability": 85, "humidity_avg": 90, "wind_speed_max": 35, "cloud_cover_avg": 95},
]

print("=" * 52)
print("TEST 1 — build_mood_forecast() direct call")
print("=" * 52)
print(build_mood_forecast("Chennai", RECORDS_PIPELINE))

# ── Test 2: Via LangChain @tool, pipeline JSON shape ──────────────────────────
pipeline_json = json.dumps({"city": "Chennai", "forecast": RECORDS_PIPELINE})

print("=" * 52)
print("TEST 2 — generate_mood_forecast() via pipeline JSON")
print("=" * 52)
print(generate_mood_forecast.invoke(pipeline_json))

# ── Test 3: Via LangChain @tool, raw list (prompt spec schema) ────────────────
RAW_LIST = [
    {"date": "2026-03-16", "temperature_max": 32, "temperature_min": 24,
     "rain_probability": 10, "humidity": 60, "wind_speed": 5,  "cloud_cover": 10},
    {"date": "2026-03-17", "temperature_max": 35, "temperature_min": 26,
     "rain_probability": 70, "humidity": 80, "wind_speed": 10, "cloud_cover": 80},
]

print("=" * 52)
print("TEST 3 — generate_mood_forecast() via raw list JSON")
print("=" * 52)
print(generate_mood_forecast.invoke(json.dumps(RAW_LIST)))

print("\nAll tests passed ✓")
