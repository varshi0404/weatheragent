"""
test_mood_forecast.py
---------------------
Pytest tests for mood_forecast.py. Run from the repo root:
    python -m pytest
No API key or network access required.
"""

import json

import pytest

from invisible_ai.mood_forecast import build_mood_forecast, generate_mood_forecast


def _invoke(tool, arg):
    return tool.invoke(arg) if hasattr(tool, "invoke") else tool(arg)


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

RAW_LIST = [
    {"date": "2026-03-16", "temperature_max": 32, "temperature_min": 24,
     "rain_probability": 10, "humidity": 60, "wind_speed": 5,  "cloud_cover": 10},
    {"date": "2026-03-17", "temperature_max": 35, "temperature_min": 26,
     "rain_probability": 70, "humidity": 80, "wind_speed": 10, "cloud_cover": 80},
]


@pytest.mark.parametrize("city", ["Chennai", "São Paulo"])
def test_build_mood_forecast_contains_city_and_every_day(city):
    out = build_mood_forecast(city, RECORDS_PIPELINE)
    assert f"Mood Forecast — {city}" in out
    assert out.count("📅") == len(RECORDS_PIPELINE)
    assert "Monday, March 16" in out


def test_generate_mood_forecast_pipeline_json():
    out = _invoke(generate_mood_forecast, json.dumps({"city": "Chennai", "forecast": RECORDS_PIPELINE}))
    assert "Chennai" in out
    assert out.count("📅") == len(RECORDS_PIPELINE)


def test_generate_mood_forecast_raw_list_json():
    out = _invoke(generate_mood_forecast, json.dumps(RAW_LIST))
    assert "Your Location" in out
    assert "35°C / 26°C" in out


def test_generate_mood_forecast_passes_errors_through():
    out = _invoke(generate_mood_forecast, json.dumps({"error": "boom"}))
    assert out == "Cannot generate mood forecast: boom"


def test_build_mood_forecast_empty():
    assert build_mood_forecast("Chennai", []) == "No weather data provided — nothing to forecast!"
