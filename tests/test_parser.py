import json

import weather_scraper
from parser import parse_weather_data

from conftest import invoke

LEGACY_NAMES = {
    "relative_humidity_2m": "relativehumidity_2m",
    "wind_speed_10m": "windspeed_10m",
    "cloud_cover": "cloudcover",
    "wind_speed_10m_max": "windspeed_10m_max",
}


def _parse(raw: dict) -> dict:
    return json.loads(invoke(parse_weather_data, json.dumps(raw)))


def test_parser_extracts_one_record_per_day(forecast_raw):
    out = _parse(forecast_raw)
    assert out["city"] == "Chennai"
    days = out["forecast"]
    assert [d["date"] for d in days] == ["2026-03-16", "2026-03-17", "2026-03-18"]
    first = days[0]
    assert first == {
        "date": "2026-03-16",
        "temp_max": 33.0,
        "temp_min": 26.0,
        "humidity_avg": 70.0,
        "rain_probability": 10,
        "wind_speed_max": 15.0,
        "cloud_cover_avg": 20.0,
    }
    assert days[1]["humidity_avg"] == 85.0
    assert days[1]["cloud_cover_avg"] == 93.8
    assert days[1]["wind_speed_max"] == 35.0


def test_parser_keeps_zero_values(forecast_raw):
    day3 = _parse(forecast_raw)["forecast"][2]
    assert day3["temp_max"] == 0.0
    assert day3["humidity_avg"] == 0.0
    assert day3["cloud_cover_avg"] == 0.0


def test_parser_accepts_legacy_variable_names(forecast_raw):
    legacy = json.loads(json.dumps(forecast_raw))
    for block in ("hourly", "daily"):
        for new, old in LEGACY_NAMES.items():
            if new in legacy[block]:
                legacy[block][old] = legacy[block].pop(new)
    assert _parse(legacy) == _parse(forecast_raw)


def test_parser_passes_errors_through():
    err = json.dumps({"error": "City 'Atlantis' not found in geocoding API."})
    assert invoke(parse_weather_data, err) == err


def test_scraper_requests_current_open_meteo_variable_names(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {}

    def fake_get(url, params=None, timeout=None):
        captured.update(params)
        return FakeResp()

    monkeypatch.setattr(weather_scraper.requests, "get", fake_get)
    weather_scraper._fetch_forecast(13.08, 80.27)
    hourly = captured["hourly"].split(",")
    daily = captured["daily"].split(",")
    assert {"relative_humidity_2m", "wind_speed_10m", "cloud_cover"} <= set(hourly)
    assert "wind_speed_10m_max" in daily
    for legacy in LEGACY_NAMES.values():
        assert legacy not in hourly and legacy not in daily
