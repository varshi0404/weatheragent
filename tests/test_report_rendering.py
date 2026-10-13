import json

from report_generator import _format_report
from weather_analysis import _analyse


def _day(date, **overrides):
    day = {"date": date, "temp_max": 30, "temp_min": 25, "humidity_avg": 60,
           "rain_probability": 0, "wind_speed_max": 10, "cloud_cover_avg": 10}
    day.update(overrides)
    return day


def test_zero_values_render_instead_of_na():
    report = _format_report({"city": "Oymyakon", "temp_min_week": 0, "temp_max_week": 0,
                             "humidity_avg_week": 0})
    assert "0–0°C" in report
    assert "avg: 0%" in report
    assert "N/A" not in report


def test_missing_values_render_na():
    report = _format_report({"city": "Nowhere"})
    assert "range: N/A" in report
    assert "avg: N/A" in report


def test_aggregation_keeps_zero_temperatures_and_humidity():
    forecast = [_day("2026-01-01", temp_max=0, humidity_avg=0),
                _day("2026-01-02", temp_max=5, humidity_avg=40)]
    result = _analyse("Oslo", forecast)
    assert result["temp_min_week"] == 0
    assert result["temp_max_week"] == 5
    assert result["humidity_avg_week"] == 20.0


def test_analysis_tolerates_missing_values():
    forecast = [_day("2026-01-01", temp_max=None, temp_min=None, humidity_avg=None,
                     rain_probability=None, cloud_cover_avg=None)]
    result = _analyse("Somewhere", forecast)
    assert result["temp_max_week"] is None
    assert result["humidity_avg_week"] is None
    json.dumps(result)
