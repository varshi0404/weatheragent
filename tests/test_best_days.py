from report_generator import _format_report
from weather_analysis import _analyse, _outdoor_score


def _day(date, **overrides):
    day = {"date": date, "temp_max": 30, "temp_min": 25, "humidity_avg": 60,
           "rain_probability": 0, "wind_speed_max": 10, "cloud_cover_avg": 10}
    day.update(overrides)
    return day


TERRIBLE = {"rain_probability": 100, "temp_max": 45, "humidity_avg": 95, "cloud_cover_avg": 100}


def test_terrible_days_score_below_zero():
    assert _outdoor_score(_day("2026-01-01", **TERRIBLE)) <= 0


def test_no_best_days_when_all_scores_non_positive():
    forecast = [_day("2026-01-01", **TERRIBLE), _day("2026-01-02", **TERRIBLE)]
    analysis = _analyse("Hell", forecast)
    assert analysis["best_outdoor_days"] == []
    assert "Best days for outdoor activities" not in _format_report(analysis)


def test_only_positive_days_listed():
    forecast = [_day("2026-01-01", **TERRIBLE), _day("2026-01-02"), _day("2026-01-03", **TERRIBLE)]
    analysis = _analyse("Mixed", forecast)
    assert analysis["best_outdoor_days"] == ["2026-01-02"]
    assert "Best days for outdoor activities: 2026-01-02." in _format_report(analysis)


def test_best_days_capped_at_two_and_ordered_by_score():
    forecast = [_day("2026-01-01", rain_probability=40), _day("2026-01-02"),
                _day("2026-01-03", rain_probability=20)]
    assert _analyse("Nice", forecast)["best_outdoor_days"] == ["2026-01-02", "2026-01-03"]
