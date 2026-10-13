import pytest

from agent import WeatherAgent


@pytest.fixture
def wa(cache_path):
    return WeatherAgent(verbose=False)


@pytest.mark.parametrize("query, expected", [
    ("Weather for Chennai", "Chennai"),
    ("Weather for London?", "London"),
    ("Analyse the weather forecast for Chennai and give insights.", "Chennai"),
    ("What's the forecast for New York this week?", "New York"),
    ("Will it rain in Mumbai next week?", "Mumbai"),
    ("Is it going to be hot in San Francisco during the weekend", "San Francisco"),
    ("weather in london", "london"),
    ("Weather at Bangalore this weekend please", "Bangalore"),
    ("Forecast for the weekend in Paris", "Paris"),
    ("Weather in Chennai for the next 3 days", "Chennai"),
    ("How's the weather in St. Louis?", "St. Louis"),
    ("Give me a mood forecast for São Paulo", "São Paulo"),
    ("Weather for Delhi in the evening", "Delhi"),
    ("Mood diary for Tokyo, please", "Tokyo"),
])
def test_extract_city(wa, query, expected):
    assert wa._extract_city(query) == expected


@pytest.mark.parametrize("query", [
    "What's the weather like?",
    "Forecast for tomorrow",
    "Will it rain this weekend",
    "Give me insights for the next week",
])
def test_extract_city_none_when_missing(wa, query):
    assert wa._extract_city(query) is None


def test_resolve_city_warns_when_defaulting(wa):
    assert wa._resolve_city("What's the weather like?") == (
        "Chennai", "City not specified — defaulting to Chennai"
    )


def test_resolve_city_explicit_default_is_silent(cache_path):
    wa = WeatherAgent(verbose=False, default_city="Madurai")
    assert wa._resolve_city("What's the weather like?") == ("Madurai", None)


def test_pipeline_appends_default_city_warning(cache_path, offline_scraper):
    wa = WeatherAgent(verbose=False)
    report = wa.run("What's the weather like?")
    assert offline_scraper == ["Chennai"]
    assert report.rstrip().endswith("City not specified — defaulting to Chennai")


def test_pipeline_uses_extracted_city(cache_path, offline_scraper):
    wa = WeatherAgent(verbose=False)
    report = wa.run("Will it rain in Mumbai next week?")
    assert offline_scraper == ["Mumbai"]
    assert "WEATHER INSIGHTS REPORT — MUMBAI" in report
    assert "City not specified" not in report
