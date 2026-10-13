import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def invoke(tool, arg):
    """Call a tool whether or not LangChain's @tool wrapper is active."""
    return tool.invoke(arg) if hasattr(tool, "invoke") else tool(arg)


@pytest.fixture
def forecast_raw() -> dict:
    return json.loads((FIXTURES / "open_meteo_forecast.json").read_text(encoding="utf-8"))


@pytest.fixture
def forecast_raw_json(forecast_raw) -> str:
    return json.dumps(forecast_raw)


@pytest.fixture
def cache_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "cache.json"
    monkeypatch.setenv("WEATHERAGENT_CACHE_PATH", str(path))
    return path


@pytest.fixture
def offline_scraper(monkeypatch, forecast_raw):
    """Replace scrape_weather in agent.py with a fixture-backed fake; records requested cities."""
    import agent

    calls: list[str] = []

    def fake_scrape(city):
        calls.append(city)
        data = dict(forecast_raw, _city=city)
        return json.dumps(data)

    monkeypatch.setattr(agent, "scrape_weather", fake_scrape)
    return calls
