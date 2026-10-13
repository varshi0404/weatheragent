import json
import time

import agent
from agent import CACHE_MAX_ENTRIES, WeatherAgent


def test_cache_path_configurable_via_env(cache_path):
    assert WeatherAgent(verbose=False).cache_path == cache_path


def test_cache_path_defaults_next_to_agent(monkeypatch):
    monkeypatch.delenv("WEATHERAGENT_CACHE_PATH", raising=False)
    assert WeatherAgent(verbose=False, cache_enabled=False).cache_path == agent.DEFAULT_CACHE_PATH


def test_no_cache_write_on_scrape_error(cache_path, monkeypatch):
    monkeypatch.setattr(agent, "scrape_weather",
                        lambda city: json.dumps({"error": f"City '{city}' not found in geocoding API."}))
    wa = WeatherAgent(verbose=False)
    report = wa.run("Weather for Atlantis")
    assert "not found" in report
    assert not cache_path.exists()
    assert wa._cache == {}


def test_no_cache_write_on_error_in_mood_mode(cache_path, monkeypatch):
    monkeypatch.setattr(agent, "scrape_weather", lambda city: json.dumps({"error": "timeout"}))
    wa = WeatherAgent(verbose=False, output_mode="mood")
    assert wa.run("Weather for Paris") == "Cannot generate mood forecast: timeout"
    assert not cache_path.exists()


def test_successful_result_cached_and_reused(cache_path, offline_scraper):
    wa = WeatherAgent(verbose=False)
    first = wa.run("Weather for Paris")
    assert cache_path.exists()
    stored = json.loads(cache_path.read_text(encoding="utf-8"))
    assert stored["report:weather for paris"]["report"] == first
    assert WeatherAgent(verbose=False).run("Weather for Paris") == first
    assert offline_scraper == ["Paris"]


def test_atomic_write_leaves_no_temp_files(cache_path, offline_scraper):
    WeatherAgent(verbose=False).run("Weather for Paris")
    assert [p.name for p in cache_path.parent.iterdir()] == ["cache.json"]


def test_expired_and_invalid_entries_evicted_on_load(cache_path):
    now = time.time()
    cache_path.write_text(json.dumps({
        "report:fresh": {"ts": now, "report": "fresh"},
        "report:stale": {"ts": now - 10_000, "report": "stale"},
        "report:bad-ts": {"ts": "yesterday", "report": "bad"},
        "report:not-a-dict": "oops",
    }), encoding="utf-8")
    wa = WeatherAgent(verbose=False, cache_ttl_sec=1800)
    assert list(wa._cache) == ["report:fresh"]


def test_cache_capped_to_newest_entries(cache_path):
    now = time.time()
    entries = {f"report:q{i}": {"ts": now - i, "report": str(i)} for i in range(CACHE_MAX_ENTRIES + 25)}
    cache_path.write_text(json.dumps(entries), encoding="utf-8")
    wa = WeatherAgent(verbose=False)
    assert len(wa._cache) == CACHE_MAX_ENTRIES
    assert "report:q0" in wa._cache
    assert f"report:q{CACHE_MAX_ENTRIES}" not in wa._cache
    wa._set_cached("brand new", "x")
    on_disk = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(on_disk) == CACHE_MAX_ENTRIES
    assert "report:brand new" in on_disk
