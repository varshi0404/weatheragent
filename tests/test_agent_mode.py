import logging

import pytest

import agent
import prompts
from agent import TOOLS, WeatherAgent
from invisible_ai.mood_forecast import generate_mood_forecast
from invisible_ai.prompts import MOOD_MODE_INSTRUCTION

pytest.importorskip("langchain")
pytest.importorskip("langchain_ollama")

from langchain_core.language_models.fake import FakeListLLM  # noqa: E402
from langchain_core.tools import tool  # noqa: E402


def test_mood_tool_registered():
    assert generate_mood_forecast in TOOLS


def test_no_langchain_hub_prompt_reference():
    assert not hasattr(agent, "LC_REACT_PROMPT")


@pytest.fixture
def fake_llm(monkeypatch):
    """Patch ChatOllama so WeatherAgent(use_agent=True) runs against scripted responses."""
    import langchain_ollama

    holder = {"responses": []}

    def factory(**_kwargs):
        return FakeListLLM(responses=holder["responses"])

    monkeypatch.setattr(langchain_ollama, "ChatOllama", factory)
    return holder


@pytest.fixture
def offline_agent_tools(monkeypatch, forecast_raw_json):
    @tool
    def scrape_weather(city: str) -> str:
        """Fetch weather forecast data for a given city."""
        return forecast_raw_json

    tools = [scrape_weather if agent._tool_name(t) == "scrape_weather" else t for t in TOOLS]
    monkeypatch.setattr(agent, "TOOLS", tools)


@pytest.mark.parametrize("mode", ["report", "mood"])
def test_prompt_is_locally_built(cache_path, fake_llm, mode):
    wa = WeatherAgent(verbose=False, use_agent=True, output_mode=mode)
    template = wa.prompt.template
    assert "Question: {input}" in template
    assert {"tools", "tool_names", "input", "agent_scratchpad"} <= set(wa.prompt.input_variables)
    rendered = wa.prompt.format(tools="T", tool_names="N", input="Q", agent_scratchpad="")
    assert "{{" not in rendered
    if mode == "mood":
        assert MOOD_MODE_INSTRUCTION.strip() in template
        assert "generate_mood_forecast" in rendered
    else:
        assert prompts.SYSTEM_PROMPT.strip() in template
        assert MOOD_MODE_INSTRUCTION.strip() not in template


def test_agent_mood_mode_yields_mood_forecast(cache_path, fake_llm, offline_agent_tools, forecast_raw_json):
    parsed = agent.parse_weather_data.invoke(forecast_raw_json)
    mood = generate_mood_forecast.invoke(parsed)
    fake_llm["responses"] = [
        "Thought: fetch data\nAction: scrape_weather\nAction Input: Chennai",
        f"Thought: parse it\nAction: parse_weather_data\nAction Input: {forecast_raw_json}",
        f"Thought: mood time\nAction: generate_mood_forecast\nAction Input: {parsed}",
        f"Thought: I now know the final answer\nFinal Answer: {mood}",
    ]
    wa = WeatherAgent(verbose=False, use_agent=True, output_mode="mood")
    out = wa.run("Give me a mood forecast for Chennai")
    assert "Mood Forecast — Chennai" in out
    assert "[notice]" not in out


def test_agent_mood_mode_falls_back_when_llm_ignores_mood(cache_path, fake_llm, offline_scraper, caplog):
    fake_llm["responses"] = ["Thought: I now know the final answer\nFinal Answer: It will be sunny."]
    wa = WeatherAgent(verbose=True, use_agent=True, output_mode="mood")
    wa.executor.verbose = False
    with caplog.at_level(logging.WARNING, logger="agent"):
        out = wa.run("Mood forecast for Chennai")
    assert "Mood Forecast — Chennai" in out
    assert "[notice] LLM agent failed" in out
    assert any("falling back to deterministic pipeline" in r.message for r in caplog.records)


def test_agent_failure_logged_and_quiet_without_verbose(cache_path, fake_llm, offline_scraper, caplog):
    wa = WeatherAgent(verbose=False, use_agent=True)
    wa.executor = None  # any executor failure triggers the fallback
    with caplog.at_level(logging.WARNING, logger="agent"):
        out = wa.run("Weather for Paris")
    assert "WEATHER INSIGHTS REPORT — PARIS" in out
    assert "[notice]" not in out
    assert any("falling back" in r.message for r in caplog.records)
