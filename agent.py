"""
agent.py
--------
WeatherAgent: LangChain ReAct agent that orchestrates the four weather tools.

Usage
-----
    from agent.agent import WeatherAgent
    wa = WeatherAgent()
    print(wa.run("Analyse the weather for Chennai and give insights."))
"""

import os
import re
import json
import time
from pathlib import Path

from weather_scraper import scrape_weather
from parser import parse_weather_data
from weather_analysis import analyze_weather
from report_generator import generate_weather_insights
from invisible_ai.mood_forecast import generate_mood_forecast
from prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLE

LC_REACT_PROMPT = None


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = [
    scrape_weather,
    parse_weather_data,
    analyze_weather,
    generate_weather_insights,
]

def _tool_name(t):
    return getattr(t, "name", getattr(t, "__name__", "tool"))

def _tool_desc(t):
    return getattr(t, "description", "")

TOOL_NAMES = ", ".join(_tool_name(t) for t in TOOLS)
TOOL_DOCS  = "\n\n".join(
    f"{_tool_name(t)}: {_tool_desc(t)}" for t in TOOLS
)


# ── ReAct prompt ──────────────────────────────────────────────────────────────
# Standard ReAct template expected by create_react_agent.

REACT_TEMPLATE = f"""{SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLE}

You have access to the following tools:

{{tools}}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
Thought:{{agent_scratchpad}}"""

REACT_PROMPT = None


# ── WeatherAgent class ────────────────────────────────────────────────────────

class WeatherAgent:
    """
    High-level wrapper around a LangChain ReAct agent configured for
    weather analysis.

    Parameters
    ----------
    model_name : str
        Ollama model to use (default: llama3:8b).
    temperature : float
        LLM temperature (0 = deterministic, recommended for tool use).
    verbose : bool
        Print agent reasoning steps to stdout.
    """

    def __init__(
        self,
        model_name:  str   = "llama3:8b",
        temperature: float = 0.0,
        verbose:     bool  = True,
        use_agent:   bool  = False,
        cache_enabled: bool = True,
        cache_ttl_sec: int = 1800,
        output_mode: str = "report",
    ):
        self.use_agent = use_agent
        self.executor = None
        self.cache_enabled = cache_enabled
        self.cache_ttl_sec = cache_ttl_sec
        self.cache_path = Path(__file__).with_name("cache.json")
        self._cache = self._load_cache()
        self.output_mode = output_mode
        if self.use_agent:
            try:
                from langchain_ollama import ChatOllama  # local Ollama LLM
            except ImportError as exc:
                raise ImportError("langchain-ollama is required for use_agent=True") from exc

            # Import LangChain agent pieces only when needed (avoids errors in UI mode)
            try:
                from langchain.agents import create_react_agent
            except ImportError:  # pragma: no cover - compatibility shim
                from langchain.agents.react.agent import create_react_agent

            try:
                from langchain.agents import AgentExecutor
            except ImportError:  # pragma: no cover - compatibility shim
                from langchain.agents.agent import AgentExecutor

            global LC_REACT_PROMPT
            if LC_REACT_PROMPT is None:
                try:
                    from langchain.agents.react.prompt import REACT_PROMPT as LC_REACT_PROMPT
                except ImportError:  # pragma: no cover
                    LC_REACT_PROMPT = None

            global REACT_PROMPT
            if REACT_PROMPT is None:
                try:
                    from langchain_core.prompts import PromptTemplate
                    REACT_PROMPT = PromptTemplate.from_template(REACT_TEMPLATE)
                except ImportError:  # pragma: no cover
                    REACT_PROMPT = None

            llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                stop=["\nObservation:"],
            )

            prompt = LC_REACT_PROMPT if LC_REACT_PROMPT is not None else REACT_PROMPT
            if prompt is None:
                raise ImportError("LangChain prompt tools are not available. Install langchain-core.")
            react_agent = create_react_agent(
                llm=llm,
                tools=TOOLS,
                prompt=prompt,
            )

            self.executor = AgentExecutor(
                agent=react_agent,
                tools=TOOLS,
                verbose=verbose,
                handle_parsing_errors=True,
                max_iterations=10,
            )

        self.default_city = "Chennai"

    def _load_cache(self) -> dict:
        if not self.cache_enabled:
            return {}
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_cache(self) -> None:
        if not self.cache_enabled:
            return
        try:
            with self.cache_path.open("w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception:
            # Cache failures should never break the app
            pass

    def _cache_key(self, query: str) -> str:
        base = re.sub(r"\s+", " ", query.strip().lower())
        return f"{self.output_mode}:{base}"

    def _get_cached(self, query: str) -> str | None:
        if not self.cache_enabled:
            return None
        key = self._cache_key(query)
        entry = self._cache.get(key)
        if not isinstance(entry, dict):
            return None
        ts = entry.get("ts")
        if not isinstance(ts, (int, float)):
            return None
        if (time.time() - ts) > self.cache_ttl_sec:
            return None
        return entry.get("report")

    def _set_cached(self, query: str, report: str) -> None:
        if not self.cache_enabled:
            return
        key = self._cache_key(query)
        self._cache[key] = {"ts": time.time(), "report": report}
        self._save_cache()

    def _extract_city(self, query: str) -> str:
        """Best-effort city extraction from a natural-language query."""
        patterns = [
            r"\bfor\s+([A-Za-z][A-Za-z\s\.-]+)",
            r"\bin\s+([A-Za-z][A-Za-z\s\.-]+)",
            r"\bat\s+([A-Za-z][A-Za-z\s\.-]+)",
        ]
        for pat in patterns:
            m = re.search(pat, query, flags=re.IGNORECASE)
            if m:
                city = m.group(1).strip()
                # Trim trailing clauses like "and give insights"
                city = re.split(r"\s+(and|with|for|to|on|please)\b", city, maxsplit=1, flags=re.IGNORECASE)[0]
                return city.strip(" ,.!?;:")
        return self.default_city

    def _run_pipeline(self, query: str) -> str:
        """Deterministic tool pipeline (no LLM tool orchestration)."""
        cached = self._get_cached(query)
        if cached:
            return cached
        city = self._extract_city(query)
        def _call_tool(t, arg):
            return t.invoke(arg) if hasattr(t, "invoke") else t(arg)

        raw_json = _call_tool(scrape_weather, city)
        parsed_json = _call_tool(parse_weather_data, raw_json)
        if self.output_mode == "mood":
            report = _call_tool(generate_mood_forecast, parsed_json)
        else:
            analysis_json = _call_tool(analyze_weather, parsed_json)
            report = _call_tool(generate_weather_insights, analysis_json)
        self._set_cached(query, report)
        return report

    def run(self, query: str) -> str:
        """
        Run the weather agent on a natural-language query.

        Parameters
        ----------
        query : str
            e.g. "Analyse the weather forecast for Chennai and give insights."

        Returns
        -------
        str
            The final answer / weather insights report.
        """
        if not self.use_agent:
            return self._run_pipeline(query)
        try:
            result = self.executor.invoke({"input": query})
            return result.get("output", "No output produced.")
        except Exception:
            return self._run_pipeline(query)
