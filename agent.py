"""
agent.py
--------
WeatherAgent: LangChain ReAct agent that orchestrates the weather tools.

Usage
-----
    from agent import WeatherAgent
    wa = WeatherAgent()
    print(wa.run("Analyse the weather for Chennai and give insights."))
"""

import os
import re
import json
import time
import logging
import tempfile
from pathlib import Path

from weather_scraper import scrape_weather
from parser import parse_weather_data
from weather_analysis import analyze_weather
from report_generator import generate_weather_insights
from invisible_ai.mood_forecast import generate_mood_forecast
from prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLE
from invisible_ai.prompts import (
    SYSTEM_PROMPT as MOOD_SYSTEM_PROMPT,
    MOOD_MODE_INSTRUCTION,
    MOOD_FEW_SHOT_EXAMPLE,
)

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path(__file__).with_name("cache.json")
CACHE_PATH_ENV_VAR = "WEATHERAGENT_CACHE_PATH"
CACHE_MAX_ENTRIES = 500
MOOD_FORECAST_MARKER = "Mood Forecast"


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = [
    scrape_weather,
    parse_weather_data,
    analyze_weather,
    generate_weather_insights,
    generate_mood_forecast,
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
# Standard ReAct format expected by create_react_agent.

_REACT_FORMAT = """
You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def _build_react_template(system_prompt: str, few_shot: str) -> str:
    return f"{system_prompt}\n\n{few_shot}\n{_REACT_FORMAT}"


REACT_TEMPLATE = _build_react_template(SYSTEM_PROMPT, FEW_SHOT_EXAMPLE)
MOOD_REACT_TEMPLATE = _build_react_template(
    f"{MOOD_SYSTEM_PROMPT}\n{MOOD_MODE_INSTRUCTION}", MOOD_FEW_SHOT_EXAMPLE
)


# ── City extraction ───────────────────────────────────────────────────────────

_CITY_CHARS = r"[^\W\d_](?:[^\W\d_]|[\s.'-])*"
_CITY_PATTERNS = [
    rf"\bfor\s+({_CITY_CHARS})",
    rf"\bin\s+({_CITY_CHARS})",
    rf"\bat\s+({_CITY_CHARS})",
]
_CLAUSE_TRIM = r"\s+(?:and|with|for|to|on|please|in|at|the|during|this|next)\b"
_NON_CITY_WORDS = {
    "the", "this", "next", "today", "tonight", "tomorrow", "week", "weekend",
    "morning", "afternoon", "evening", "night", "coming", "upcoming", "now",
}


def _has_error(json_str: str) -> bool:
    try:
        data = json.loads(json_str)
    except (TypeError, ValueError):
        return True
    return isinstance(data, dict) and "error" in data


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
    default_city : str | None
        City used when the query names none. If omitted, Chennai is used and
        a warning is appended to the report.
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
        default_city: str | None = None,
    ):
        self.use_agent = use_agent
        self.verbose = verbose
        self.executor = None
        self.prompt = None
        self.cache_enabled = cache_enabled
        self.cache_ttl_sec = cache_ttl_sec
        self.cache_path = Path(os.environ.get(CACHE_PATH_ENV_VAR) or DEFAULT_CACHE_PATH)
        self._cache = self._load_cache()
        self.output_mode = output_mode
        self.default_city = default_city or "Chennai"
        self._default_city_explicit = bool(default_city)
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

            try:
                from langchain_core.prompts import PromptTemplate
            except ImportError as exc:  # pragma: no cover
                raise ImportError("LangChain prompt tools are not available. Install langchain-core.") from exc

            template = MOOD_REACT_TEMPLATE if output_mode == "mood" else REACT_TEMPLATE
            self.prompt = PromptTemplate.from_template(template)

            llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                stop=["\nObservation:"],
            )

            react_agent = create_react_agent(
                llm=llm,
                tools=TOOLS,
                prompt=self.prompt,
            )

            self.executor = AgentExecutor(
                agent=react_agent,
                tools=TOOLS,
                verbose=verbose,
                handle_parsing_errors=True,
                max_iterations=10,
            )

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _is_fresh(self, entry, now: float) -> bool:
        if not isinstance(entry, dict):
            return False
        ts = entry.get("ts")
        return isinstance(ts, (int, float)) and (now - ts) <= self.cache_ttl_sec

    def _prune_cache(self, cache: dict) -> dict:
        """Drop expired/invalid entries and keep only the newest CACHE_MAX_ENTRIES."""
        now = time.time()
        fresh = [(k, v) for k, v in cache.items() if self._is_fresh(v, now)]
        fresh.sort(key=lambda kv: kv[1]["ts"], reverse=True)
        return dict(fresh[:CACHE_MAX_ENTRIES])

    def _load_cache(self) -> dict:
        if not self.cache_enabled:
            return {}
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return self._prune_cache(data) if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_cache(self) -> None:
        if not self.cache_enabled:
            return
        tmp_name = None
        try:
            self._cache = self._prune_cache(self._cache)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.cache_path.parent,
                prefix=f".{self.cache_path.name}.", suffix=".tmp", delete=False,
            ) as f:
                tmp_name = f.name
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
            os.replace(tmp_name, self.cache_path)
        except Exception:
            # Cache failures should never break the app
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass

    def _cache_key(self, query: str) -> str:
        base = re.sub(r"\s+", " ", query.strip().lower())
        return f"{self.output_mode}:{base}"

    def _get_cached(self, query: str) -> str | None:
        if not self.cache_enabled:
            return None
        entry = self._cache.get(self._cache_key(query))
        if not self._is_fresh(entry, time.time()):
            return None
        return entry.get("report")

    def _set_cached(self, query: str, report: str) -> None:
        if not self.cache_enabled:
            return
        key = self._cache_key(query)
        self._cache[key] = {"ts": time.time(), "report": report}
        self._save_cache()

    # ── City extraction ───────────────────────────────────────────────────────

    def _extract_city(self, query: str) -> str | None:
        """Best-effort city extraction from a natural-language query; None if not found."""
        for pat in _CITY_PATTERNS:
            for m in re.finditer(pat, query, flags=re.IGNORECASE):
                # Trim trailing clauses like "and give insights" / "this week"
                city = re.split(_CLAUSE_TRIM, m.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
                city = city.strip(" ,.!?;:'-")
                if city and city.split()[0].lower() not in _NON_CITY_WORDS:
                    return city
        return None

    def _resolve_city(self, query: str) -> tuple[str, str | None]:
        """Return (city, warning). Warning is set when silently defaulting would occur."""
        city = self._extract_city(query)
        if city:
            return city, None
        if self._default_city_explicit:
            return self.default_city, None
        return self.default_city, f"City not specified — defaulting to {self.default_city}"

    # ── Execution ─────────────────────────────────────────────────────────────

    def _run_pipeline(self, query: str) -> str:
        """Deterministic tool pipeline (no LLM tool orchestration)."""
        cached = self._get_cached(query)
        if cached:
            return cached
        city, warning = self._resolve_city(query)
        def _call_tool(t, arg):
            return t.invoke(arg) if hasattr(t, "invoke") else t(arg)

        raw_json = _call_tool(scrape_weather, city)
        parsed_json = _call_tool(parse_weather_data, raw_json)
        failed = _has_error(parsed_json)
        if self.output_mode == "mood":
            report = _call_tool(generate_mood_forecast, parsed_json)
        else:
            analysis_json = _call_tool(analyze_weather, parsed_json)
            failed = failed or _has_error(analysis_json)
            report = _call_tool(generate_weather_insights, analysis_json)
        if warning:
            report = f"{report}\n\n⚠️  {warning}"
        if not failed:
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
            output = result.get("output", "No output produced.")
            if self.output_mode == "mood" and MOOD_FORECAST_MARKER not in str(output):
                raise ValueError("agent did not return a mood forecast")
            return output
        except Exception as exc:
            logger.warning(
                "ReAct agent failed (%s: %s); falling back to deterministic pipeline.",
                type(exc).__name__, exc,
            )
            logger.debug("ReAct agent failure details", exc_info=True)
            report = self._run_pipeline(query)
            if self.verbose:
                report += (
                    f"\n\n[notice] LLM agent failed ({type(exc).__name__}: {exc}); "
                    "this result came from the deterministic pipeline."
                )
            return report
