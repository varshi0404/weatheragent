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
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI               # swap for another LLM if needed

from weather_scraper import scrape_weather
from parser import parse_weather_data
from weather_analysis import analyze_weather
from report_generator import generate_weather_insights
from invisible_ai.mood_forecast import generate_mood_forecast
from invisible_ai.prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLE


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = [
    scrape_weather,
    parse_weather_data,
    analyze_weather,
    generate_weather_insights,
    generate_mood_forecast,       # ← mood diary tool
]

TOOL_NAMES = ", ".join(t.name for t in TOOLS)
TOOL_DOCS  = "\n\n".join(
    f"{t.name}: {t.description}" for t in TOOLS
)


# ── ReAct prompt ──────────────────────────────────────────────────────────────
# Standard ReAct template expected by create_react_agent.

REACT_TEMPLATE = f"""{SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLE}

You have access to the following tools:

{TOOL_DOCS}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{TOOL_NAMES}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
Thought:{{agent_scratchpad}}"""

REACT_PROMPT = PromptTemplate.from_template(REACT_TEMPLATE)


# ── WeatherAgent class ────────────────────────────────────────────────────────

class WeatherAgent:
    """
    High-level wrapper around a LangChain ReAct agent configured for
    weather analysis.

    Parameters
    ----------
    model_name : str
        OpenAI model to use (default: gpt-4o-mini for cost efficiency).
    temperature : float
        LLM temperature (0 = deterministic, recommended for tool use).
    verbose : bool
        Print agent reasoning steps to stdout.
    """

    def __init__(
        self,
        model_name:  str   = "gpt-4o-mini",
        temperature: float = 0.0,
        verbose:     bool  = True,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set.\n"
                "Set it before running: export OPENAI_API_KEY=sk-..."
            )

        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

        react_agent = create_react_agent(
            llm=llm,
            tools=TOOLS,
            prompt=REACT_PROMPT,
        )

        self.executor = AgentExecutor(
            agent=react_agent,
            tools=TOOLS,
            verbose=verbose,
            handle_parsing_errors=True,
            max_iterations=10,
        )

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
        result = self.executor.invoke({"input": query})
        return result.get("output", "No output produced.")
