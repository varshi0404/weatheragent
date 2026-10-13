# WeatherAgent

Natural-language weather insights built on the free [Open-Meteo](https://open-meteo.com/en/docs) API
(no API key). Queries run through a deterministic tool pipeline by default, or optionally through a
LangChain ReAct agent backed by a local Ollama model.

```
scrape_weather → parse_weather_data → analyze_weather → generate_weather_insights   (report mode)
scrape_weather → parse_weather_data → generate_mood_forecast                        (mood mode)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py "Will it rain in Chennai this week?"   # analytical report (default)
python main.py --mode mood "Weather for Paris"        # friendly day-by-day mood diary
python main.py --interactive                          # REPL; type quit/exit to stop
python main.py --use-agent "Weather in Tokyo"         # LLM (ReAct) tool orchestration
python main.py --use-agent --model llama3:8b --mode mood -q "Mood forecast for Mumbai"
streamlit run streamlit_app.py                        # web UI
```

| Flag | Meaning |
| --- | --- |
| `--mode report\|mood` | Analytical insights report (default) or mood forecast. |
| `--use-agent` | Let the LLM orchestrate the tools. Falls back to the deterministic pipeline (logged, plus a notice when not `-q`) if the agent errors or, in mood mode, doesn't return a mood forecast. |
| `--model` | Ollama model for `--use-agent` (default `llama3:8b`). |
| `--interactive`, `-i` | Interactive prompt loop. |
| `--quiet`, `-q` | Hide agent reasoning steps and fallback notices. |

The city is extracted from phrases like "for/in/at &lt;city&gt;". If none is found, Chennai is used and
the output ends with `City not specified — defaulting to Chennai`.

### Ollama

Ollama is **only** required with `--use-agent` (or "Use LLM orchestration" in the Streamlit UI):

```bash
ollama pull llama3:8b
ollama serve
```

### Cache

Successful results are cached for 30 minutes in `cache.json` next to `agent.py` (git-ignored).
Override the location with `WEATHERAGENT_CACHE_PATH=/path/to/cache.json`. Errors are never cached,
expired entries are evicted on load, and the cache keeps at most the newest 500 entries.

## Tests

```bash
python -m pytest
```

Tests use fixture JSON in `tests/fixtures/` and make no network calls. Agent-mode tests use a fake
LLM and are skipped when LangChain isn't installed.
