"""
main.py
-------
Entry point for the Weather Agent system.

Run:
    python main.py                        # prompts for city
    python main.py "Weather for London?"  # custom query
    python main.py --interactive          # interactive REPL

    python main.py --mode mood "Weather for Paris"   # friendly mood diary
    python main.py --use-agent "Weather in Tokyo"    # LLM (ReAct) orchestration

Environment:
    The default deterministic pipeline needs only network access to Open-Meteo.
    Ollama (with the chosen --model pulled) is required only with --use-agent.
"""

import sys
import argparse
from agent import WeatherAgent


def main():
    parser = argparse.ArgumentParser(description="Weather AI Agent")
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Natural-language weather query.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive REPL mode.",
    )
    parser.add_argument(
        "--model",
        default="llama3:8b",
        help="Ollama model name, used with --use-agent (default: llama3:8b).",
    )
    parser.add_argument(
        "--mode",
        choices=["report", "mood"],
        default="report",
        help="Output mode: report or mood.",
    )
    parser.add_argument(
        "--use-agent",
        action="store_true",
        help="Use LLM tool orchestration (may be less stable with local models).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress agent reasoning steps.",
    )
    args = parser.parse_args()

    print("\n🌤  Weather Agent initialising…")
    agent = WeatherAgent(
        model_name=args.model,
        verbose=not args.quiet,
        use_agent=args.use_agent,
        output_mode=args.mode,
    )
    print("✅  Agent ready.\n")

    if args.interactive:
        print("Interactive mode — type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if query.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            if not query:
                continue
            answer = agent.run(query)
            print(f"\n{answer}\n")
    else:
        query = args.query
        if not query:
            city = input("Enter city name: ").strip()
            if not city:
                print("No city provided. Exiting.")
                return
            query = f"Weather for {city}"
        print(f"Query: {query}\n")
        answer = agent.run(query)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
