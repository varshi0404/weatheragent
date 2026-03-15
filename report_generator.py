"""
report_generator.py
-------------------
LangChain tool: generate_weather_insights(analysis_json)

Converts the structured analysis dict into a polished, human-friendly
plain-text report.
"""

import json
try:
    from langchain.tools import tool
except ImportError:  # fallback when langchain isn't installed
    def tool(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func


# ── formatting helpers ────────────────────────────────────────────────────────

_TREND_LABEL = {
    "rising":  "📈 Rising throughout the week",
    "falling": "📉 Falling throughout the week",
    "stable":  "➡️  Relatively stable",
}

_RAIN_LABEL = {
    "rising":  "Rain chances are increasing as the week progresses.",
    "falling": "Rain chances are decreasing as the week progresses.",
    "stable":  "Rain chances remain fairly consistent all week.",
}


def _format_report(analysis: dict) -> str:
    city       = analysis.get("city", "Unknown")
    tmin       = analysis.get("temp_min_week")
    tmax       = analysis.get("temp_max_week")
    hum        = analysis.get("humidity_avg_week")
    t_trend    = analysis.get("temp_trend", "stable")
    h_trend    = analysis.get("humidity_trend", "stable")
    r_trend    = analysis.get("rain_trend", "stable")
    rainy      = analysis.get("rainy_days", [])
    best       = analysis.get("best_outdoor_days", [])
    alerts     = analysis.get("alerts", [])
    forecast   = analysis.get("forecast", [])

    # ── Summary numbers ─────────────────────────────────────────
    temp_range = f"{tmin}–{tmax}°C" if tmin and tmax else "N/A"
    hum_str    = f"{hum}%" if hum else "N/A"

    # ── Key insights ─────────────────────────────────────────────
    insights: list[str] = []

    insights.append(f"Temperature will range between {temp_range} this week.")

    if rainy:
        rainy_str = ", ".join(rainy[:3])
        insights.append(f"Rain expected on: {rainy_str}{'...' if len(rainy) > 3 else ''}.")
    else:
        insights.append("No significant rain expected this week.")

    insights.append(f"Humidity is averaging {hum_str} — "
                    + ("expect muggy conditions." if (hum or 0) > 75 else "comfortable levels."))

    if best:
        insights.append(f"Best days for outdoor activities: {', '.join(best)}.")

    for alert in alerts[:4]:            # cap to avoid overwhelming output
        insights.append(f"⚠️  {alert}")

    # ── Recommendations ───────────────────────────────────────────
    recommendations: list[str] = []

    # Outdoor advice
    if rainy:
        recommendations.append(f"Plan outdoor events on {', '.join(best) or 'early in the week'}; "
                                f"avoid {', '.join(rainy[:2])}.")
    else:
        recommendations.append("All days look good for outdoor plans.")

    # Travel advice
    high_wind_days = [d["date"] for d in forecast if (d.get("wind_speed_max") or 0) > 30]
    if high_wind_days:
        recommendations.append(f"High winds on {', '.join(high_wind_days[:2])} — "
                                "allow extra travel time.")
    else:
        recommendations.append("No significant wind disruptions expected.")

    # Clothing advice
    if (tmax or 0) > 33:
        recommendations.append("Wear light, breathable clothing. Stay hydrated during peak heat hours (11 AM–4 PM).")
    else:
        recommendations.append("Light casuals should be comfortable throughout the day.")

    if rainy:
        recommendations.append("Carry an umbrella from " + rainy[0] + " onward.")

    # ── Build report string ───────────────────────────────────────
    lines = [
        "=" * 52,
        f"  WEATHER INSIGHTS REPORT — {city.upper()}",
        "=" * 52,
        "",
        "Temperature Trend:",
        f"  {_TREND_LABEL.get(t_trend, t_trend)}  (range: {temp_range})",
        "",
        "Humidity Trend:",
        f"  {_TREND_LABEL.get(h_trend, h_trend)}  (avg: {hum_str})",
        "",
        "Rain Forecast:",
        f"  {_RAIN_LABEL.get(r_trend, '')}",
        "",
        "Key Insights:",
    ]
    for ins in insights:
        lines.append(f"  • {ins}")

    lines += ["", "Recommendations:"]
    for rec in recommendations:
        lines.append(f"  • {rec}")

    lines += ["", "=" * 52]
    return "\n".join(lines)


# ── LangChain tool ────────────────────────────────────────────────────────────

@tool
def generate_weather_insights(analysis_json: str) -> str:
    """
    Generate a human-friendly Weather Insights Report from analysis data.

    Args:
        analysis_json: JSON string returned by analyze_weather.

    Returns:
        Formatted plain-text weather insights report, or error message.
    """
    try:
        analysis = json.loads(analysis_json)
        if "error" in analysis:
            return f"Could not generate report: {analysis['error']}"
        return _format_report(analysis)
    except Exception as exc:
        return f"Report generation failed: {exc}"
