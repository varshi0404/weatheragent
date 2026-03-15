"""
mood_forecast.py
----------------
LangChain tool: generate_mood_forecast(parsed_json)

Converts a list of daily weather records into a friendly, human-readable
"mood diary" — one 1-2 sentence entry per day that describes how the
weather will feel and suggests a gentle matching activity.

Accepts the same JSON shape produced by parse_weather_data:
    {"city": "...", "forecast": [ {date, temp_max, temp_min,
      humidity_avg, rain_probability, wind_speed_max, cloud_cover_avg}, ... ]}

Can also accept raw daily records directly (list at top level), making the
tool usable standalone without the full pipeline:
    [{"date": "2026-03-16", "temperature_max": 32, ...}, ...]

Field aliases handled automatically:
    temperature_max  ↔  temp_max
    temperature_min  ↔  temp_min
    humidity         ↔  humidity_avg
    wind_speed       ↔  wind_speed_max
    cloud_cover      ↔  cloud_cover_avg
    rain_probability ↔  rain_probability   (same in both schemas)

Importable as a plain Python function (no LangChain dependency needed):
    from insights.mood_forecast import build_mood_forecast
    text = build_mood_forecast("Chennai", records)

Or as a LangChain @tool:
    from insights.mood_forecast import generate_mood_forecast
    # register in TOOLS list inside agent.py
"""

from __future__ import annotations

import json
from datetime import datetime
try:
    from langchain.tools import tool
except ImportError:  # fallback when langchain isn't installed
    def tool(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func


# ── Field normaliser ──────────────────────────────────────────────────────────

def _normalise(raw: dict) -> dict:
    """
    Accept both the pipeline schema (temp_max, humidity_avg …)
    and the prompt spec schema (temperature_max, humidity …)
    and return a unified dict.
    """
    def pick(*keys):
        for k in keys:
            if raw.get(k) is not None:
                return raw[k]
        return None

    return {
        "date":             raw.get("date", "Unknown date"),
        "temp_max":         pick("temp_max",  "temperature_max"),
        "temp_min":         pick("temp_min",  "temperature_min"),
        "humidity":         pick("humidity_avg", "humidity"),
        "rain_probability": pick("rain_probability"),
        "wind_speed":       pick("wind_speed_max", "wind_speed"),
        "cloud_cover":      pick("cloud_cover_avg", "cloud_cover"),
    }


# ── Mood classification helpers ───────────────────────────────────────────────

def _classify_sky(cloud: float | None, rain: float | None) -> str:
    rain  = rain  or 0
    cloud = cloud or 0
    if rain > 70:                      return "rainy"
    if rain > 40:                      return "likely_rainy"
    if cloud > 75:                     return "overcast"
    if cloud > 40:                     return "partly_cloudy"
    return "sunny"

def _classify_heat(tmax: float | None) -> str:
    if tmax is None:                   return "mild"
    if tmax >= 38:                     return "scorching"
    if tmax >= 35:                     return "hot"
    if tmax >= 30:                     return "warm"
    if tmax >= 22:                     return "pleasant"
    if tmax >= 15:                     return "cool"
    return "cold"

def _classify_wind(wind: float | None) -> str:
    wind = wind or 0
    if wind >= 50:                     return "stormy"
    if wind >= 30:                     return "windy"
    if wind >= 15:                     return "breezy"
    return "calm"

def _classify_humidity(hum: float | None) -> str:
    hum = hum or 0
    if hum >= 85:                      return "oppressive"
    if hum >= 70:                      return "muggy"
    if hum >= 50:                      return "comfortable"
    return "dry"


# ── Sentence bank ─────────────────────────────────────────────────────────────
# Each entry is (mood_sentence, activity_suggestion).
# Keys are tuples of (sky, heat, wind) — match most-specific first.

_MOODS: dict[tuple, tuple[str, str]] = {

    # ── Rainy combos ──────────────────────────────────────────────────────────
    ("rainy", "scorching", "stormy"): (
        "A stormy, sweltering day — the rain might cool things a little but it'll feel thick and heavy.",
        "Best to stay safely indoors; crack a window and enjoy the dramatic skies with something cold to drink.",
    ),
    ("rainy", "hot", "windy"): (
        "Hot, windy rain — the kind that soaks you sideways.",
        "A great excuse to stay in and catch up on that show you've been meaning to watch.",
    ),
    ("rainy", "warm", "calm"): (
        "A warm, gentle rainy day that feels almost tropical.",
        "Curl up near a window with a good book — the sound of soft rain is perfect company.",
    ),
    ("rainy", "cool", "calm"): (
        "A cool, steady rainy day — moody in the best way.",
        "Brew a big pot of tea, light a candle, and let the rain do the rest.",
    ),
    ("rainy", "cold", "windy"): (
        "Cold, blustery rain — not the day for adventures outside.",
        "Stay cosy indoors; thick socks and comfort food are fully justified today.",
    ),
    ("likely_rainy", "warm", "calm"): (
        "Rain is possible — the air feels heavy and the clouds are building.",
        "Keep an umbrella close just in case, but light tasks outside early in the day should be fine.",
    ),
    ("likely_rainy", "hot", "breezy"): (
        "A humid, unsettled day with showers likely later.",
        "Get any outdoor errands done in the morning before the sky decides to open up.",
    ),
    ("likely_rainy", "cool", "windy"): (
        "Grey and breezy with on-and-off drizzle likely.",
        "Layer up if you're heading out — a light rain jacket will save the day.",
    ),

    # ── Overcast combos ───────────────────────────────────────────────────────
    ("overcast", "hot", "calm"): (
        "Thick clouds keep the sun away, but the heat lingers underneath.",
        "A shaded walk or a lazy afternoon at a café feels just right.",
    ),
    ("overcast", "warm", "breezy"): (
        "Soft grey skies and a cooling breeze — surprisingly refreshing.",
        "Great conditions for a longer walk or a bike ride without the glare of the sun.",
    ),
    ("overcast", "pleasant", "calm"): (
        "A calm, overcast day that feels oddly peaceful and unhurried.",
        "Perfect for a slow morning — maybe journaling outside or tending to the garden.",
    ),
    ("overcast", "cool", "windy"): (
        "Cool and windy under a thick blanket of cloud.",
        "If you go out, keep it brisk — you'll warm up quickly once you get moving.",
    ),

    # ── Partly cloudy combos ──────────────────────────────────────────────────
    ("partly_cloudy", "scorching", "calm"): (
        "A scorching day with clouds giving occasional blessed shade.",
        "Stay near water if you can — a pool, a fountain, or even a cold shower between tasks.",
    ),
    ("partly_cloudy", "hot", "breezy"): (
        "Warm and breezy with patches of cloud keeping the sun honest.",
        "Ideal for a picnic in a shady spot or a slow stroll in the late afternoon.",
    ),
    ("partly_cloudy", "warm", "calm"): (
        "A lovely, balanced day — not too hot, not too gloomy.",
        "Almost any outdoor plan will work beautifully today; enjoy it.",
    ),
    ("partly_cloudy", "pleasant", "breezy"): (
        "Comfortable temperatures with a gentle breeze and dappled sunshine.",
        "One of those rare days that's perfect for pretty much anything — make the most of it!",
    ),
    ("partly_cloudy", "cool", "windy"): (
        "Breezy and cool with bits of sun breaking through now and then.",
        "A scarf and a warm jacket will keep you comfortable for an outdoor walk.",
    ),

    # ── Sunny combos ─────────────────────────────────────────────────────────
    ("sunny", "scorching", "calm"): (
        "A blazing, sunny day — the kind that makes the pavement shimmer.",
        "Step outside early or after 5 PM; midday heat is best enjoyed from the shade.",
    ),
    ("sunny", "hot", "calm"): (
        "Clear skies and plenty of sunshine — bright and vibrant.",
        "Perfect for a morning walk, outdoor breakfast, or any activity before the peak afternoon heat.",
    ),
    ("sunny", "hot", "breezy"): (
        "Gloriously sunny with a breeze taking the edge off the heat.",
        "A beach day, a park visit, or any outdoor gathering will feel absolutely lovely.",
    ),
    ("sunny", "warm", "calm"): (
        "Sun-soaked and warm — a genuinely delightful day.",
        "Grab a friend, head outside, and make some good memories.",
    ),
    ("sunny", "pleasant", "calm"): (
        "Golden sunshine and mild temperatures — basically perfect.",
        "An ideal day for a long walk, gardening, or simply sitting outside with a coffee.",
    ),
    ("sunny", "pleasant", "breezy"): (
        "Crisp sunshine with a refreshing breeze — the kind of day that puts you in a good mood.",
        "Great for outdoor exercise, a farmers' market visit, or an afternoon cycle.",
    ),
    ("sunny", "cool", "calm"): (
        "Clear and cool — fresh air and bright light without any swelter.",
        "A brisk walk in the sunshine will do wonders for your energy today.",
    ),
    ("sunny", "cold", "calm"): (
        "Sunshine brightens the day but the cold air bites a little.",
        "Layer up and enjoy the light — a sunny cold day has its own cheerful magic.",
    ),
    ("sunny", "cold", "windy"): (
        "Bright blue skies, but the wind makes it feel sharp and raw.",
        "Best enjoyed through a window; if you do go out, bundle up properly.",
    ),
}

# Fallback for any combination not explicitly listed
_FALLBACK = (
    "A mixed kind of day with changeable conditions.",
    "Take it easy and adapt as the day unfolds — there's always something good to find in any weather.",
)


def _lookup_mood(sky: str, heat: str, wind: str) -> tuple[str, str]:
    """Try increasingly broad matches until a sentence pair is found."""
    key = (sky, heat, wind)
    if key in _MOODS:
        return _MOODS[key]

    # Drop wind specificity
    for w in ("calm", "breezy", "windy", "stormy"):
        if (sky, heat, w) in _MOODS:
            return _MOODS[(sky, heat, w)]

    # Drop heat specificity
    for h in ("pleasant", "warm", "cool"):
        if (sky, h, wind) in _MOODS:
            return _MOODS[(sky, h, wind)]

    # Sky only
    for entry_key, value in _MOODS.items():
        if entry_key[0] == sky:
            return value

    return _FALLBACK


# ── Humidity modifier ─────────────────────────────────────────────────────────

_HUMIDITY_SUFFIX: dict[str, str] = {
    "oppressive": " The humidity is truly oppressive — keep water handy and take it slow.",
    "muggy":      " It's a bit muggy, so you may feel more tired than usual.",
    "dry":        " The air is refreshingly dry today.",
    "comfortable": "",   # no modifier needed
}


# ── Date formatter ────────────────────────────────────────────────────────────

def _friendly_date(date_str: str) -> str:
    """Return e.g. 'Monday, March 16' from '2026-03-16'."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%A, %B %-d")   # Linux/Mac; Windows needs %#d
    except ValueError:
        return date_str


# ── Core builder (importable without LangChain) ───────────────────────────────

def build_mood_forecast(city: str, records: list[dict]) -> str:
    """
    Generate a friendly mood-diary forecast from a list of daily weather records.

    Parameters
    ----------
    city : str
        City name, used in the header.
    records : list[dict]
        Daily weather dicts — accepts both pipeline schema and prompt spec schema.

    Returns
    -------
    str
        Multi-line mood forecast, one entry per day.
    """
    if not records:
        return "No weather data provided — nothing to forecast!"

    lines: list[str] = [
        f"🌤  Mood Forecast — {city}",
        "─" * 44,
        "",
    ]

    for raw in records:
        day = _normalise(raw)

        sky  = _classify_sky(day["cloud_cover"], day["rain_probability"])
        heat = _classify_heat(day["temp_max"])
        wind = _classify_wind(day["wind_speed"])
        hum  = _classify_humidity(day["humidity"])

        mood_sentence, activity = _lookup_mood(sky, heat, wind)
        hum_suffix = _HUMIDITY_SUFFIX.get(hum, "")

        friendly = _friendly_date(day["date"])
        temp_str  = (
            f"{day['temp_max']:.0f}°C / {day['temp_min']:.0f}°C"
            if day["temp_max"] is not None and day["temp_min"] is not None
            else ""
        )
        rain_str = f"  🌧 {day['rain_probability']:.0f}% rain" if (day["rain_probability"] or 0) > 20 else ""

        # Assemble entry
        header = f"📅 {friendly}"
        if temp_str:
            header += f"  ({temp_str}){rain_str}"

        lines.append(header)
        lines.append(f"   {mood_sentence}{hum_suffix}")
        lines.append(f"   💡 {activity}")
        lines.append("")

    lines.append("─" * 44)
    return "\n".join(lines)


# ── LangChain @tool wrapper ───────────────────────────────────────────────────

@tool
def generate_mood_forecast(parsed_json: str) -> str:
    """
    Generate a friendly day-by-day mood forecast from weather data.

    Accepts either:
      - The JSON string returned by parse_weather_data
        ({"city": "...", "forecast": [...]})
      - A raw JSON list of daily records
        ([{"date": "...", "temperature_max": ..., ...}])

    Each day gets a 1-2 sentence human-friendly mood description and a
    gentle activity suggestion — no technical jargon.

    Args:
        parsed_json: JSON string — pipeline output or raw daily list.

    Returns:
        Plain-text mood diary, one entry per day.
    """
    try:
        data = json.loads(parsed_json)

        # Handle both input shapes
        if isinstance(data, list):
            # Raw list of records (prompt spec format)
            city    = "Your Location"
            records = data
        elif isinstance(data, dict):
            if "error" in data:
                return f"Cannot generate mood forecast: {data['error']}"
            city    = data.get("city", "Unknown")
            records = data.get("forecast", [])
        else:
            return "Unexpected input format — expected a JSON object or list."

        return build_mood_forecast(city, records)

    except json.JSONDecodeError as exc:
        return f"Could not parse input JSON: {exc}"
    except Exception as exc:
        return f"Mood forecast generation failed: {exc}"
