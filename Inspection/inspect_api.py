import json
import os

import pandas as pd
import requests
from dotenv import load_dotenv

# ------------------------------------------------------------------
# 1. SETUP
# ------------------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"

if not API_KEY:
    raise ValueError("API_KEY nicht gefunden!")

HEADERS = {
    "x-apisports-key": API_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io",
}

OUTPUT_DIR = r"/Datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 2. HILFSFUNKTIONEN
# ------------------------------------------------------------------
def fetch_api_data(endpoint, params=None):
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, headers=HEADERS, params=params, timeout=10)

    remaining = response.headers.get("x-ratelimit-requests-remaining")
    print(f"[API Call] Verbleibende Tages-Requests: {remaining}")

    data = response.json()
    return data.get("response", [])


# ------------------------------------------------------------------
# 3. VORSCHAU DRUCKEN UND GANZE SAISON ALS CSV SPEICHERN
# ------------------------------------------------------------------
print("Starte API-Abfrage für die gesamte Saison 2024...")

# Wir fragen die GANZE Saison ab (ohne 'round'-Einschränkung) -> Kostet exakt 1 Request!
raw_data = fetch_api_data("fixtures", {"league": 78, "season": 2024})

print(f"Erfolgreich geladen! {len(raw_data)} Spiele gefunden.\n")

if raw_data:
    # --- TEIL A: FEATURE-VORSCHAU IN DER KONSOLE ---
    print(
        "================ VORSCHAU DER JSON-STRUKTUR (1. Spiel) ================"
    )
    print(json.dumps(raw_data[0], indent=4))
    print(
        "=======================================================================\n"
    )

    # --- TEIL B: ENTCHACHTELN & ALS CSV SPEICHERN ---
    parsed_matches = []
    for item in raw_data:
        fixture = item.get("fixture", {})
        league = item.get("league", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        score = item.get("score", {})

        home_goals = goals.get("home")
        away_goals = goals.get("away")

        # Label für ML berechnen
        if home_goals is None or away_goals is None:
            result = None
        elif home_goals > away_goals:
            result = 0  # Heimsieg
        elif home_goals == away_goals:
            result = 1  # Unentschieden
        else:
            result = 2  # Auswärtssieg

        parsed_matches.append({
            "fixture_id": fixture.get("id"),
            "date": fixture.get("date"),
            "timestamp": fixture.get("timestamp"),
            "referee": fixture.get("referee"),
            "season": league.get("season"),
            "round": league.get("round"),
            "home_team_id": teams.get("home", {}).get("id"),
            "home_team": teams.get("home", {}).get("name"),
            "away_team_id": teams.get("away", {}).get("id"),
            "away_team": teams.get("away", {}).get("name"),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "halftime_home_goals": score.get("halftime", {}).get("home"),
            "halftime_away_goals": score.get("halftime", {}).get("away"),
            "result": result,
            "status": fixture.get("status", {}).get("short"),
        })

    df = pd.DataFrame(parsed_matches)

    # Speichern als CSV
    csv_path = os.path.join(OUTPUT_DIR, "inspect_season_2024.csv")
    df.to_csv(csv_path, index=False)
    print(
        f" CSV mit {len(df)} Zeilen wurde gespeichert unter:\n {csv_path}"
    )