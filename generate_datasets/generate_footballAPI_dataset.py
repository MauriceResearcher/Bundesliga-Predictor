import os
import time
from dotenv import load_dotenv
import pandas as pd
import requests

# ------------------------------------------------------------------
# 1. SETUP & ENVIRONMENT
# ------------------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"

if not API_KEY:
    raise ValueError(
        "API_KEY nicht gefunden! Bitte in der .env-Datei 'API_FOOTBALL_KEY=dein_key' eintragen."
    )

HEADERS = {
    "x-apisports-key": API_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io",
}

OUTPUT_DIR = r"/Datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 2. SAFE API REQUEST (MIT RATE LIMIT HANDLING)
# ------------------------------------------------------------------
def fetch_api_data_safe(endpoint, params=None):
    """Führt einen API-Call durch und wartet automatisch bei Rate-Limit-Fehlern."""
    url = f"{BASE_URL}/{endpoint}"

    while True:
        try:
            response = requests.get(
                url, headers=HEADERS, params=params, timeout=10
            )
            remaining = response.headers.get("x-ratelimit-requests-remaining")
            print(f"[API Call] /{endpoint} | Verbleibende Tages-Calls: {remaining}")

            data = response.json()
            errors = data.get("errors", {})

            # 1. Per-Minute Limit abgefangen -> Warten und erneut versuchen
            if isinstance(errors, dict) and "rateLimit" in errors:
                print(
                    " Per-Minute Rate-Limit erreicht. Warte 10 Sekunden..."
                )
                time.sleep(10)
                continue

            # 2. Andere API Fehler (z.B. Plan Restriction)
            if errors and len(errors) > 0:
                print(f" API Fehler: {errors}")
                return None

            return data.get("response", [])

        except Exception as e:
            print(f" Netzwerkfehler: {e}")
            return None


def get_team_stats_dict(season=2024, league_id=78, team_ids=[]):
    """Holt Saison-Statistiken für jedes Team mit Rate-Limit-Pause."""
    print(f"\n--- Lade Team-Statistiken für {len(team_ids)} Teams ---")
    team_stats = {}

    for team_id in team_ids:
        raw_stats = fetch_api_data_safe(
            "teams/statistics",
            {"league": league_id, "season": season, "team": team_id},
        )
        if raw_stats:
            goals = raw_stats.get("goals", {})
            clean_sheets = raw_stats.get("clean_sheet", {}).get("total")

            team_stats[team_id] = {
                "goals_for_avg_home": goals.get("for", {})
                .get("average", {})
                .get("home"),
                "goals_against_avg_home": goals.get("against", {})
                .get("average", {})
                .get("home"),
                "goals_for_avg_away": goals.get("for", {})
                .get("average", {})
                .get("away"),
                "goals_against_avg_away": goals.get("against", {})
                .get("average", {})
                .get("away"),
                "clean_sheets_total": clean_sheets,
            }

        # 6.5s Pause -> Maximal 9 Calls pro Minute zur absoluten Sicherheit
        time.sleep(6.5)

    return team_stats


# ------------------------------------------------------------------
# 4. HOUPTFUNKTION: DATENSATZ GENERIEREN
# ------------------------------------------------------------------
def generate_complete_dataset(season=2024, league_id=78):
    print(f"\n1. Hole Spielplan für Saison {season}...")
    raw_matches = fetch_api_data_safe(
        "fixtures", {"league": league_id, "season": season}
    )

    if not raw_matches:
        print("Keine Spiele gefunden!")
        return

    team_ids = list(
        set([item.get("teams", {}).get("home", {}).get("id") for item in raw_matches])
    )

    # 2. Team Stats holen (18 Teams * 6.5s ~ 2 Minuten)
    team_stats_data = get_team_stats_dict(
        season=season, league_id=league_id, team_ids=team_ids
    )


    parsed_dataset = []

    for item in raw_matches:
        fixture = item.get("fixture", {})
        league = item.get("league", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        score = item.get("score", {})

        f_id = fixture.get("id")
        home_id = teams.get("home", {}).get("id")
        away_id = teams.get("away", {}).get("id")

        home_goals = goals.get("home")
        away_goals = goals.get("away")

        if home_goals is None or away_goals is None:
            result = None
        elif home_goals > away_goals:
            result = 0
        elif home_goals == away_goals:
            result = 1
        else:
            result = 2


        home_stats = team_stats_data.get(home_id, {})
        away_stats = team_stats_data.get(away_id, {})

        row = {
            "fixture_id": f_id,
            "date": fixture.get("date"),
            "season": league.get("season"),
            "round": league.get("round"),
            "referee": fixture.get("referee"),
            "home_team_id": home_id,
            "home_team": teams.get("home", {}).get("name"),
            "away_team_id": away_id,
            "away_team": teams.get("away", {}).get("name"),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "halftime_home_goals": score.get("halftime", {}).get("home"),
            "halftime_away_goals": score.get("halftime", {}).get("away"),
            "result": result,
            "home_avg_goals_scored": home_stats.get("goals_for_avg_home"),
            "home_avg_goals_conceded": home_stats.get("goals_against_avg_home"),
            "home_clean_sheets": home_stats.get("clean_sheets_total"),
            "away_avg_goals_scored": away_stats.get("goals_for_avg_away"),
            "away_avg_goals_conceded": away_stats.get("goals_against_avg_away"),
            "away_clean_sheets": away_stats.get("clean_sheets_total"),
        }
        parsed_dataset.append(row)

    df = pd.DataFrame(parsed_dataset)
    save_path = os.path.join(
        OUTPUT_DIR, f"bundesliga_full_dataset_{season}.csv"
    )
    df.to_csv(save_path, index=False)
    print(
        f"\n Fertig! Datensatz ({len(df)} Zeilen, {len(df.columns)} Spalten) sauber gespeichert unter:\n {save_path}"
    )


generate_complete_dataset(season=2025)