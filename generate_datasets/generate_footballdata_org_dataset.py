import os
import time
from dotenv import load_dotenv
import pandas as pd
import requests

# ------------------------------------------------------------------
# 1. SETUP & ENVIRONMENT
# ------------------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_KEY")
BASE_URL = "https://api.football-data.org/v4"  # Die Backend-API von Native-Stats

if not API_KEY:
    raise ValueError(
        "API_KEY nicht gefunden! Bitte 'FOOTBALL_DATA_KEY' in der .env-Datei eintragen."
    )

HEADERS = {
    "X-Auth-Token": API_KEY,
}

OUTPUT_DIR = r"D:\PycharmProjects\Bundesliga\Datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 2. HELPER: SAFE API REQUEST
# ------------------------------------------------------------------
def fetch_data(endpoint, params=None):
    """Führt eine API-Anfrage aus und berücksichtigt Rate-Limits."""
    url = f"{BASE_URL}/{endpoint}"
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)

        if response.status_code == 429:
            print(" Rate Limit erreicht! Warte 60 Sekunden...")
            time.sleep(60)
            return fetch_data(endpoint, params)

        if response.status_code != 200:
            print(f" Fehler HTTP {response.status_code}: {response.text}")
            return None

        return response.json()
    except Exception as e:
        print(f" Netzwerkfehler: {e}")
        return None


# ------------------------------------------------------------------
# 3. SPIELPLAN & MATCH-DATEN ABHOLEN
# ------------------------------------------------------------------
def get_matches_dataset(season=2025, league_code="BL1"):
    print(f"\n1. Lade Match-Daten für {league_code} (Saison {season})...")

    data = fetch_data(f"competitions/{league_code}/matches", {"season": season})
    if not data or "matches" not in data:
        print("Keine Matches gefunden!")
        return None

    parsed_rows = []
    for match in data["matches"]:
        # Quoten (odds) abgreifen, falls von der API mitgeliefert
        odds = match.get("odds", {})

        score = match.get("score", {})
        full_time = score.get("fullTime", {})
        half_time = score.get("halfTime", {})

        home_goals = full_time.get("home")
        away_goals = full_time.get("away")

        # Result-Label: 0=Heimsieg, 1=Unentschieden, 2=Auswärtssieg
        if home_goals is None or away_goals is None:
            result = None
        elif home_goals > away_goals:
            result = 0
        elif home_goals == away_goals:
            result = 1
        else:
            result = 2

        parsed_rows.append(
            {
                "fixture_id": match.get("id"),
                "date": match.get("utcDate"),
                "matchday": match.get("matchday"),
                "status": match.get("status"),
                "home_team": match.get("homeTeam", {}).get("name"),
                "home_team_id": match.get("homeTeam", {}).get("id"),
                "away_team": match.get("awayTeam", {}).get("name"),
                "away_team_id": match.get("awayTeam", {}).get("id"),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "halftime_home": half_time.get("home"),
                "halftime_away": half_time.get("away"),
                "result": result,
                # Quoten-Spalten (Home, Draw, Away)
                "odds_home": odds.get("homeWin"),
                "odds_draw": odds.get("draw"),
                "odds_away": odds.get("awayWin"),
            }
        )

    df = pd.DataFrame(parsed_rows)
    save_path = os.path.join(OUTPUT_DIR, f"native_stats_matches_{season}.csv")
    df.to_csv(save_path, index=False)
    print(f" Match-Daten gespeichert ({len(df)} Spiele): {save_path}")
    return df


# ------------------------------------------------------------------
# 4. TABELLE / STANDINGS ABHOLEN
# ------------------------------------------------------------------
def get_standings_dataset(season=2025, league_code="BL1"):
    print(f"\n2. Lade aktuelle Tabelle für {league_code}...")

    data = fetch_data(f"competitions/{league_code}/standings", {"season": season})
    if not data or "standings" not in data:
        print("Keine Tabellendaten gefunden!")
        return None

    standings_rows = []
    # Die Haupttabelle ist meistens der erste Eintrag ("TOTAL")
    total_table = data["standings"][0].get("table", [])

    for entry in total_table:
        team = entry.get("team", {})
        standings_rows.append(
            {
                "position": entry.get("position"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "played_games": entry.get("playedGames"),
                "won": entry.get("won"),
                "draw": entry.get("draw"),
                "lost": entry.get("lost"),
                "points": entry.get("points"),
                "goals_for": entry.get("goalsFor"),
                "goals_against": entry.get("goalsAgainst"),
                "goal_difference": entry.get("goalDifference"),
            }
        )

    df_standings = pd.DataFrame(standings_rows)
    save_path = os.path.join(OUTPUT_DIR, f"native_stats_standings_{season}.csv")
    df_standings.to_csv(save_path, index=False)
    print(f" Tabellendaten gespeichert ({len(df_standings)} Teams): {save_path}")
    return df_standings


# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
if __name__ == "__main__":
    SEASON = 2026

    # 1. Daten holen
    get_matches_dataset(season=SEASON)

    # 2. Daten laden & Säubern
    csv_path = os.path.join(OUTPUT_DIR, f"native_stats_matches_{SEASON}.csv")
    df = pd.read_csv(csv_path)

    # Nur reguläre Spieltage (1 bis 34) behalten
    df_clean = df[(df["matchday"] >= 1) & (df["matchday"] <= 34)].copy()

    # Sicherstellen, dass 'season' existiert
    df_clean["season"] = SEASON

    # Sortieren nach Spieltag und Datum
    df_clean.sort_values(by=["matchday", "date"], inplace=True)

    # 3. Ergebnis speichern
    output_path = os.path.join(OUTPUT_DIR, f"bundesliga_{SEASON}_cleaned.csv")
    df_clean.to_csv(output_path, index=False)

    print(
        f"\n Erfolgreich! Bereinigt: {len(df)} -> {len(df_clean)} Zeilen."
    )
    print(f"Gespeichert unter: {output_path}")