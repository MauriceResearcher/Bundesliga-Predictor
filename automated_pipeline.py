"""

Ganze pipeline: Neuen Spieltag laden, In df einfügen, alles ausrechnen, zu nn schicken und trainieren, senden

"""

import os
import urllib.request
import pandas as pd
import requests
from dotenv import load_dotenv

# Eigene Module importieren
from feature_generation.compute_elo import compute_elo_rating
from feature_generation.compute_form_features import (
    build_full_featured_dataframe,
)
from generate_datasets.generate_whole_dataset import process_all_raw_files
from neural_network import predict_fixtures, train_and_predict
from send_email import send_prediction_email

load_dotenv()
MAIL_ADDRESS = os.getenv("GMAIL")

DOWNLOAD_DIR = r"C:\Users\mauri\Downloads"
DATASET_DIR = r"D:\PycharmProjects\Bundesliga\Datasets"
PROCESSED_DF_PATH = os.path.join(DATASET_DIR, "bundesliga_processed.csv")

# Mapping von OpenLigaDB-Namen auf D1.csv/Standard-Namen
TEAM_NAME_MAPPING = {
    "FC Bayern München": "Bayern Munich",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "RB Leipzig": "RB Leipzig",
    "VfB Stuttgart": "Stuttgart",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "TSG Hoffenheim": "Hoffenheim",
    "SC Freiburg": "Freiburg",
    "1. FC Heidenheim": "Heidenheim",
    "SV Werder Bremen": "Werder Bremen",
    "VfL Wolfsburg": "Wolfsburg",
    "FC Augsburg": "Augsburg",
    "Borussia Mönchengladbach": "M'gladbach",
    "1. FC Union Berlin": "Union Berlin",
    "VfL Bochum": "Bochum",
    "FC St. Pauli": "St Pauli",
    "Holstein Kiel": "Holstein Kiel",
    "1. FSV Mainz 05": "Mainz",
    "1. FC Köln": "FC Koln",
    "FC Schalke 04": "Schalke 04",
    "Hamburger SV": "Hamburg",
    "SC Paderborn 07": "Paderborn",
    "SV 07 Elversberg": "Elversberg",
}


# 1. Aktuelle D1.csv herunterladen (für aktuelle Saison 2026)
def download_latest_d1(target_dir):
    url = "https://www.football-data.co.uk/mmz4281/2627/D1.csv"
    file_path = os.path.join(target_dir, "D1.csv")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with (
            urllib.request.urlopen(req) as response,
            open(file_path, "wb") as out_file,
        ):
            out_file.write(response.read())
        print("D1.csv erfolgreich aktualisiert.")
        return file_path
    except Exception as e:
        print(f"Fehler beim Download der D1.csv: {e}")
        return None


# 2. OpenLigaDB als Quelle für kommende Paarungen
def get_next_matchday_from_openliga(target_season=2026, target_matchday=None):
    url = f"https://api.openligadb.de/getmatchdata/bl1/{target_season}"
    res = requests.get(url)
    if res.status_code != 200:
        print(f"Fehler bei OpenLigaDB API-Abfrage: {res.status_code}")
        return pd.DataFrame()

    matches = res.json()
    upcoming = []

    for m in matches:
        matchday = m.get("group", {}).get("groupOrderID")

        if target_matchday is None or matchday == target_matchday:
            home_raw = m.get("team1", {}).get("teamName")
            away_raw = m.get("team2", {}).get("teamName")

            upcoming.append(
                {
                    "date": m.get("matchDateTime"),
                    "season": target_season,
                    "matchday": matchday,
                    "HomeTeam": TEAM_NAME_MAPPING.get(home_raw, home_raw),
                    "AwayTeam": TEAM_NAME_MAPPING.get(away_raw, away_raw),
                    "FTHG": None,
                    "FTAG": None,
                    "FTR": None,
                }
            )

    return pd.DataFrame(upcoming)


# --- MAIN PIPELINE ---
def run_pipeline():
    # A. Aktuelle D1.csv herunterladen
    download_latest_d1(DOWNLOAD_DIR)

    # B. Alle Rohdateien verarbeiten und zu 'bundesliga_all_seasons.csv' zusammenfügen
    process_all_raw_files(DATASET_DIR)

    # C. Kombiniertes Dataset laden
    all_seasons_path = os.path.join(DATASET_DIR, "bundesliga_all_seasons.csv")
    d1_df = pd.read_csv(all_seasons_path)

    # Spaltennamen zur Sicherheit auf Kleinschreibung prüfen/harmonisieren
    matchday_col = "matchday" if "matchday" in d1_df.columns else "Matchday"
    season_col = "season" if "season" in d1_df.columns else "Season"

    # D. Ermitteln des nächsten Spieltags für die AKTUELLE Saison (2026)
    season_2026_df = d1_df[d1_df[season_col] == 2026]
    current_max_spieltag = (
        season_2026_df[matchday_col].max() if not season_2026_df.empty else 0
    )
    next_matchday = int(current_max_spieltag + 1)
    print(f"Nächster vorherzusagender Spieltag in Saison 2026: {next_matchday}")

    # E. Kommenden Spieltag via OpenLigaDB abfragen
    next_fixtures_df = get_next_matchday_from_openliga(
        target_season=2026, target_matchday=next_matchday
    )

    if next_fixtures_df.empty:
        print(
            f"Keine Paarungen für Spieltag {next_matchday} auf OpenLigaDB gefunden."
        )
        return

    # --- KORREKTUR: Spalten von OpenLigaDB an d1_df angleichen ---
    column_mapping = {
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "FTR",
    }
    next_fixtures_df = next_fixtures_df.rename(columns=column_mapping)

    # team_id und fehlende Grundspalten ergänzen
    if "home_team_id" not in next_fixtures_df.columns:
        next_fixtures_df["home_team_id"] = next_fixtures_df["home_team"]
    if "away_team_id" not in next_fixtures_df.columns:
        next_fixtures_df["away_team_id"] = next_fixtures_df["away_team"]

    # F. Datensätze zusammenführen
    full_raw_df = pd.concat([d1_df, next_fixtures_df], ignore_index=True)

    # Duplikate entfernen (auf Basis einheitlicher Spaltennamen)
    full_raw_df = full_raw_df.drop_duplicates(
        subset=[season_col, matchday_col, "home_team", "away_team"],
        keep="last",
    )

    # G. Feature Engineering & Elo über GESAMTES DataFrame berechnen
    df_with_form = build_full_featured_dataframe(full_raw_df, 5)
    df_processed = compute_elo_rating(
        df_with_form,
        initial_elo=1500,
        k_factor=20,
        mean_reversion=0.3,
        base_ha=60,
    )

    # H. Aufteilen in Trainingsdaten und Vorhersagedaten
    train_df = df_processed[df_processed["result"].notna()]
    predict_df = df_processed[
        (df_processed[season_col] == 2026)
        & (df_processed[matchday_col] == next_matchday)
    ]

    # I. Modell trainieren & Vorhersage erstellen
    predictions = predict_fixtures(train_df, predict_df)
    print(predictions.to_string())

    # J. Ergebnisse per E-Mail versenden
    send_prediction_email(predictions, MAIL_ADDRESS)

    # K. Verarbeitetes Gesamt-DF abspeichern
    df_processed.to_csv(PROCESSED_DF_PATH, index=False)
    print("Pipeline erfolgreich durchgelaufen!")


if __name__ == "__main__":
    run_pipeline()