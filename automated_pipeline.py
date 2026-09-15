"""

Ganze pipeline: Neuen Spieltag laden, In df einfügen, alles ausrechnen, zu nn schicken und trainieren, senden

"""

import os
import urllib.request
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv

from feature_generation.compute_elo import compute_elo_rating
from feature_generation.compute_form_features import build_full_featured_dataframe
from generate_datasets.generate_whole_dataset import process_all_raw_files
from send_email import send_prediction_email
from train_models import train_and_predict_multi_models

# ... deine bisherigen Modul-Imports ...

load_dotenv()
MAIL_ADDRESS = os.getenv("GMAIL")

# --- RELATIVE PFADE & ORDNERSTRUKTUR ---
# Ermittelt das Projekt-Hauptverzeichnis (1 Ebene höher, falls dieses Skript im Hauptordner liegt)
BASE_DIR = Path(__file__).resolve().parent

# Pfade relativ zum Projektverzeichnis definieren
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_DIR = DATA_DIR / "datasets"
PROCESSED_DF_PATH = DATASET_DIR / "bundesliga_processed.csv"

# Ordner automatisch erstellen, falls sie lokal oder bei GitHub Actions noch nicht existieren
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# 1. Aktuelle D1.csv herunterladen
def download_latest_d1(target_dir: Path):
    url = "https://www.football-data.co.uk/mmz4281/2627/D1.csv"
    file_path = target_dir / "D1.csv"
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

# 1. Aktuelle D1.csv herunterladen
def download_latest_d1(target_dir: Path):
    url = "https://www.football-data.co.uk/mmz4281/2627/D1.csv"
    file_path = target_dir / "D1.csv"
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
    download_latest_d1(RAW_DATA_DIR)

    # B. Alle Rohdateien verarbeiten und zu 'bundesliga_all_seasons.csv' zusammenfügen
    process_all_raw_files(str(DATASET_DIR))

    # C. Kombiniertes Dataset laden
    all_seasons_path = DATASET_DIR / "bundesliga_all_seasons.csv"
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

    # Verarbeiteten DataFrame lokal als CSV speichern, damit du ihn analysieren kannst <---
    PROCESSED_DF_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(PROCESSED_DF_PATH, index=False)
    print(f"Verarbeiteter DataFrame erfolgreich gespeichert unter: {PROCESSED_DF_PATH}")

    # H. Aufteilen in Trainingsdaten und Vorhersagedaten
    train_df = df_processed[df_processed["result"].notna()]
    predict_df = df_processed[
        (df_processed[season_col] == 2026)
        & (df_processed[matchday_col] == next_matchday)
    ]

    # I. Modell trainieren & Vorhersage erstellen
    print(
        f"Trainiere Ensemble (NN, RF, XGBoost) auf {len(train_df)} absolvierten Partien..."
    )
    predictions, performances, _ = train_and_predict_multi_models(
        train_df, predict_df
    )

    # Konsole-Ausgabe der Performance-Metriken (Train- & Val-Accuracy)
    print("\n--- MODELL PERFORMANCE (HISTORISCHES VALIDATION SET) ---")
    for m_name, perf in performances.items():
        print(
            f"{m_name:15} | Train Acc: {perf['train_acc'] * 100:.1f}% | Val Acc: {perf['val_acc'] * 100:.1f}% | Val MAE Tore: {perf['val_mae']:.2f}"
        )

    print("\n--- VORHERSAGEN FÜR SPIELTAG ---")
    print(predictions.to_string())

    # J. Ergebnisse per E-Mail versenden (inkl. Performance-Daten)
    send_prediction_email(predictions, performances, MAIL_ADDRESS)


if __name__ == "__main__":
    run_pipeline()