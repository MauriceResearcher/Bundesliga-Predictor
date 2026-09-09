import os
import glob
import pandas as pd
import numpy as np


def parse_season_file(file_path, season_year):
    """Lädt eine Football-Data CSV und konvertiert sie in das interne Pipeline-Format."""
    try:
        df = pd.read_csv(file_path, encoding="latin1")
    except Exception as e:
        print(f"Fehler beim Lesen von {file_path}: {e}")
        return None

    # Benötigte Mindestspalten prüfen
    required_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    if not all(col in df.columns for col in required_cols):
        print(
            f"Überspringe {file_path}: Fehlende Standardspalten ({df.columns.tolist()[:10]})"
        )
        return None

    # Nur gültige Zeilen behalten (ungespielte Spiele herausfiltern, falls vorhanden)
    df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()

    # 1. Ergebnis-Label mappen (0: Heimsieg, 1: Remis, 2: Auswärtssieg)
    result_map = {"H": 0, "D": 1, "A": 2}
    df["result"] = df["FTR"].map(result_map)

    # 2. Quoten ermitteln (Bet365 als Standard, sonst Marktdurchschnitt / Fallback)
    if "B365H" in df.columns:
        df["odds_home"] = df["B365H"]
        df["odds_draw"] = df["B365D"]
        df["odds_away"] = df["B365A"]
    elif "BbAvH" in df.columns:  # Ältere Saisons (Betbrain Average)
        df["odds_home"] = df["BbAvH"]
        df["odds_draw"] = df["BbAvD"]
        df["odds_away"] = df["BbAvA"]
    elif "AvgH" in df.columns:
        df["odds_home"] = df["AvgH"]
        df["odds_draw"] = df["AvgD"]
        df["odds_away"] = df["AvgA"]
    else:
        df["odds_home"] = np.nan
        df["odds_draw"] = np.nan
        df["odds_away"] = np.nan

    # 3. Datum parsen
    df["date"] = pd.to_datetime(
        df["Date"], format="%d/%m/%y", errors="coerce"
    ).fillna(
        pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    )
    df = df.sort_values("date").reset_index(drop=True)

    # 4. Spieltag (matchday) berechnen (1 bis 34)
    # Zählt, das wievielte Spiel jedes Team in der Saison absolviert
    team_match_count = {}
    matchday_list = []

    for idx, row in df.iterrows():
        h_team = row["HomeTeam"]
        a_team = row["AwayTeam"]

        team_match_count[h_team] = team_match_count.get(h_team, 0) + 1
        team_match_count[a_team] = team_match_count.get(a_team, 0) + 1

        # Der Spieltag ist das Maximum der bisherigen Spiele der beiden Teams
        matchday_list.append(
            max(team_match_count[h_team], team_match_count[a_team])
        )

    df["matchday"] = matchday_list
    df["season"] = season_year

    # 5. Einheitliche Spaltenstruktur erstellen
    clean_df = pd.DataFrame(
        {
            "fixture_id": [f"{season_year}_{i+1}" for i in range(len(df))],
            "date": df["date"],
            "season": df["season"],
            "matchday": df["matchday"],
            "home_team": df["HomeTeam"],
            "home_team_id": df[
                "HomeTeam"
            ],  # Vereinsnamen dienen als eindeutige ID
            "away_team": df["AwayTeam"],
            "away_team_id": df["AwayTeam"],
            "home_goals": df["FTHG"],
            "away_goals": df["FTAG"],
            "halftime_home": df.get("HTHG", np.nan),
            "halftime_away": df.get("HTAG", np.nan),
            "result": df["result"],
            "odds_home": df["odds_home"],
            "odds_draw": df["odds_draw"],
            "odds_away": df["odds_away"],
            # Erweiterte Match-Statistiken aus Football-Data (falls vorhanden)
            "home_shots": df.get("HS", np.nan),
            "away_shots": df.get("AS", np.nan),
            "home_shots_target": df.get("HST", np.nan),
            "away_shots_target": df.get("AST", np.nan),
        }
    )

    return clean_df


def process_all_raw_files(dataset_dir):
    """Verarbeitet alle Bundesliga_raw CSVs und D1.csv chronologisch."""
    file_season_mapping = {
        r"C:\Users\mauri\Downloads\D1.csv": 2026,  # Aktuelle Saison 2026/2027
        r"C:\Users\mauri\Downloads\Bundesliga_raw (1).csv": 2025,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (2).csv": 2024,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (3).csv": 2023,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (4).csv": 2022,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (5).csv": 2021,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (6).csv": 2020,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (7).csv": 2019,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (8).csv": 2018,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (9).csv": 2017,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (10).csv": 2016,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (11).csv": 2015,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (12).csv": 2014,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (13).csv": 2013,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (14).csv": 2012,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (15).csv": 2011,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (16).csv": 2010,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (17).csv": 2009,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (18).csv": 2008,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (19).csv": 2007,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (20).csv": 2006,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (21).csv": 2005,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (22).csv": 2004,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (23).csv": 2003,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (24).csv": 2002,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (25).csv": 2001,
        r"C:\Users\mauri\Downloads\Bundesliga_raw (26).csv": 2000,
    }

    all_dfs = []

    for filename, season in file_season_mapping.items():
        file_path = os.path.join(dataset_dir, filename)
        if os.path.exists(file_path):
            print(f"Verarbeite {filename} (Saison {season})...")
            df_season = parse_season_file(file_path, season)
            if df_season is not None:
                all_dfs.append(df_season)
        else:
            print(f"Datei nicht gefunden: {filename}")

    if not all_dfs:
        print("Keine Dateien verarbeitet!")
        return

    # Alle Saisons vertikal zusammenfügen
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Nach Saison, Spieltag und Datum sortieren
    combined_df = combined_df.sort_values(
        by=["season", "matchday", "date"]
    ).reset_index(drop=True)

    # Rohdaten aller Saisons speichern
    output_path = os.path.join(dataset_dir, "bundesliga_all_seasons.csv")
    combined_df.to_csv(output_path, index=False)
    print(
        f"\nErfolgreich {len(combined_df)} Spiele aus {len(all_dfs)} Saisons kombiniert!"
    )
    print(f"Gespeichert unter: {output_path}")


if __name__ == "__main__":
    DATASET_DIR = r"D:\PycharmProjects\Bundesliga\Datasets"
    process_all_raw_files(DATASET_DIR)