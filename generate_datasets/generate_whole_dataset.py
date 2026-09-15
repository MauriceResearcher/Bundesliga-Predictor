import glob
import os
import numpy as np
import pandas as pd
import urllib.request

DOWNLOAD_DIR = r"C:\Users\mauri\Downloads"

def download_D1(directory):
    url = "https://www.football-data.co.uk/mmz4281/2627/D1.csv"
    file_path = os.path.join(directory, "D1.csv")
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


download_D1(DOWNLOAD_DIR)


def parse_season_file(file_path, season_year):
    """Lädt eine Football-Data CSV und konvertiert sie in das interne Pipeline-Format."""
    try:
        # engine='python' und on_bad_lines='skip' fangen überflüssige Kommata am Zeilenende ab
        df = pd.read_csv(
            file_path, encoding="latin1", on_bad_lines="skip", engine="python"
        )
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

    # Nur gültige Zeilen behalten
    df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()

    # 1. Ergebnis-Label mappen (0: Heimsieg, 1: Remis, 2: Auswärtssieg)
    result_map = {"H": 0, "D": 1, "A": 2}
    df["result"] = df["FTR"].map(result_map)

    # 2. Quoten ermitteln
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
    team_match_count = {}
    matchday_list = []

    for idx, row in df.iterrows():
        h_team = row["HomeTeam"]
        a_team = row["AwayTeam"]

        team_match_count[h_team] = team_match_count.get(h_team, 0) + 1
        team_match_count[a_team] = team_match_count.get(a_team, 0) + 1

        matchday_list.append(
            max(team_match_count[h_team], team_match_count[a_team])
        )

    df["matchday"] = matchday_list
    df["season"] = season_year

    # 5. Dein definiertes Wunscheschema
    clean_df = pd.DataFrame(
        {
            "fixture_id": [f"{season_year}_{i + 1}" for i in range(len(df))],
            "date": df["date"],
            "season": df["season"],
            "matchday": df["matchday"],
            "home_team": df["HomeTeam"],
            "home_team_id": df["HomeTeam"],
            "away_team": df["AwayTeam"],
            "away_team_id": df["AwayTeam"],
            "home_goals": df["FTHG"],
            "away_goals": df["FTAG"],
            "halftime_home": df.get("HTHG", np.nan),
            "halftime_away": df.get("HTAG", np.nan),
            "result": df["result"],

            # Quoten (1X2 & Over/Under)
            "odds_home": df["odds_home"],
            "odds_draw": df["odds_draw"],
            "odds_away": df["odds_away"],
            "odds_over25": df.get("Avg>2.5", np.nan),
            "odds_under25": df.get("Avg<2.5", np.nan),

            # Expected Goals (falls vorhanden)
            "home_xg": df.get("HxG", np.nan),
            "away_xg": df.get("AxG", np.nan),

            # Match-Statistiken
            "home_shots": df.get("HS", np.nan),
            "away_shots": df.get("AS", np.nan),
            "home_shots_target": df.get("HST", np.nan),
            "away_shots_target": df.get("AST", np.nan),
            "home_corners": df.get("HC", np.nan),
            "away_corners": df.get("AC", np.nan),
            "home_fouls": df.get("HF", np.nan),
            "away_fouls": df.get("AF", np.nan),
            "home_yellow": df.get("HY", np.nan),
            "away_yellow": df.get("AY", np.nan),
            "home_red": df.get("HR", np.nan),
            "away_red": df.get("AR", np.nan),
        }
    )

    return clean_df


def download_file(url, target_path):
    """Lädt eine Datei von einer URL herunter."""
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
            out_file.write(response.read())
        return True
    except Exception as e:
        print(f"Fehler beim Download von {url}: {e}")
        return False


def process_all_raw_files(dataset_dir):
    """Lädt die D1.csv für die Saisons 2000 bis 2026 direkt herunter und verarbeitet sie."""
    os.makedirs(dataset_dir, exist_ok=True)

    # Generiere URLs dynamisch: Saison 2000/01 ist '0001', 2025/26 ist '2526'
    all_dfs = []

    for season_start in range(2000, 2027):
        yy_start = str(season_start)[-2:]
        yy_end = str(season_start + 1)[-2:]
        season_code = f"{yy_start}{yy_end}"

        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/D1.csv"
        temp_file = os.path.join(dataset_dir, f"D1_{season_start}.csv")

        if download_file(url, temp_file):
            print(f"Verarbeite Saison {season_start} ({season_code})...")
            df_season = parse_season_file(temp_file, season_start)
            if df_season is not None:
                all_dfs.append(df_season)

            # Aufräumen
            if os.path.exists(temp_file):
                os.remove(temp_file)

    if not all_dfs:
        print("Keine Saisons verarbeitet!")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by=["season", "matchday", "date"]).reset_index(drop=True)

    output_path = os.path.join(dataset_dir, "bundesliga_all_seasons.csv")
    combined_df.to_csv(output_path, index=False)
    print(f"\nErfolgreich {len(combined_df)} Spiele aus {len(all_dfs)} Saisons verarbeitet!")


if __name__ == "__main__":
    DATASET_DIR = r"D:\PycharmProjects\Bundesliga\Datasets"
    process_all_raw_files(DATASET_DIR)