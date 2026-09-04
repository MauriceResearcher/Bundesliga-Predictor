import os
import pandas as pd
from generate_bl_dataset import create_dataframe

# Verwende r"..." für Windows-Pfade mit Backslashes
path = r"D:\PycharmProjects\Bundesliga\Datasets"

# Zielordner erstellen, falls er noch nicht existiert
os.makedirs(path, exist_ok=True)


def download_and_save_season(season):
    season_dfs = []

    print(f"Lade Daten für Saison {season} herunter...")

    # Eine Bundesliga-Saison hat 34 Spieltage
    for matchday in range(1, 35):
        df_matchday = create_dataframe(season, matchday)

        # Falls Daten für den Spieltag zurückgeliefert wurden, an die Liste anhängen
        if not df_matchday.empty:
            season_dfs.append(df_matchday)

    # Alle Spieltage zu einem großen DataFrame zusammenfügen
    if season_dfs:
        full_season_df = pd.concat(season_dfs, ignore_index=True)

        # Pfad für die CSV-Datei bauen
        file_path = os.path.join(path, f"bundesliga_{season}.csv")

        # Als CSV speichern (index=False verhindert eine extra Spalte für den Index)
        full_season_df.to_csv(file_path, index=False)
        print(
            f"Saison {season} erfolgreich gespeichert unter: {file_path} ({len(full_season_df)} Spiele)"
        )
    else:
        print(f"Keine Daten für Saison {season} gefunden.")


# Saisons 2024 und 2025 herunterladen und speichern
for season in [2024, 2025]:
    download_and_save_season(season)