import os
import pandas as pd
from feature_generation.compute_elo import compute_elo_rating
from feature_generation.compute_form_features import (
    build_full_featured_dataframe,
)


def main():
    # 1. Konfiguration & Pfade
    DATASET_DIR = r"D:\PycharmProjects\Bundesliga\Datasets"
    ALL_SEASONS_PATH = os.path.join(DATASET_DIR, "bundesliga_all_seasons.csv")
    OUTPUT_PATH = os.path.join(DATASET_DIR, "bundesliga_processed.csv")

    # 2. Kombinierte Rohdaten aus prepare_football_data.py laden
    print("=== 1. Lade zusammengefügte Rohdaten ===")
    if not os.path.exists(ALL_SEASONS_PATH):
        raise FileNotFoundError(
            f"Datei '{ALL_SEASONS_PATH}' nicht gefunden! "
            "Bitte zuerst 'prepare_football_data.py' ausführen."
        )

    df_all = pd.read_csv(ALL_SEASONS_PATH)

    # Sortierung zur Sicherheit nach Saison, Spieltag und Datum
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.sort_values(
        by=["season", "matchday", "date"]
    ).reset_index(drop=True)

    print(
        f"Gesamtanzahl Spiele geladen: {len(df_all)} über Saisons {df_all['season'].min()} bis {df_all['season'].max()}"
    )

    # 3. Form-Features berechnen (rollierend über alle Saisons)
    print("\n=== 2. Berechne Form-Features ===")
    df_with_form = build_full_featured_dataframe(df_all, iterations=5)

    # 4. Elo-Ratings berechnen (inkl. Reset am Saisonübergang)
    print("=== 3. Berechne Elo-Ratings ===")
    final_df = compute_elo_rating(
        df=df_with_form,
        initial_elo=1500,
        k_factor=20,
        mean_reversion=0.30,  # 30% Reset am Saisonübergang
        base_ha=60,  # Basis-Heimvorteil
    )

    # 5. Kontrolle
    print("\n=== 4. Fertigstellung & Kontrolle ===")
    print(f"Verarbeitete Zeilen insgesamt: {len(final_df)}")
    print("Verfügbare Spalten im verarbeiteten Datensatz:")
    print(final_df.columns.tolist())

    # 6. Finale verarbeitete Datei speichern
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nErfolgreich gespeichert unter: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()