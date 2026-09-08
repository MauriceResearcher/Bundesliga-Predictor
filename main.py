import os
import pandas as pd
from compute_elo import compute_elo_rating
from compute_form_features import Types, build_full_featured_dataframe


def main():
    # 1. Dateipfade definieren

    path = r"D:\PycharmProjects\Bundesliga\Datasets\bundesliga_all_seasons.csv"

    output_path = (
        r"D:\PycharmProjects\Bundesliga\Datasets\bundesliga_processed.csv"
    )

     # 2. Basis-Daten laden
    print("Lade Rohdaten...")
    df = pd.read_csv(path)

    df = df.sort_values(["season", "matchday"]).reset_index(drop=True)

    # 3. Form-Features berechnen
    # Erstellt die Spalten: home_form_pts_home_5, away_form_pts_away_5, etc.
    print("Berechne Form-Features...")
    df_with_form = build_full_featured_dataframe(df, iterations=5)

    # 4. Elo-Ratings berechnen
    # Greift auf 'home_form_pts_home_5' zu, um den dynamischen Heimvorteil zu ermitteln
    print("Berechne Elo-Ratings...")
    final_df = compute_elo_rating(
        df=df_with_form,
        initial_elo=1500,
        k_factor=20,
        mean_reversion=0.30,  # 30% Reset am Saisonende
        base_ha=60,  # Basis-Heimvorteil
    )

    # 5. Kontrolle der neuen Spalten
    print("\nVerfügbare Spalten im fertigen Datensatz:")
    print(final_df.columns.tolist())

    # 6. Verarbeiteten Datensatz speichern
    final_df.to_csv(output_path, index=False)
    print(f"\nErfolgreich gespeichert unter: {output_path}")


if __name__ == "__main__":
    main()