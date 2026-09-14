import os
import pandas as pd

# Pfad zu deinem Datensatz
DATASET_PATH = (
    r"D:\PycharmProjects\Bundesliga\Datasets\bundesliga_full_dataset_2024.csv"
)


def inspect_csv(file_path):
    if not os.path.exists(file_path):
        print(f"Datei nicht gefunden: {file_path}")
        return

    print("=" * 70)
    print(f" INSPEKTION VON: {os.path.basename(file_path)}")
    print("=" * 70)

    # 1. Datensatz laden
    df = pd.read_csv(file_path)

    # 2. Basis-Dimensionen
    print(
        f"\n 1. DIMENSIONEN:\n  - Zeilen (Spiele): {df.shape[0]}\n  - Spalten (Features): {df.shape[1]}"
    )

    # 3. Spaltenübersicht & Datentypen
    print("\n 2. SPALTEN UND DATENTYPEN:")
    print(df.dtypes.to_string())

    # 4. Überprüfung auf Fehlwerte (Missing Values / NaNs)
    print("\n 3. FEHLWERTE (NaNs) PRO SPALTE:")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        print(missing_cols.to_string())
    else:
        print("  Keine Fehlwerte gefunden! Datensatz ist vollständig.")

    # 5. Verteilung der Spielergebnisse (Machine Learning Targets)
    if "result" in df.columns:
        print("\n 4. VERTEILUNG DES ZIEL-LABELS (result):")
        print("  0: Heimsieg | 1: Unentschieden | 2: Auswärtssieg | None: Offen")
        results_count = df["result"].value_counts(dropna=False).sort_index()
        print(results_count.to_string())

    # 6. Relegation & Spieltage analysieren
    if "round" in df.columns:
        print("\n 5. SPIELTAG-ÜBERSICHT (round):")
        unique_rounds = df["round"].unique()
        print(f"  Gefundene Runden: {len(unique_rounds)}")
        print(f"  Runden-Beispiel: {unique_rounds[:3]} ... {unique_rounds[-3:]}")

    # 7. Erste 3 Zeilen als Vorschau
    print("\n 6. VORSCHAU DER ERSTEN 3 ZEILEN:")
    print(df.head(3).to_string())

    print("\n" + "=" * 70)


if __name__ == "__main__":
    inspect_csv(DATASET_PATH)