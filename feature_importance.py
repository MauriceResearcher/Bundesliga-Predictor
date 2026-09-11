import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from train_models import FEATURE_COLS, train_and_predict_multi_models


def plot_feature_importance(model, feature_names, model_name="Random Forest"):
    """Extrahiert und visualisiert die Feature Importance von MultiOutput-Tree-Modellen."""
    # Bei MultiOutputRegressor nehmen wir den Durchschnitt der beiden Schätzer (Home & Away Goals)
    importances = np.mean(
        [estimator.feature_importances_ for estimator in model.estimators_],
        axis=0,
    )

    feature_imp = pd.Series(importances, index=feature_names).sort_values(
        ascending=True
    )

    plt.figure(figsize=(10, 6))
    feature_imp.plot(kind="barh", color="skyblue")
    plt.title(f"Feature Importance ({model_name})")
    plt.xlabel("Relative Wichtigkeit")
    plt.tight_layout()
    plt.savefig(
        f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    )
    plt.close()

    print(f"\nTop 5 Features ({model_name}):")
    print(feature_imp.tail(5)[::-1])


# --- Inspektion ausführen ---
if __name__ == "__main__":
    # 1. Daten laden (dein verarbeitetes Gesamt-DF)
    df = pd.read_csv("Datasets/bundesliga_processed.csv")

    train_df = df[df["result"].notna()]
    predict_df = df[df["result"].isna()]  # Oder bestimmter Spieltag

    # 2. Vorhersage & Modell-Training durchführen
    predictions, performances, trained_models = train_and_predict_multi_models(
        train_df, predict_df
    )

    # 2. Feature Importance für Random Forest visualisieren
    plot_feature_importance(
        trained_models["Random Forest"],
        FEATURE_COLS,
        model_name="Random Forest",
    )

    # 3. Feature Importance für XGBoost visualisieren
    plot_feature_importance(
        trained_models["XGBoost"], FEATURE_COLS, model_name="XGBoost"
    )