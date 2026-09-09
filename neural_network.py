import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------------------------------------------------
# 0. Hilfsfunktionen
# ------------------------------------------------------------------
def get_chronological_split(df, test_season=2025):
    """Spaltet die Daten chronologisch auf.

    Alle gespielten Saisons vor `test_season` bilden das Trainingsset.
    Gespielte Partien ab `test_season` bilden das Testset.
    """
    df = df.sort_values(by=["season", "matchday"]).reset_index(drop=True)

    # Nur absolvierte Spiele für Training & Evaluation nutzen
    df_played = df[df["result"].notnull()].copy()
    df_played["result"] = df_played["result"].astype(int)

    train_mask = df_played["season"] < test_season
    test_mask = df_played["season"] >= test_season

    train_df = df_played[train_mask].copy()
    test_df = df_played[test_mask].copy()

    print(f"Training auf {len(train_df)} Spielen (Saisons < {test_season})...")
    print(
        f"Testen auf {len(test_df)} vergangenen Spielen (Saisons >= {test_season})..."
    )

    return train_df, test_df


# ------------------------------------------------------------------
# 1. Daten laden und vorbereiten
# ------------------------------------------------------------------
# Pfad auf 'bundesliga_all_seasons.csv' angepasst
df = pd.read_csv(
    r"D:\PycharmProjects\Bundesliga\Datasets\bundesliga_processed.csv"
)

# NEU: Erweiterte Feature-Liste inkl. Quoten & vollständiger Form-Metriken
feature_cols = [
    # Elo Features
    "home_elo",
    "away_elo",
    "elo_diff",
    # Wettquoten
    "odds_home",
    "odds_draw",
    "odds_away",
    # Heimteam Form-Features
    "home_form_pts_both_5",
    "home_form_pts_home_5",
    "home_form_goals_both_5",
    "home_form_conceded_both_5",
    "home_form_goals_home_5",
    # Auswärtsteam Form-Features
    "away_form_pts_both_5",
    "away_form_pts_away_5",
    "away_form_goals_both_5",
    "away_form_conceded_both_5",
    "away_form_goals_away_5",
]

# Chronologischer Split (Training: < 2025, Test: >= 2025)
train_df, test_df = get_chronological_split(df, test_season=2025)

# Missing Values in Quoten oder Form-Features absichern (falls vorhanden)
train_df[feature_cols] = train_df[feature_cols].fillna(0)
test_df[feature_cols] = test_df[feature_cols].fillna(0)

# Features & Targets extrahieren
X_train = train_df[feature_cols].values
y_train = train_df["result"].values

X_test = test_df[feature_cols].values
y_test = test_df["result"].values

# Features skalieren (Scaler wird NUR auf Trainingsdaten gefittet!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PyTorch Tensor-Konvertierung
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)


# ------------------------------------------------------------------
# 2. Architektur des Neuronalen Netzes
# ------------------------------------------------------------------
class BundesligaPredictor(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Verhindert Overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),  # 3 Logits (0: Heim, 1: Remis, 2: Auswärts)
        )

    def forward(self, x):
        return self.net(x)


# Dynamische Eingabedimension basierend auf den neuen Feature-Spalten
model = BundesligaPredictor(input_dim=X_train.shape[1])

class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

print("\nAutomatisch berechnete Klassengewichte:")
print(f"Heimsieg: {class_weights[0]:.2f}")
print(f"Unentschieden: {class_weights[1]:.2f}")
print(f"Auswärtssieg: {class_weights[2]:.2f}\n")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


# ------------------------------------------------------------------
# 3. Modell trainieren
# ------------------------------------------------------------------
epochs = 150
batch_size = 32

dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)

model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 30 == 0:
        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}"
        )


# ------------------------------------------------------------------
# 4. Evaluierung auf Testdaten
# ------------------------------------------------------------------
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_probs = torch.softmax(test_logits, dim=1).numpy()
    preds = np.argmax(test_probs, axis=1)

print("\n--- Modellauswertung auf Testdaten ---")
print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        preds,
        target_names=["Heimsieg", "Unentschieden", "Auswärtssieg"],
    )
)


# ------------------------------------------------------------------
# 5. Vorhersage für kommende / ungespielte Matches
# ------------------------------------------------------------------
"""unplayed_df = df[df["result"].isnull()].copy()

if len(unplayed_df) > 0:
    print(f"\n--- Vorhersagen für {len(unplayed_df)} kommende Duelle ---")
    unplayed_df[feature_cols] = unplayed_df[feature_cols].fillna(0)
    X_unplayed = unplayed_df[feature_cols].values
    X_unplayed_scaled = scaler.transform(X_unplayed)
    X_unplayed_t = torch.tensor(X_unplayed_scaled, dtype=torch.float32)

    with torch.no_grad():
        unplayed_logits = model(X_unplayed_t)
        unplayed_probs = torch.softmax(unplayed_logits, dim=1).numpy()

    labels = ["Heimsieg", "Unentschieden", "Auswärtssieg"]
    for idx, (_, row) in enumerate(unplayed_df.iterrows()):
        p_home, p_draw, p_away = unplayed_probs[idx]
        pred_label = labels[np.argmax(unplayed_probs[idx])]
        print(
            f"Saison {row['season']} Spieltag {row['matchday']}: "
            f"{row['home_team']} vs. {row['away_team']} -> Tipp: {pred_label} "
            f"(Heim: {p_home:.1%}, Remis: {p_draw:.1%}, Auswärts: {p_away:.1%})"
        )

"""