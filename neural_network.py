import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# 1. Daten laden und vorbereiten
# ------------------------------------------------------------------
df = pd.read_csv(
    r"D:\PycharmProjects\Bundesliga\Datasets\bundesliga_processed.csv"
)

# Feature-Spalten auswählen
feature_cols = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form_pts_both_5",
    "home_form_pts_home_5",
    "home_form_goals_both_5",
    "home_form_conceded_both_5",
    "away_form_pts_both_5",
    "away_form_pts_away_5",
    "away_form_goals_both_5",
    "away_form_conceded_both_5",
]

X = df[feature_cols].values
y = df["result"].values

# Train-Test-Split (Aufteilung chronologisch oder randomisiert)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Features skalieren (StandardScaler ist essenziell für Neuronale Netze)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PyTorch Tensor-Konvertierung
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)


# ------------------------------------------------------------------
# 2. Architektur des Neuronalen Netzes definieren
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


model = BundesligaPredictor(input_dim=X_train.shape[1])

# Loss-Funktion und Optimizer festlegen
criterion = nn.CrossEntropyLoss()
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
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

# ------------------------------------------------------------------
# 4. Evaluierung & Vorhersage
# ------------------------------------------------------------------
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    # Probabilities berechnen mittels Softmax
    test_probs = torch.softmax(test_logits, dim=1).numpy()
    preds = np.argmax(test_probs, axis=1)

print("\n--- Modellauswertung auf Testdaten ---")
print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
print("\nClassification Report:")
print(
    classification_report(
        y_test, preds, target_names=["Heimsieg", "Unentschieden", "Auswärtssieg"]
    )
)