import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

FEATURE_COLS = [
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


class BundesligaPredictor(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),  # 0: Heim, 1: Remis, 2: Auswärts
        )

    def forward(self, x):
        return self.net(x)


def train_model(train_df, epochs=150, batch_size=32, lr=0.001):
    """Trainiert das Modell auf absolvierten Spielen."""
    train_df = train_df.copy()
    train_df[FEATURE_COLS] = train_df[FEATURE_COLS].fillna(0)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["result"].values.astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    model = BundesligaPredictor(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    return model, scaler


def predict_fixtures(train_df, predict_df):
    """Macht Vorhersagen für predict_df auf Basis des auf train_df trainierten Modells."""
    if predict_df.empty:
        print("Keine Spiele zum Vorhersagen übergeben!")
        return pd.DataFrame()

    print(
        f"Trainiere Modell auf {len(train_df)} absolvierten Partien..."
    )
    model, scaler = train_model(train_df)

    predict_df = predict_df.copy()
    predict_df[FEATURE_COLS] = predict_df[FEATURE_COLS].fillna(0)

    X_predict = predict_df[FEATURE_COLS].values
    X_predict_scaled = scaler.transform(X_predict)
    X_predict_t = torch.tensor(X_predict_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(X_predict_t)
        probs = torch.softmax(logits, dim=1).numpy()

    results = []
    labels = ["Heimsieg", "Unentschieden", "Auswärtssieg"]  # 1 = Heim, X = Remis, 2 = Auswärts

    for idx, (_, row) in enumerate(predict_df.iterrows()):
        p_home, p_draw, p_away = probs[idx]
        pred_index = np.argmax(probs[idx])
        pred_label = labels[pred_index]

        results.append(
            {
                "Saison": row["season"],
                "Spieltag": row["matchday"],
                "Heimteam": row["home_team"],
                "Auswärtsteam": row["away_team"],
                "Vorhersage": pred_label,
                "Wahrscheinlichkeit Heimsieg": round(float(p_home), 4),
                "Wahrscheinlichkeit Unentschieden": round(float(p_draw), 4),
                "Wahrscheinlichkeit Auswärtssieg": round(float(p_away), 4),
            }
        )

    return pd.DataFrame(results)


def train_and_predict(train_df, predict_df):
    return predict_fixtures(train_df, predict_df)