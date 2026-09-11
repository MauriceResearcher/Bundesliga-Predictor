import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

FEATURE_COLS = [
    # Elo Features
    "home_elo",
    "away_elo",
    "elo_diff",
    # Wettquoten (falls vorhanden, sonst mit 0 befüllt)
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
    # Table positions
    "home_table_pos",
    "away_table_pos",
    "table_pos_diff",
    # Head to head
    "h2h_home_wins_5",
    "h2h_draws_5",
    "h2h_away_wins_5",
    "h2h_home_goals_5",
    "h2h_away_goals_5",
]


# --- 1. PyTorch Neural Network Regressor (xG Output: [home_goals, away_goals]) ---
class GoalsNNModule(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2),  # Output: xG Heim, xG Auswärts
        )

    def forward(self, x):
        return self.net(x)


class PyTorchGoalsRegressor:
    """Wrapper, damit sich PyTorch wie ein Sklearn-Modell verhält (.fit, .predict)."""

    def __init__(self, epochs=120, lr=0.003, batch_size=32):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model = GoalsNNModule(input_dim=X.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X_tensor).numpy()
        return np.clip(preds, 0, None)


# --- 2. Metrik-Berechnung für 1X2 Tendenz & xG MAE ---
def evaluate_predictions(y_true_goals, y_pred_goals):
    mae_home = mean_absolute_error(y_true_goals[:, 0], y_pred_goals[:, 0])
    mae_away = mean_absolute_error(y_true_goals[:, 1], y_pred_goals[:, 1])

    def goals_to_outcome(home, away):
        if home > away:
            return 0  # Heimsieg
        elif home == away:
            return 1  # Unentschieden
        else:
            return 2  # Auswärtssieg

    true_outcomes = [goals_to_outcome(g[0], g[1]) for g in y_true_goals]
    pred_outcomes = [
        goals_to_outcome(np.round(g[0]), np.round(g[1])) for g in y_pred_goals
    ]

    acc = accuracy_score(true_outcomes, pred_outcomes)
    return (mae_home + mae_away) / 2.0, acc


# --- 3. Hauptfunktion: Trainiert NN, RF, XGBoost & Ensemble ---
def train_and_predict_multi_models(train_df, predict_df):
    train_df = train_df.copy()
    predict_df = predict_df.copy()

    # Verfügbare Spalten absichern
    available_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    for c in available_cols:
        train_df[c] = train_df[c].fillna(0)
        predict_df[c] = predict_df[c].fillna(0)

    X = train_df[available_cols].values
    y = train_df[["home_goals", "away_goals"]].values.astype(float)

    X_pred = predict_df[available_cols].values

    # Chronologischer Validation-Split (letzte 20% der Historie zum Evaluieren)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    models = {
        "Neural Network": PyTorchGoalsRegressor(epochs=120, lr=0.003),
        "Random Forest": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=42
            )
        ),
        "XGBoost": MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.03,
                random_state=42,
            )
        ),
    }

    model_performances = {}
    next_match_predictions = {}

    # A. Einzelmodelle trainieren & evaluieren
    for name, model in models.items():
        model.fit(X_tr, y_tr)

        val_preds = model.predict(X_val)
        tr_preds = model.predict(X_tr)

        mae_tr, acc_tr = evaluate_predictions(y_tr, tr_preds)
        mae_val, acc_val = evaluate_predictions(y_val, val_preds)

        model_performances[name] = {
            "train_acc": acc_tr,
            "val_acc": acc_val,
            "train_mae": mae_tr,
            "val_mae": mae_val,
        }

        # Komplett auf allen absolvierten Spielen fitten für den echten Prediction-run
        model.fit(X, y)
        next_match_predictions[name] = model.predict(X_pred)

    # B. Ensemble bilden (Durchschnitt der xG-Tore aller 3 Modelle)
    ensemble_pred = (
        next_match_predictions["Neural Network"]
        + next_match_predictions["Random Forest"]
        + next_match_predictions["XGBoost"]
    ) / 3.0
    next_match_predictions["Ensemble"] = ensemble_pred

    # Ensemble Performance als Schnitt der Einzel-Val-Accuracies
    ens_tr_acc = np.mean([p["train_acc"] for p in model_performances.values()])
    ens_val_acc = np.mean([p["val_acc"] for p in model_performances.values()])
    ens_tr_mae = np.mean([p["train_mae"] for p in model_performances.values()])
    ens_val_mae = np.mean([p["val_mae"] for p in model_performances.values()])

    model_performances["Ensemble"] = {
        "train_acc": ens_tr_acc,
        "val_acc": ens_val_acc,
        "train_mae": ens_tr_mae,
        "val_mae": ens_val_mae,
    }

    # C. Ergebnis-DataFrame aufbauen
    summary_rows = []

    for idx, (_, row) in enumerate(predict_df.iterrows()):
        item = {
            "Saison": row["season"],
            "Spieltag": row["matchday"],
            "Heimteam": row["home_team"],
            "Auswärtsteam": row["away_team"],
        }

        for model_name, preds in next_match_predictions.items():
            hg_xg, ag_xg = preds[idx][0], preds[idx][1]
            hg_round, ag_round = int(np.round(hg_xg)), int(np.round(ag_xg))

            item[f"{model_name}_xG"] = f"{hg_xg:.2f} : {ag_xg:.2f}"
            item[f"{model_name}_Tipp"] = f"{hg_round} : {ag_round}"

        summary_rows.append(item)

    return pd.DataFrame(summary_rows), model_performances, models