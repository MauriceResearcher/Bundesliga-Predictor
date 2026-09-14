# ⚽ Bundesliga AI Predictor & Pipeline

Eine automatisierte Machine-Learning-Pipeline zur Vorhersage von Bundesliga-Spielergebnissen (xG-Regression und 1-X-2 Tendenzen) unter Verwendung eines Multi-Model-Ensembles (Neural Network, Random Forest, XGBoost).

Die Vorhersagen sowie detaillierte Performance-Metriken (MAE & Validierungs-Accuracy) werden automatisch aufbereitet und per strukturell formatierter HTML-E-Mail versendet.

---

## 🚀 Features & Funktionsweise

* **Automatische Datenbeschaffung:** Lädt historische Spieldaten von Football-Data.co.uk und ruft anstehende Paarungen direkt über die OpenLigaDB API ab.
* **Feature Engineering:** Berechnet dynamische Elo-Ratings (mit Mean-Reversion & Heimvorteil-Anpassung), Head-to-Head (H2H) Bilanzen, Form-Features (Pts/Tore über die letzten 5 Spiele) sowie Tabellenpositionen.
* **Multi-Model Ensemble:**
  * Custom PyTorch Neural Network (xG Regression)
  * Random Forest Regressor (MultiOutput)
  * XGBoost Regressor (MultiOutput)
  * Weighted Averaging Ensemble zur Kombination aller Einzelmodelle
* **HTML E-Mail Reporting:** Automatische Generierung und Auslieferung von Vorhersage-Tabellen und Modell-Confidence-Metriken.

---

## 🛠️ Tech Stack

* **Sprache & Package Manager:** Python 3.10+, `uv`
* **Machine Learning & Data Science:** PyTorch, Scikit-Learn, XGBoost, Pandas, NumPy, Matplotlib
* **Data Sources & APIs:** OpenLigaDB API, Football-Data.co.uk CSV Archives
* **Mail Delivery:** Python `ezgmail` 

---

## 📂 Projektstruktur

```text
.
├── feature_generation/       # Module zur Berechnung von Elo-Ratings & Form-Features
├── generate_datasets/        # ETL-Skripte zur Aggregation aller historischen Saisons
├── train_models.py           # Multi-Model-Training & Ensemble-Logik
├── send_email.py             # HTML-E-Mail-Generierung und Versendungs-Pipeline
├── automated_pipeline.py     # Hauptprogramm (End-to-End Orchestrierung)
├── feature_importance.py     # Inspektion & Visualisierung der Baum-Feature-Importances
└── Figures/                  # Generierte Plots (z. B. Feature Importance)

### 1. Repository klonen & Umgebung mit uv einrichten
git clone [https://github.com/DEIN_USERNAME/bundesliga-predictor.git](https://github.com/DEIN_USERNAME/bundesliga-predictor.git)
cd bundesliga-predictor
uv sync

### 2. Umgebungsvariablen konfigurieren
Erstelle eine .env-Datei im Hauptverzeichnis und trage deine Ziel-E-Mail-Adresse ein 
(dort werden die Ergebnisse hingeschickt):
GMAIL=deine_ziel_email@example.com

### 3. Pipeline ausführen
uv run automated_pipeline.py

---
````

## 📌 Roadmap & Geplante Updates

- [ ] **Performance & Tuning (Accuracy steigern):**
  - Integration von erweiterten Match-Statistiken (Schüsse aufs Tor, Ecken, xG-Historie) aus den D1.csv-Archiven.
  - Integration weiterer Statistiken z.b. Kaderwerte/ Verletze Spieler etc.
  - Implementierung von Post-Processing-Regeln / Probability Thresholding zur Schärfung von Unentschieden-Tendenzen.
- [ ] **Vollautomatisches Cloud-Hosting:**
  - Einrichtung einer GitHub Action (Cronjob), die die Pipeline vor jedem Spieltag völlig autonom prüft und ausführt.
- [ ] **Automatischer Kicktipp-Bot:**
  - Implementierung von Browser-Automation (via Playwright / Selenium) zum automatischen Eintragen der berechneten Ensemble-Tipps in die Kicktipp-Runde.

---

## 🏆 Kicktipp-Runde

Die von der KI generierten Tipps werden aktuell in folgender Tipprunde eingetragen:  
🔗 [Hier geht es zur Kicktipp-Runde](https://www.kicktipp.de/ki-comparison)