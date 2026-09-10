from enum import Enum
import pandas as pd


class Types(Enum):
    BOTH = 0
    HOME = 1
    AWAY = 2


def filter_by_type(df, team_id, match_type):
    """Filtert den DataFrame basierend auf dem gewählten Spiel-Typ (Heim, Auswärts oder Beide)."""
    if match_type == Types.HOME:
        return df[df["home_team_id"] == team_id]
    elif match_type == Types.AWAY:
        return df[df["away_team_id"] == team_id]
    else:  # Types.BOTH
        return df[(df["home_team_id"] == team_id) | (df["away_team_id"] == team_id)]


def compute_goals_scored(df, team_id):
    """Berechnet die erzielten Tore im bereits gefilterten DataFrame."""
    home_goals = df[df["home_team_id"] == team_id]["home_goals"].sum()
    away_goals = df[df["away_team_id"] == team_id]["away_goals"].sum()
    return int(home_goals + away_goals)


def compute_goals_conceded(df, team_id):
    """Berechnet die kassierten Gegentore im bereits gefilterten DataFrame."""
    home_conceded = df[df["home_team_id"] == team_id]["away_goals"].sum()
    away_conceded = df[df["away_team_id"] == team_id]["home_goals"].sum()
    return int(home_conceded + away_conceded)


def compute_points(df, team_id):
    """Berechnet die erzielten Punkte im bereits gefilterten DataFrame."""
    total_points = 0
    for _, row in df.iterrows():
        res = row["result"]
        is_home = row["home_team_id"] == team_id

        if is_home:
            if res == 0:
                total_points += 3  # Heimsieg
            elif res == 1:
                total_points += 1  # Remis
        else:
            if res == 2:
                total_points += 3  # Auswärtssieg
            elif res == 1:
                total_points += 1  # Remis

    return total_points


def compute_form(
    df, iterations, curr_season, curr_matchday, team_id, match_type=Types.BOTH
):
    # 1. Nur Vergangene Spiele finden (frühere Saisons ODER selbe Saison mit kleinerem Spieltag)
    past_matches = df[
        (df["season"] < curr_season)
        | ((df["season"] == curr_season) & (df["matchday"] < curr_matchday))
    ].copy()

    # 2. Nach dem gewünschten Type (HOME, AWAY, BOTH) filtern
    type_filtered_matches = filter_by_type(past_matches, team_id, match_type)

    # 3. Chronologisch nach Saison und Spieltag sortieren und die letzten N (iterations) Spiele nehmen
    type_filtered_matches = type_filtered_matches.sort_values(
        by=["season", "matchday"], ascending=True
    )
    last_x_matches = type_filtered_matches.tail(iterations)

    # 4. Metriken berechnen
    total_points = compute_points(last_x_matches, team_id)
    goals_scored = compute_goals_scored(last_x_matches, team_id)
    goals_conceded = compute_goals_conceded(last_x_matches, team_id)

    return {
        "pts": total_points,
        "goals_scored": goals_scored,
        "goals_conceded": goals_conceded,
        "goal_diff": goals_scored - goals_conceded,
    }



def build_full_featured_dataframe(df, iterations=5):
    """Durchläuft das gesamte mehrjährige DataFrame und fügt für jedes Match alle Form-Features an."""
    df_clean = df.copy()

    # Fallback, falls 'season' in der CSV fehlt
    if "season" not in df_clean.columns:
        df_clean["season"] = 2025

    # Datentypen für sicheres Aggregieren sicherstellen
    numeric_cols = ["home_goals", "away_goals", "matchday", "season", "result"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Chronologisch nach Saison UND Matchday sortieren
    df_clean = df_clean.sort_values(
        by=["season", "matchday"], ascending=True
    ).reset_index(drop=True)

    featured_rows = []

    for idx, row in df_clean.iterrows():
        curr_season = row["season"]
        curr_matchday = row["matchday"]
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]

        row_dict = row.to_dict()

        # 1. Form-Analysen für Heim-Team
        home_both = compute_form(
            df_clean,
            iterations,
            curr_season,
            curr_matchday,
            home_id,
            Types.BOTH,
        )
        home_home = compute_form(
            df_clean,
            iterations,
            curr_season,
            curr_matchday,
            home_id,
            Types.HOME,
        )
        home_away = compute_form(
            df_clean,
            iterations,
            curr_season,
            curr_matchday,
            home_id,
            Types.AWAY,
        )

        # 2. Form-Analysen für Auswärts-Team
        away_both = compute_form(
            df_clean,
            iterations,
            curr_season,
            curr_matchday,
            away_id,
            Types.BOTH,
        )
        away_home = compute_form(
            df_clean,
            iterations,
            curr_season,
            curr_matchday,
            away_id,
            Types.HOME,
        )
        away_away = compute_form(
            df_clean,
            iterations,
            curr_season,
            curr_matchday,
            away_id,
            Types.AWAY,
        )

        # 3. Features anfügen
        row_dict[f"home_form_pts_both_{iterations}"] = home_both["pts"]
        row_dict[f"home_form_goals_both_{iterations}"] = home_both[
            "goals_scored"
        ]
        row_dict[f"home_form_conceded_both_{iterations}"] = home_both[
            "goals_conceded"
        ]

        row_dict[f"home_form_pts_home_{iterations}"] = home_home["pts"]
        row_dict[f"home_form_goals_home_{iterations}"] = home_home[
            "goals_scored"
        ]

        row_dict[f"home_form_pts_away_{iterations}"] = home_away["pts"]

        row_dict[f"away_form_pts_both_{iterations}"] = away_both["pts"]
        row_dict[f"away_form_goals_both_{iterations}"] = away_both[
            "goals_scored"
        ]
        row_dict[f"away_form_conceded_both_{iterations}"] = away_both[
            "goals_conceded"
        ]

        row_dict[f"away_form_pts_away_{iterations}"] = away_away["pts"]
        row_dict[f"away_form_goals_away_{iterations}"] = away_away[
            "goals_scored"
        ]

        featured_rows.append(row_dict)

    return pd.DataFrame(featured_rows)


# Anwendung:
if __name__ == "__main__":

    SEASON = 2025
    path = rf"D:\PycharmProjects\Bundesliga\Datasets\bundesliga_{SEASON}_cleaned.csv"
    df = pd.read_csv(path)

    df_features = build_full_featured_dataframe(df, iterations=5)

    save_path = (
        rf"D:\PycharmProjects\Bundesliga\Datasets\bundesliga_{SEASON}_with_form.csv"
    )
    df_features.to_csv(save_path, index=False)
    print(f"Datensatz mit Form-Features erfolgreich gespeichert: {save_path}")





