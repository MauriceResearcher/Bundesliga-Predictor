"""

compute a form analysis of teams regarding the last x matches regarding the current matchday

"""

"""
for reference

match_dict = {
            "match_id": match.get("matchID"),
            "date": match.get("matchDateTimeUTC"),
            "season": match.get("leagueSeason"),
            "matchday": match.get("group", {}).get("groupOrderID"),
            "home_team_id": match.get("team1", {}).get("teamId"),
            "home_team": match.get("team1", {}).get("teamName"),
            "away_team_id": match.get("team2", {}).get("teamId"),
            "away_team": match.get("team2", {}).get("teamName"),
            "home_goals": goals_home,
            "away_goals": goals_away,
            "result": final_result,
        }

"""

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
    # Wenn wir Heimteam sind, sind gegnerische Tore 'away_goals'
    home_conceded = df[df["home_team_id"] == team_id]["away_goals"].sum()
    # Wenn wir Auswärtsteam sind, sind gegnerische Tore 'home_goals'
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

def compute_form(df, iterations, curr_matchday, team_id, match_type=Types.BOTH):
    past_matches = df[df["matchday"] < curr_matchday].copy()
    type_filtered_matches = filter_by_type(past_matches, team_id, match_type)
    type_filtered_matches = type_filtered_matches.sort_values(
        by="matchday", ascending=True
    )
    last_x_matches = type_filtered_matches.tail(iterations)

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
    """Durchläuft das gesamte DataFrame und fügt für jedes Match alle Form-Features

    (Home-Team & Away-Team jeweils für HOME, AWAY, BOTH) an.
    """
    featured_rows = []

    # Chronologisch nach Spieltag sortieren
    df = df.sort_values(by="matchday", ascending=True).reset_index(drop=True)

    for idx, row in df.iterrows():
        curr_matchday = row["matchday"]
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]

        # Kopie der ursprünglichen Zeile als Dictionary
        row_dict = row.to_dict()

        # 1. Form-Analysen für Heim-Team (Gesamt, Nur Heim, Nur Auswärts)
        home_both = compute_form(
            df, iterations, curr_matchday, home_id, Types.BOTH
        )
        home_home = compute_form(
            df, iterations, curr_matchday, home_id, Types.HOME
        )
        home_away = compute_form(
            df, iterations, curr_matchday, home_id, Types.AWAY
        )

        # 2. Form-Analysen für Auswärts-Team (Gesamt, Nur Heim, Nur Auswärts)
        away_both = compute_form(
            df, iterations, curr_matchday, away_id, Types.BOTH
        )
        away_home = compute_form(
            df, iterations, curr_matchday, away_id, Types.HOME
        )
        away_away = compute_form(
            df, iterations, curr_matchday, away_id, Types.AWAY
        )

        # 3. Anfügen der berechneten Features mit klaren Spaltennamen
        # --- Heimteam Features ---
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

        # --- Auswärtsteam Features ---
        row_dict[f"away_form_pts_both_{iterations}"] = away_both["pts"]
        row_dict[f"away_form_goals_both_{iterations}"] = away_both[
            "goals_scored"
        ]
        row_dict[f"away_form_conceded_both_{iterations}"] = away_both[
            "goals_conceded"
        ]

        row_dict[f"away_form_pts_away_{iterations}"] = away_away[
            "pts"
        ]  # Wichtig: Spezifische Auswärtsform des Gastes!
        row_dict[f"away_form_goals_away_{iterations}"] = away_away[
            "goals_scored"
        ]

        featured_rows.append(row_dict)

    return pd.DataFrame(featured_rows)


# Beispiel-Aufruf:
# path = r"D:\PycharmProjects\Bundesliga\Datasets\bundesliga_2024.csv"
# df = pd.read_csv(path)
# df_all_features = build_full_featured_dataframe(df, iterations=5)
# print(df_all_features.head())





