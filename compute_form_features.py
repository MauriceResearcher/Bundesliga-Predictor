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
    # 1. Nur vergangene Spiele vor dem aktuellen Spieltag
    past_matches = df[df["matchday"] < curr_matchday].copy()

    # 2. Nach dem gewünschten Type (HOME, AWAY, BOTH) filtern
    type_filtered_matches = filter_by_type(past_matches, team_id, match_type)

    # 3. Chronologisch sortieren und die letzten N (iterations) Spiele nehmen
    type_filtered_matches = type_filtered_matches.sort_values(
        by="matchday", ascending=True
    )
    last_x_matches = type_filtered_matches.tail(iterations)

    # 4. Metriken über die separaten Hilfsfunktionen berechnen
    total_points = compute_points(last_x_matches, team_id)
    goals_scored = compute_goals_scored(last_x_matches, team_id)
    goals_conceded = compute_goals_conceded(last_x_matches, team_id)

    # 5. Dictionary aufbauen
    return_dict = {
        "team_id": team_id,
        "match_type": match_type.name,
        "matches_evaluated": len(last_x_matches),
        "total_form_pts": total_points,
        "goals_scored": goals_scored,
        "goals_conceded": goals_conceded,
        "goal_difference": goals_scored - goals_conceded,
    }

    return return_dict





