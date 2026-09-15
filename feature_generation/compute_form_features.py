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

def compute_table(df, curr_season, curr_matchday):
    """Berechnet die offizielle Tabelle der aktuellen Saison VOR dem gegebenen Spieltag."""
    # 1. Alle vergangenen Spiele der AKTUELLEN Saison ermitteln
    past_matches = df[
        (df["season"] == curr_season) & (df["matchday"] < curr_matchday)
    ]

    # Dynamische Liste aller Teams der aktuellen Saison
    season_matches = df[df["season"] == curr_season]
    teams = list(
        set(season_matches["home_team_id"]).union(
            set(season_matches["away_team_id"])
        )
    )

    # Am 1. Spieltag haben alle 0 Punkte und 0 Tore
    if curr_matchday == 1 or past_matches.empty:
        # Standard-Sortierung zu Beginn: Alphabetisch nach Team-ID (Platz 1 bis N)
        sorted_teams = sorted(teams)
        return {team_id: idx + 1 for idx, team_id in enumerate(sorted_teams)}

    # 2. Stats pro Team berechnen
    table_stats = []
    for team_id in teams:
        pts = compute_points(past_matches, team_id)
        goals = compute_goals_scored(past_matches, team_id)
        conceded = compute_goals_conceded(past_matches, team_id)
        goal_diff = goals - conceded

        table_stats.append(
            {
                "team_id": team_id,
                "pts": pts,
                "goal_diff": goal_diff,
                "goals_scored": goals,
            }
        )

    # 3. Sortieren nach DFB-Regeln: Punkte DESC -> Tordifferenz DESC -> Tore DESC
    table_df = pd.DataFrame(table_stats)
    table_df = table_df.sort_values(
        by=["pts", "goal_diff", "goals_scored"], ascending=[False, False, False]
    ).reset_index(drop=True)

    # 4. Dictionary mit Team -> Platzierung (1-basiert) zurückgeben
    table_positions = {
        row["team_id"]: idx + 1 for idx, row in table_df.iterrows()
    }
    return table_positions


def get_table_pos(team_id, table_positions):
    """Liest die Tabellenposition eines Teams aus dem Tabellen-Dictionary aus."""
    return table_positions.get(team_id, 18)  # Fallback: Platz 18

# full vibecoded bis jetzt:
def compute_h2h_features(df, curr_season, curr_matchday, home_id, away_id, n_matches=5):
    """
    Berechnet die Head-to-Head (H2H) Bilanz der letzten n_matches Aufeinandertreffen
    zwischen home_id und away_id vor dem aktuellen Spieltag.
    """
    # 1. Nur vergangene Duelle ermitteln
    past_matches = df[
        (df["season"] < curr_season)
        | ((df["season"] == curr_season) & (df["matchday"] < curr_matchday))
    ]

    # 2. Duelle filtern, bei denen beide Teams gegeneinander gespielt haben
    h2h_matches = past_matches[
        ((past_matches["home_team_id"] == home_id) & (past_matches["away_team_id"] == away_id))
        | ((past_matches["home_team_id"] == away_id) & (past_matches["away_team_id"] == home_id))
    ].copy()

    # Nach Saison und Spieltag sortieren und die letzten n Duelle nehmen
    h2h_matches = h2h_matches.sort_values(by=["season", "matchday"], ascending=True).tail(n_matches)

    if h2h_matches.empty:
        return {
            f"h2h_home_wins_{n_matches}": 0,
            f"h2h_draws_{n_matches}": 0,
            f"h2h_away_wins_{n_matches}": 0,
            f"h2h_home_goals_{n_matches}": 0,
            f"h2h_away_goals_{n_matches}": 0,
        }

    home_wins = 0
    draws = 0
    away_wins = 0
    home_goals = 0
    away_goals = 0

    for _, row in h2h_matches.iterrows():
        res = row["result"]
        was_home_in_past = row["home_team_id"] == home_id

        # Tore addieren
        if was_home_in_past:
            home_goals += row["home_goals"]
            away_goals += row["away_goals"]
        else:
            home_goals += row["away_goals"]
            away_goals += row["home_goals"]

        # Ergebnis bewerten (0 = Heimsieg damals, 1 = Remis, 2 = Auswärtssieg damals)
        if res == 0:
            if was_home_in_past:
                home_wins += 1
            else:
                away_wins += 1
        elif res == 1:
            draws += 1
        elif res == 2:
            if was_home_in_past:
                away_wins += 1
            else:
                home_wins += 1

    return {
        f"h2h_home_wins_{n_matches}": home_wins,
        f"h2h_draws_{n_matches}": draws,
        f"h2h_away_wins_{n_matches}": away_wins,
        f"h2h_home_goals_{n_matches}": int(home_goals),
        f"h2h_away_goals_{n_matches}": int(away_goals),
    }

def get_team_stats_last_n(df, curr_season, curr_matchday, team_id, n_matches):
    """Hilfsfunktion: Holt die letzten n Spiele eines Teams (egal ob Heim oder Auswärts)

    und berechnet die Pro-Spiel-Durchschnitte aller erweiterten Match-Statistiken.
    """
    # 1. Alle vergangenen Spiele filtern
    past_matches = df[
        (df["season"] < curr_season)
        | ((df["season"] == curr_season) & (df["matchday"] < curr_matchday))
    ]

    # 2. Spiele finden, an denen das Team beteiligt war
    team_matches = past_matches[
        (past_matches["home_team_id"] == team_id)
        | (past_matches["away_team_id"] == team_id)
    ].copy()

    # Nach Saison & Spieltag sortieren und die letzten n Spiele nehmen
    team_matches = team_matches.sort_values(
        by=["season", "matchday"], ascending=True
    ).tail(n_matches)

    stats = {
        "xg": 0.0,
        "shots": 0.0,
        "shots_target": 0.0,
        "corners": 0.0,
        "fouls": 0.0,
        "yellow": 0.0,
        "red": 0.0,
    }

    if team_matches.empty:
        return stats

    match_count = len(team_matches)

    for _, row in team_matches.iterrows():
        is_home = row["home_team_id"] == team_id

        prefix = "home_" if is_home else "away_"

        stats["xg"] += row.get(f"{prefix}xg", 0) or 0
        stats["shots"] += row.get(f"{prefix}shots", 0) or 0
        stats["shots_target"] += row.get(f"{prefix}shots_target", 0) or 0
        stats["corners"] += row.get(f"{prefix}corners", 0) or 0
        stats["fouls"] += row.get(f"{prefix}fouls", 0) or 0
        stats["yellow"] += row.get(f"{prefix}yellow", 0) or 0
        stats["red"] += row.get(f"{prefix}red", 0) or 0

    # Pro-Spiel-Durchschnitt berechnen
    return {
        metric: round(val / match_count, 2) for metric, val in stats.items()
    }


def build_full_xg_features(
    df, curr_season, curr_matchday, home_id, away_id, n_matches=5
):
    """Berechnet die rollierenden Durchschnitte der letzten n Spiele für Heim- und Auswärtsteam."""
    home_stats = get_team_stats_last_n(
        df, curr_season, curr_matchday, home_id, n_matches
    )
    away_stats = get_team_stats_last_n(
        df, curr_season, curr_matchday, away_id, n_matches
    )

    feature_dict = {}

    # Heimteam Features anfügen
    for metric, val in home_stats.items():
        feature_dict[f"home_{metric}_last_{n_matches}"] = val

    # Auswärtsteam Features anfügen
    for metric, val in away_stats.items():
        feature_dict[f"away_{metric}_last_{n_matches}"] = val

    return feature_dict


def build_full_featured_dataframe(df, iterations=5):
    """Durchläuft das gesamte mehrjährige DataFrame und fügt Form-, Elo-, Tabellen- und allgemeine Match-Statistik-Features an."""
    df_clean = df.copy()

    if "season" not in df_clean.columns:
        df_clean["season"] = 2025

    numeric_cols = ["home_goals", "away_goals", "matchday", "season", "result"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_clean = df_clean.sort_values(
        by=["season", "matchday"], ascending=True
    ).reset_index(drop=True)

    featured_rows = []

    # Cache für die Tabelle pro Saison & Spieltag (spart enorme Rechenzeit)
    table_cache = {}

    for idx, row in df_clean.iterrows():
        curr_season = row["season"]
        curr_matchday = row["matchday"]
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]

        row_dict = row.to_dict()

        # --- 1. Tabellen-Features berechnen ---
        cache_key = (curr_season, curr_matchday)
        if cache_key not in table_cache:
            table_cache[cache_key] = compute_table(
                df_clean, curr_season, curr_matchday
            )

        current_table = table_cache[cache_key]

        home_pos = get_table_pos(home_id, current_table)
        away_pos = get_table_pos(away_id, current_table)

        row_dict["home_table_pos"] = home_pos
        row_dict["away_table_pos"] = away_pos
        row_dict["table_pos_diff"] = (
            home_pos - away_pos
        )  # Negativ = Heimteam steht weiter oben

        # --- 2. Form-Analysen für Heim- & Auswärtsteam ---
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

        # --- 3. Form Features anfügen ---
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

        # --- 4. H2H Feature für die letzten 5 Duelle berechnen ---
        h2h_dict = compute_h2h_features(
            df_clean,
            curr_season,
            curr_matchday,
            home_id,
            away_id,
            n_matches=5,
        )
        row_dict.update(h2h_dict)

        # --- 5. Allgemeines xG & Match-Statistiken-Feature (Letzte N Spiele) ---
        match_stats_dict = build_full_xg_features(
            df_clean,
            curr_season,
            curr_matchday,
            home_id,
            away_id,
            n_matches=iterations,
        )
        row_dict.update(match_stats_dict)

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





