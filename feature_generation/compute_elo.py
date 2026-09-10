"""

compute elo ratings for the teams starting at 1500
,each season the elo is reset by x %
,new teams will receive the elo of the current 3rd last team

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
        
Index(['match_id', 'date', 'season', 'matchday', 'home_team_id', 'home_team',
       'away_team_id', 'away_team', 'home_goals', 'away_goals', 'result',
       'home_form_pts_both_5', 'home_form_goals_both_5',
       'home_form_conceded_both_5', 'home_form_pts_home_5',
       'home_form_goals_home_5', 'home_form_pts_away_5',
       'away_form_pts_both_5', 'away_form_goals_both_5',
       'away_form_conceded_both_5', 'away_form_pts_away_5',
       'away_form_goals_away_5'],
      dtype='str')

"""


import numpy as np
import pandas as pd


def compute_elo_rating(
    df, initial_elo=1500, k_factor=20, mean_reversion=0.3, base_ha=60
):
    """Berechnet Elo-Ratings mit dynamischem Heimvorteil, Aufsteiger-Sonderregelung

    und saisonalem Mean-Reversion Reset basierend auf dem neuen CSV-Format.
    """
    df_clean = df.copy()

    # Datentypen für sicheres Sortieren und Rechnen sicherstellen
    df_clean["season"] = pd.to_numeric(df_clean["season"], errors="coerce")
    df_clean["matchday"] = pd.to_numeric(df_clean["matchday"], errors="coerce")

    # 1. Nach Saison und Spieltag chronologisch sortieren
    df_clean = df_clean.sort_values(by=["season", "matchday"]).reset_index(
        drop=True
    )

    current_elo = {}

    home_elo_before = []
    away_elo_before = []
    elo_diff_list = []

    curr_season = None
    promoted_team_elo = initial_elo

    for idx, row in df_clean.iterrows():
        season = row["season"]
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]
        res = row["result"]  # 0: Heim, 1: Remis, 2: Auswärts

        # Form-Feature für dynamischen Heimvorteil (0 bis 15 Punkte)
        # Baut auf den zuvor erstellten Form-Features auf
        home_home_pts = row.get("home_form_pts_home_5", 7.5)
        if pd.isna(home_home_pts):
            home_home_pts = 7.5

        # -------------------------------------------------------------
        # 2. NEUE SAISON DETEKTIEREN (Reset & Aufsteiger-Initialisierung)
        # -------------------------------------------------------------
        if curr_season is not None and season != curr_season:
            # a) Mean Reversion auf bestehende Teams anwenden
            for team in current_elo:
                current_elo[team] = current_elo[team] * (
                    1 - mean_reversion
                ) + (initial_elo * mean_reversion)

            # b) Drittletzten Elo-Wert für neue Teams ermitteln (Platz 16 bei 18 Teams)
            sorted_teams_by_elo = sorted(
                current_elo.items(), key=lambda item: item[1]
            )

            if len(sorted_teams_by_elo) >= 3:
                # 3. von unten (Index 2 in aufsteigender Liste)
                promoted_team_elo = sorted_teams_by_elo[2][1]
            else:
                promoted_team_elo = initial_elo

            # c) Nur Teams behalten, die in der KOMMENDEN Saison mitspielen
            upcoming_season_matches = df_clean[df_clean["season"] == season]
            active_teams = set(upcoming_season_matches["home_team_id"]).union(
                set(upcoming_season_matches["away_team_id"])
            )

            # Dictionary bereinigen (Absteiger fliegen raus)
            current_elo = {
                team_id: elo
                for team_id, elo in current_elo.items()
                if team_id in active_teams
            }

            curr_season = season
        elif curr_season is None:
            curr_season = season

        # Falls ein Team zum ersten Mal auftaucht (z. B. Aufsteiger)
        if home_id not in current_elo:
            current_elo[home_id] = promoted_team_elo
        if away_id not in current_elo:
            current_elo[away_id] = promoted_team_elo

        # -------------------------------------------------------------
        # 3. ELO VOR DEM SPIEL SPEICHERN
        # -------------------------------------------------------------
        r_home = current_elo[home_id]
        r_away = current_elo[away_id]

        home_elo_before.append(r_home)
        away_elo_before.append(r_away)
        elo_diff_list.append(r_home - r_away)

        # -------------------------------------------------------------
        # 4. DYNAMISCHER HEIMVORTEIL & ERWARTUNGSWERTE
        # -------------------------------------------------------------
        norm_form = (home_home_pts - 7.5) / 7.5
        dynamic_ha = base_ha + (40 * norm_form)

        # Erwartungswert inklusive dynamischem Heimvorteil
        e_home = 1 / (1 + 10 ** ((r_away - (r_home + dynamic_ha)) / 400))
        e_away = 1 - e_home

        # -------------------------------------------------------------
        # 5. ELO-UPDATE NACH DEM SPIEL
        # -------------------------------------------------------------
        if pd.notna(res):
            if res == 0:
                s_home, s_away = 1.0, 0.0
            elif res == 1:
                s_home, s_away = 0.5, 0.5
            else:  # res == 2
                s_home, s_away = 0.0, 1.0

            current_elo[home_id] = r_home + k_factor * (s_home - e_home)
            current_elo[away_id] = r_away + k_factor * (s_away - e_away)

    # Spalten an das DataFrame anfügen
    df_clean["home_elo"] = home_elo_before
    df_clean["away_elo"] = away_elo_before
    df_clean["elo_diff"] = elo_diff_list

    return df_clean











