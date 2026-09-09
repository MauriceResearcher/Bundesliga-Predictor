import time
import pandas as pd
import requests


def get_full_season_data(season):
    """Holt ALLE Spiele einer kompletten Saison in einem einzigen API-Call.

    Das ist viel schneller und verhindert Rate-Limits.
    """
    url = f"https://api.openligadb.de/getmatchdata/bl1/{season}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f" Fehler bei Saison {season}: Status Code {response.status_code}")
            return pd.DataFrame()
        r = response.json()
    except Exception as e:
        print(f" Fehler bei Anforderung Saison {season}: {e}")
        return pd.DataFrame()

    if not isinstance(r, list):
        print(f" Unerwartete Antwort für Saison {season}")
        return pd.DataFrame()

    results = []

    for match in r:
        if not isinstance(match, dict):
            continue

        match_results = match.get("matchResults", [])

        # Endergebnis suchen (resultTypeID == 2)
        end_result = None
        for x in match_results:
            if x.get("resultTypeID") == 2:
                end_result = x
                break

        # Falls Spiel noch nicht stattgefunden hat / kein Endergebnis vorliegt
        if not end_result:
            continue

        goals_home = end_result.get("pointsTeam1")
        goals_away = end_result.get("pointsTeam2")

        if goals_home > goals_away:
            final_result = 0  # Heimsieg
        elif goals_home < goals_away:
            final_result = 2  # Auswärtssieg
        else:
            final_result = 1  # Unentschieden

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
        results.append(match_dict)

    df_season = pd.DataFrame(results)
    print(f"-> Saison {season}: {len(df_season)} beendete Spiele geladen.")
    return df_season


# ------------------------------------------------------------------
# Saisons 2020 bis 2025 laden
# ------------------------------------------------------------------
all_seasons_dfs = []

# range(2020, 2026) lädt 2020, 2021, 2022, 2023, 2024, 2025
for season in range(2020, 2026):
    df_s = get_full_season_data(season)
    if not df_s.empty:
        all_seasons_dfs.append(df_s)
    time.sleep(0.2)

if all_seasons_dfs:
    full_dataset = pd.concat(all_seasons_dfs, ignore_index=True)

    output_path = (
        r"D:\PycharmProjects\Bundesliga\Datasets\bundesliga_all_seasons.csv"
    )
    full_dataset.to_csv(output_path, index=False)

    print(
        f"\nFERTIG! Insgesamt {len(full_dataset)} Spiele in {output_path} gespeichert."
    )
else:
    print("\nKEINE Daten geladen!")