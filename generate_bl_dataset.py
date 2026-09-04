import pandas as pd
import requests


def create_dataframe(season, matchday):
    url = f"https://api.openligadb.de/getmatchdata/bl1/{season}/{matchday}"
    r = requests.get(url).json()

    # SICHERHEITSCHECK 1: Falls die API einen String statt einer Liste von Matches zurückgibt
    if not isinstance(r, list):
        return pd.DataFrame()

    results = []

    for match in r:
        # SICHERHEITSCHECK 2: Prüfen, ob das Element ein Dictionary ist
        if not isinstance(match, dict):
            continue

        match_results = match.get("matchResults", [])

        end_result = None
        for x in match_results:
            result_type = x.get("resultTypeID")

            if result_type == 2:
                end_result = x
                break

        # Falls kein Endergebnis vorliegt (Spiel noch nicht gespielt/abgeschlossen)
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

    df = pd.DataFrame(results)
    return df

df_spieltag1 = create_dataframe(2020, 1)
print(df_spieltag1)



