from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import pandas as pd


def scrape_game(home, visitor, number, week):
    """
    Scrapes statistics for given game
    :param home: home team separated by - (e.g. kansas-city-chiefs)
    :param visitor: away team separated by - (e.g. houston-texans)
    :param number: unique number associated with date of game
    :param week: week number
    :return: dictionary with game stats
    """
    url = "https://www.footballdb.com/games/boxscore/" \
          + visitor + "-vs-" + home + "-" + number
    hdr = {
        'User-Agent': """Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) 
    Chrome/92.0.4515.159 Safari/537.36""",
        'Connection': 'close'}
    req = Request(url, headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, "html.parser")

    team_stats = soup.find("div", attrs={"id": "divBox_team",
                                         "class": "hidden-xs"})
    tables = team_stats.find_all("tbody")

    stats = {"Week": week, "Home": home, "Away": visitor,
             "H_1st": tables[0].find_all("td")[2].text, "A_1st": tables[0].find_all("td")[1].text,
             "H_R_1st": tables[0].find_all("td")[5].text, "A_R_1st": tables[0].find_all("td")[4].text,
             "H_P_1st": tables[0].find_all("td")[8].text, "A_P_1st": tables[0].find_all("td")[7].text,
             "H_Total_Y": tables[0].find_all("td")[14].text, "A_Total_Y": tables[0].find_all("td")[13].text,
             "H_Rush_Yds": tables[0].find_all("td")[17].text, "A_Rush_Yds": tables[0].find_all("td")[16].text,
             "H_Rush_Ply": tables[0].find_all("td")[20].text, "A_Rush_Ply": tables[0].find_all("td")[19].text,
             "H_Pass_Yds": tables[0].find_all("td")[35].text, "A_Pass_Yds": tables[0].find_all("td")[34].text,
             "H_Att": tables[0].find_all("td")[29].text.split('-')[0],
             "A_Att": tables[0].find_all("td")[28].text.split('-')[0],
             "H_Cmp": tables[0].find_all("td")[29].text.split('-')[1],
             "A_Cmp": tables[0].find_all("td")[28].text.split('-')[1],
             "H_Int": tables[0].find_all("td")[29].text.split('-')[2],
             "A_Int": tables[0].find_all("td")[28].text.split('-')[2],
             "H_Sacks": tables[0].find_all("td")[31].text.split('-')[0],
             "A_Sacks": tables[0].find_all("td")[32].text.split('-')[0],
             "H_Sack_Yds": tables[0].find_all("td")[31].text.split('-')[1],
             "A_Sack_Yds": tables[0].find_all("td")[32].text.split('-')[1],
             "H_Punts": tables[1].find_all("td")[2].text.split('-')[0],
             "A_Punts": tables[1].find_all("td")[1].text.split('-')[0],
             "H_Punt_Yds": tables[1].find_all("td")[2].text.split('-')[1],
             "A_Punt_Yds": tables[1].find_all("td")[1].text.split('-')[1],
             "H_Punt_Ret_Yds": tables[1].find_all("td")[8].text.split('-')[1],
             "A_Punt_Ret_Yds": tables[1].find_all("td")[7].text.split('-')[1],
             "H_Kick_Ret_Yds": tables[1].find_all("td")[11].text.split('-')[1],
             "A_Kick_Ret_Yds": tables[1].find_all("td")[10].text.split('-')[1],
             "H_Int_Yds": tables[1].find_all("td")[14].text.split('-')[1],
             "A_Int_Yds": tables[1].find_all("td")[13].text.split('-')[1],
             "H_Pen_Yds": tables[1].find_all("td")[17].text.split('-')[1],
             "A_Pen_Yds": tables[1].find_all("td")[16].text.split('-')[1],
             "H_Fum": tables[1].find_all("td")[19].text.split('-')[1],
             "A_Fum": tables[1].find_all("td")[20].text.split('-')[1],
             "H_Fg_Att": tables[1].find_all("td")[23].text.split('-')[1],
             "A_Fg_Att": tables[1].find_all("td")[22].text.split('-')[1],
             "H_Fg_Cmp": tables[1].find_all("td")[23].text.split('-')[0],
             "A_Fg_Cmp": tables[1].find_all("td")[22].text.split('-')[0],
             "H_3rd_Att": tables[1].find_all("td")[26].text.split('-')[1],
             "A_3rd_Att": tables[1].find_all("td")[25].text.split('-')[1],
             "H_3rd_Cmp": tables[1].find_all("td")[26].text.split('-')[0],
             "A_3rd_Cmp": tables[1].find_all("td")[25].text.split('-')[0],
             "H_4th_Att": tables[1].find_all("td")[29].text.split('-')[1],
             "A_4th_Att": tables[1].find_all("td")[28].text.split('-')[1],
             "H_4th_Cmp": tables[1].find_all("td")[29].text.split('-')[0],
             "A_4th_Cmp": tables[1].find_all("td")[28].text.split('-')[0],
             "H_Total_Ply": tables[1].find_all("td")[32].text, "A_Total_Ply": tables[1].find_all("td")[31].text,
             "H_Poss": tables[1].find_all("td")[38].text, "A_Poss": tables[1].find_all("td")[37].text
             }

    return stats


def scrape_season(year):
    url = "https://www.footballdb.com/games/index.html?lg=NFL&yr=" + year
    hdr = {
        'User-Agent': """Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) 
        Chrome/92.0.4515.159 Safari/537.36""",
        'Connection': 'close'}
    req = Request(url, headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, "html.parser")

    season_stats = pd.DataFrame()

    weeks = soup.find_all("tbody")
    for week in range(len(weeks)):
        print(week)
        games = weeks[week].find_all("tr")
        for game in games:
            link = game.find_all("td")[6].a["href"]
            away = link.split('/')[-1].split('vs')[0][:-1]
            home = "-".join(link.split('/')[-1].split('vs')[1].split('-')[:-1])[1:]
            number = link.split('/')[-1].split('vs')[1].split('-')[-1]
            game_stats = scrape_game(home, away, number, week + 1)
            game_stats["H_Score"] = game.find_all("td")[4].text
            game_stats["A_Score"] = game.find_all("td")[2].text
            game_stats["Year"] = year
            season_stats = season_stats.append(game_stats, ignore_index=True)

    return season_stats


def convert_poss(time):
    """
    Assumes time is minutes and seconds separated by :
    :param time: String "MM:SS"
    :return: Integer of seconds of possession
    """
    min_sec = time.split(":")
    mins = int(min_sec[0])
    secs = int(min_sec[1])
    return (mins * 60) + secs


def last_5_games():
    """
    Selects last 5 games of stats for each game
    :return:
    """
    # Load data
    home_df = pd.read_csv("backend/data/weekly_stats.csv")
    away_df = pd.read_csv("backend/data/weekly_stats.csv")

    home_df.rename(
        columns=lambda col: col.lower()[2:]
        if "H_" in col or len(col.split('_')) == 1
        else col.lower()[2:] + "_def",
        inplace=True
    )
    home_df.rename(columns={"ek": "week", "ar": "year", "me": "team", "ay": "opponent"}, inplace=True)
    home_df["home_away"] = 1

    away_df.rename(
        columns=lambda col: col.lower()[2:]
        if "A_" in col or len(col.split('_')) == 1
        else col.lower()[2:] + "_def",
        inplace=True
    )
    away_df.rename(columns={"ek": "week", "ar": "year", "ay": "team", "me": "opponent"}, inplace=True)
    away_df["home_away"] = 0

    # merge home and away team data into one dataframe
    teams_df = pd.concat([home_df, away_df], ignore_index=True, sort=True)
    season_length = teams_df.groupby(["team", "year"])["week"].max().reset_index()
    season_length.rename(columns={"week": "season_length"}, inplace=True)
    teams_df = pd.merge(teams_df, season_length,
                        how="left",
                        on=["team", "year"])
    # teams_df = teams_df[(teams_df["week"] > 5) | (teams_df["year"] != 2010)]

    # dataframe for last 5 games
    last_five = teams_df.copy()
    for i in range(1, 6):
        week_i = teams_df.copy()
        week_i["week"] = week_i["week"] + i
        week_i.loc[week_i["week"] > week_i["season_length"], "year"] = \
            week_i.loc[week_i["week"] > week_i["season_length"], "year"] + 1
        week_i.loc[week_i["week"] > week_i["season_length"], "week"] = \
            week_i.loc[week_i["week"] > week_i["season_length"], "week"] \
            - week_i.loc[week_i["week"] > week_i["season_length"], "season_length"]

        last_five = pd.merge(last_five, week_i,
                             how="left",
                             on=["team", "week", "year"],
                             suffixes=["", "_" + str(i)])

    last_five.fillna(0, inplace=True)
    last_five.to_csv("backend/data/last_five.csv")


def aggregate_stats():
    """
    Aggregates last 5 games of stats for each game
    :return:
    """
    df = pd.read_csv("backend/data/weekly_stats.csv")

    # Convert possession time to seconds
    df['A_Poss'] = df['A_Poss'].apply(lambda x: convert_poss(x))
    df['H_Poss'] = df['H_Poss'].apply(lambda x: convert_poss(x))

    df = df.fillna(0)

    # Columns to keep totals on (numerical attributes)
    cols = {'Fum', 'Pen_Yds', '4th_Cmp', 'Punt_Yds', '4th_Att', '3rd_Cmp', 'R_1st', 'P_1st', 'Rush_Ply', 'Punts',
            'Sack_Yds', 'Int_Yds', 'Kick_Ret_Yds', 'Poss', 'Int', 'Fg_Att', '1st', 'Sacks', 'Cmp', 'Score',
            'Punt_Ret_Yds', 'Att', 'Fg_Cmp', '3rd_Att', 'Pass_Yds', 'Rush_Yds', 'Total_Y', 'Total_Ply'}

    # Total dataframe
    total_df = pd.DataFrame()

    for index, game in df.iterrows():
        week = game["Week"]
        year = game["Year"]
        print("Week {}, Year {}".format(week, year))
        teams = [game["Home"], game["Away"]]
        for team in teams:
            totals = {"Team": team, "Week": game["Week"], "Year": game["Year"],
                      "Home": game["Home"], "Away": game["Away"]}
            if game["Home"] == team:
                totals["Opponent"] = game["Away"]
            else:
                totals["Opponent"] = game["Home"]

            for stat in cols:
                totals[stat] = 0
                totals["Opp_" + stat] = 0

            weekly_df = pd.DataFrame()
            for i in range(1, 6):
                if week - i <= 0 and year != 2010:
                    max_week = df[((df["Home"] == team) | (df["Away"] == team)) & (df["Year"] == year)]["Week"].max()
                    week_df = df[((df["Home"] == team) | (df["Away"] == team)) &
                                 (df["Week"] == max_week + (week - i)) & (df["Year"] == year)]
                    weekly_df = weekly_df.append(week_df)
                elif week - 5 <= 0 and year == 2010:
                    pass
                else:
                    week_df = df[((df["Home"] == team) | (df["Away"] == team)) &
                                 (df["Week"] == week - i) & (df["Year"] == year)]
                    weekly_df = weekly_df.append(week_df)

            if game["Home"] == team and not weekly_df.empty:
                for stat in cols:
                    totals[stat] += weekly_df["H_" + stat].sum() / weekly_df.shape[0]
                    totals["Opp_" + stat] += weekly_df["A_" + stat].sum() / weekly_df.shape[0]
            elif game["Away"] == team and not weekly_df.empty:
                for stat in cols:
                    totals[stat] += weekly_df["A_" + stat].sum() / weekly_df.shape[0]
                    totals["Opp_" + stat] += weekly_df["H_" + stat].sum() / weekly_df.shape[0]

            total_df = total_df.append(totals, ignore_index=True)

    total_df.to_csv("backend/data/aggregated_stats.csv")


def input_data():
    """
    Calculates stats that needed for model
    :return:
    """
    # Load and create dataframes
    agg_df = pd.read_csv("backend/data/aggregated_stats.csv")
    week_df = pd.read_csv("backend/data/weekly_stats.csv")
    df = pd.DataFrame()

    # Team info
    df["Team"] = agg_df["Team"]
    df["Home"] = agg_df["Home"]
    df["Away"] = agg_df["Away"]
    df["Year"] = agg_df["Year"]
    df["Week"] = agg_df["Week"]
    # Points scored per game (Home/Away)
    df["H_pts_per_game_scored"] = agg_df["H_Score"].divide(agg_df["H_Games"])
    df["A_pts_per_game_scored"] = agg_df["A_Score"].divide(agg_df["A_Games"])
    # Points allowed per game (Home/Away)
    df["H_pts_per_game_allowed"] = agg_df["Opp_A_Score"].divide(agg_df["H_Games"])
    df["A_pts_per_game_allowed"] = agg_df["Opp_H_Score"].divide(agg_df["A_Games"])
    # Yards gained per game (Home/Away)
    df["H_yds_per_game_gained"] = agg_df["H_Total_Y"].divide(agg_df["H_Games"])
    df["A_yds_per_game_gained"] = agg_df["A_Total_Y"].divide(agg_df["A_Games"])
    # Yards allowed per game (Home/Away)
    df["H_yds_per_game_allowed"] = agg_df["Opp_A_Total_Y"].divide(agg_df["H_Games"])
    df["A_yds_per_game_allowed"] = agg_df["Opp_H_Total_Y"].divide(agg_df["A_Games"])
    # Yards per point scored (Home/Away)
    df["H_yds_per_point_scored"] = agg_df["H_Total_Y"].divide(agg_df["H_Score"])
    df["A_yds_per_point_scored"] = agg_df["A_Total_Y"].divide(agg_df["A_Score"])
    # Yards per point allowed (Home/Away)
    df["H_yds_per_point_allowed"] = agg_df["Opp_A_Total_Y"].divide(agg_df["Opp_A_Score"])
    df["A_yds_per_point_allowed"] = agg_df["Opp_H_Total_Y"].divide(agg_df["Opp_H_Score"])
    # Yards per play by offense (Home/Away)
    df["H_yds_per_ply_gained"] = agg_df["H_Total_Y"].divide(agg_df["H_Total_Ply"])
    df["A_yds_per_ply_gained"] = agg_df["A_Total_Y"].divide(agg_df["A_Total_Ply"])
    # Yards per play allowed by defense (Home/Away)
    df["H_yds_per_ply_allowed"] = agg_df["Opp_A_Total_Y"].divide(agg_df["Opp_A_Total_Ply"])
    df["A_yds_per_ply_allowed"] = agg_df["Opp_H_Total_Y"].divide(agg_df["Opp_H_Total_Ply"])
    # Sacks allowed per play (Home/Away)
    df["H_sacks_per_ply_allowed"] = agg_df["Opp_A_Sacks"].divide(agg_df["H_Total_Ply"])
    df["A_sacks_per_ply_allowed"] = agg_df["Opp_H_Sacks"].divide(agg_df["A_Total_Ply"])
    # Sacks per play by defense (Home/Away)
    df["H_sacks_per_ply"] = agg_df["H_Sacks"].divide(agg_df["Opp_A_Total_Ply"])
    df["A_sacks_per_ply"] = agg_df["A_Sacks"].divide(agg_df["Opp_H_Total_Ply"])
    # Interceptions per play by defense (Home/Away)
    df["H_int_per_ply"] = agg_df["H_Int"].divide(agg_df["Opp_A_Total_Ply"])
    df["A_int_per_Ply"] = agg_df["A_Int"].divide(agg_df["Opp_H_Total_Ply"])
    # Interceptions thrown per play (Home/Away)
    df["H_int_per_ply_thrown"] = agg_df["Opp_A_Int"].divide(agg_df["H_Total_Ply"])
    df["A_int_per_ply_thrown"] = agg_df["Opp_H_Int"].divide(agg_df["A_Total_Ply"])
    # Ratio of pass to run plays (Home/Away)
    df["H_pass_run_ratio"] = agg_df["H_Att"].divide(agg_df["H_Rush_Ply"])
    df["A_pass_run_ratio"] = agg_df["A_Att"].divide(agg_df["A_Rush_Ply"])
    # Yards per pass by offense (Home/Away)
    df["H_yds_per_pass_gained"] = agg_df["H_Pass_Yds"].divide(agg_df["H_Att"])
    df["A_yds_per_pass_gained"] = agg_df["A_Pass_Yds"].divide(agg_df["A_Att"])
    # Yards per pass allowed by defense (Home/Away)
    df["H_yds_per_pass_allowed"] = agg_df["Opp_A_Pass_Yds"].divide(agg_df["Opp_A_Att"])
    df["A_yds_per_pass_allowed"] = agg_df["Opp_H_Pass_Yds"].divide(agg_df["Opp_H_Att"])
    # Yards per run by offense (Home/Away)
    df["H_yds_per_run_gained"] = agg_df["H_Rush_Yds"].divide(agg_df["H_Rush_Ply"])
    df["A_yds_per_run_gained"] = agg_df["A_Rush_Yds"].divide(agg_df["A_Rush_Ply"])
    # Yards per run allowed by defense (Home/Away)
    df["H_yds_per_run_allowed"] = agg_df["Opp_A_Rush_Yds"].divide(agg_df["Opp_A_Rush_Ply"])
    df["A_yds_per_run_allowed"] = agg_df["Opp_H_Rush_Yds"].divide(agg_df["Opp_H_Rush_Ply"])
    # Score of game
    home_scores = []
    away_scores = []
    for index, row in df.iterrows():
        f = week_df[week_df["Home"] == row["Home"]]
        f = f[f["Away"] == row["Away"]]
        f = f[f["Week"] == row["Week"]]
        f = f[f["Year"] == row["Year"]]
        home_score = f["H_Score"].iloc[0]
        away_score = f["A_Score"].iloc[0]
        home_scores.append(home_score)
        away_scores.append(away_score)
    df["H_Score"] = home_scores
    df["A_Score"] = away_scores

    # Fill nan
    df = df.fillna(0)

    df.to_csv("backend/data/input_data.csv")


def input_games():
    """
    Structures data into format needed for each game, showing home team's stats and away team's stats
    :return:
    """
    weekly_games = pd.read_csv("backend/data/weekly_stats.csv")
    input_df = pd.read_csv("backend/data/input_data.csv")

    df = pd.DataFrame()

    for index, game in weekly_games.iterrows():
        input_game = {}
        home_team = game["Home"]
        away_team = game["Away"]
        week = game["Week"]
        year = game["Year"]
        home = input_df[(input_df["Team"] == home_team) &
                        (input_df["Home"] == home_team) &
                        (input_df["Away"] == away_team) &
                        (input_df["Week"] == week) &
                        (input_df["Year"] == year)]
        away = input_df[(input_df["Team"] == away_team) &
                        (input_df["Home"] == home_team) &
                        (input_df["Away"] == away_team) &
                        (input_df["Week"] == week) &
                        (input_df["Year"] == year)]

        for col in input_df.columns:
            if col[0] == "H":
                input_game[col] = home[col].iloc[0]
            elif col[0] == "A":
                input_game[col] = away[col].iloc[0]

        input_game["Year"] = home["Year"].iloc[0]
        input_game["Week"] = home["Week"].iloc[0]
        df = df.append(input_game, ignore_index=True)

    df.to_csv("backend/data/inputs.csv")


def main():
    ten_years = pd.DataFrame()
    for season in range(2010, 2021):
        print(season)
        season_stats = scrape_season(str(season))
        ten_years = pd.concat([ten_years, season_stats], ignore_index=True)

    ten_years.to_csv("backend/data/weekly_stats.csv")

    # aggregates stats per season per team
    aggregate_stats()

    # create inputs
    input_data()


if __name__ == '__main__':
    df = last_5_games()
    print(df)
    print()
