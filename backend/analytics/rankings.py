import pandas as pd
from backend.scraping.weekly.weekly import convert_poss


def team_ranks_offense(week, season, last_num=5):
    # Load data and filter for last 5 weeks
    df = pd.read_csv("backend/data/current_season_stats.csv")
    df = df[(df["Week"] < week) & (df["Week"] >= week - last_num) & (df["Year"] == season)]

    # Split dataframe into each team
    away_cols = [col for col in df.columns if ("A_" in col) and (col != "A_Sacks") and (col != "A_Sack_Yds")]
    away_cols.extend(["Away", "H_Sacks", "H_Sack_Yds"])
    home_cols = [col for col in df.columns if ("H_" in col) and (col != "H_Sacks") and (col != "H_Sack_Yds")]
    home_cols.extend(["Home", "A_Sacks", "A_Sack_Yds"])
    away_df = df[away_cols]
    home_df = df[home_cols]

    # Rename columns
    away_cols = [col[2:] for col in away_cols]
    away_df.columns = away_cols
    away_df = away_df.rename(columns={"ay": "Team", "ek": "Week"})
    home_cols = [col[2:] for col in home_cols]
    home_df.columns = home_cols
    home_df = home_df.rename(columns={"me": "Team", "ek": "Week"})

    # Combine dataframes
    df = pd.concat([away_df, home_df])

    # Convert time of possession
    df["Poss"] = df["Poss"].apply(lambda x: convert_poss(x))

    # Group by team and sum
    df = df.groupby(["Team"]).sum()
    df = df.divide(last_num)

    # Remove unwanted columns
    df = df.drop(["Int_Yds"], axis=1)

    # Feature Engineer
    df["Cmp_perc"] = df["Cmp"] / df["Att"]
    df["TO"] = df["Int"] + df["Fum"]
    df["3rd_perc"] = df["3rd_Cmp"] / df["3rd_Att"]
    df["4th_perc"] = df["4th_Cmp"] / df["4th_Att"]
    df["Fg_perc"] = df["Fg_Cmp"] / df["Fg_Att"]
    df["Pass_Yds_Per_Cmp"] = df["Pass_Yds"] / df["Cmp"]
    df["Rush_Yds_Per_Carry"] = df["Rush_Yds"] / df["Rush_Ply"]
    df["Yds_Per_Ply"] = df["Total_Y"] / df["Total_Ply"]
    df["Yds_Per_Sack"] = df["Sack_Yds"] / df["Sacks"]

    # Rank columns
    for col in df.columns:
        df[col] = df[col].rank(method='max', ascending=False)

    # Invert table
    ranks = pd.DataFrame()
    for col in df.columns:
        rankings = df[col].sort_values(ascending=True)
        ranks[col] = rankings.index
    ranks.index = [i + 1 for i in ranks.index]

    return ranks


def team_ranks_defense(week, season, last_num=5):
    # Load data and filter for last 5 weeks
    df = pd.read_csv("backend/data/current_season_stats.csv")
    df = df[(df["Week"] < week) & (df["Week"] >= week - last_num) & (df["Year"] == season)]

    # Split dataframe into each team
    away_cols = [col for col in df.columns if ("H_" in col) and (col != "H_Sacks") and (col != "H_Sack_Yds")]
    away_cols.extend(["Away", "A_Sacks", "A_Sack_Yds"])
    home_cols = [col for col in df.columns if ("A_" in col) and (col != "A_Sacks") and (col != "A_Sack_Yds")]
    home_cols.extend(["Home", "H_Sacks", "A_Sack_Yds"])
    away_df = df[away_cols]
    home_df = df[home_cols]

    # Rename columns
    away_cols = [col[2:] for col in away_cols]
    away_df.columns = away_cols
    away_df = away_df.rename(columns={"ay": "Team", "ek": "Week"})
    home_cols = [col[2:] for col in home_cols]
    home_df.columns = home_cols
    home_df = home_df.rename(columns={"me": "Team", "ek": "Week"})

    # Combine dataframes
    df = pd.concat([away_df, home_df])

    # Convert time of possession
    df["Poss"] = df["Poss"].apply(lambda x: convert_poss(x))

    # Group by team and sum
    df = df.groupby(["Team"]).sum()
    df = df.divide(last_num)

    # Remove unwanted columns
    df = df.drop(["Kick_Ret_Yds", "Pen_Yds", "Poss", "Punt_Ret_Yds", "Punts", "Punt_Yds", ], axis=1)

    # Feature Engineer
    df["Cmp_perc"] = df["Cmp"] / df["Att"]
    df["TO"] = df["Int"] + df["Fum"]
    df["3rd_perc"] = df["3rd_Cmp"] / df["3rd_Att"]
    df["4th_perc"] = df["4th_Cmp"] / df["4th_Att"]
    df["Fg_perc"] = df["Fg_Cmp"] / df["Fg_Att"]
    df["Pass_Yds_Per_Cmp"] = df["Pass_Yds"] / df["Cmp"]
    df["Rush_Yds_Per_Carry"] = df["Rush_Yds"] / df["Rush_Ply"]
    df["Yds_Per_Ply"] = df["Total_Y"] / df["Total_Ply"]
    df["Yds_Per_Sack"] = df["Sack_Yds"] / df["Sacks"]

    # Rank columns
    descend_cols = ["Int", "Fum", "TO", "Sacks", "Sack_Yds", "Yds_Per_Sack"]
    for col in df.columns:
        ascend_bool = True
        if col in descend_cols:
            ascend_bool = False
        df[col] = df[col].rank(method='max', ascending=ascend_bool)

    # Invert table
    ranks = pd.DataFrame()
    for col in df.columns:
        rankings = df[col].sort_values(ascending=True)
        ranks[col] = rankings.index
    ranks.index = [i + 1 for i in ranks.index]

    return ranks


def week_matchups(week):
    # Load data
    offense_ranks = pd.read_excel("backend/data/rankings/offensive_ranks.xlsx", sheet_name=None)
    defense_ranks = pd.read_excel("backend/data/rankings/defensive_ranks.xlsx", sheet_name=None)
    odds = pd.read_excel("backend/data/odds/nfl odds 2021-22.xlsx")

    # Filter data
    odds = odds[odds["Week"] == week]

    # Stats
    stats = ["First Downs Gained By Passing", "Passing Yards", "First Downs By Rushing", "Rushing Plays",
             "Rushing Yards", "Scoring", "Total Yards", "Sacks", "Completion Percentage", "Turnovers",
             "3rd Down Percentage", "4th Down Percentage", "Yards Per Completion", "Yards Per Carry", "Yards Per Play"]

    # Match up data for each game
    games = {}
    for index, game in odds.iterrows():
        home = game["Home"].split("-")[-1]
        away = game["Away"].split("-")[-1]
        mathcup = pd.DataFrame()
        for stat in stats:
            off_df = offense_ranks[stat]
            def_df = defense_ranks[stat]
            row = [off_df[off_df[off_df.columns[1]] == game["Home"]]["Unnamed: 0"].values[0],
                   def_df[def_df[def_df.columns[1]] == game["Home"]]["Unnamed: 0"].values[0],
                   off_df[off_df[off_df.columns[1]] == game["Away"]]["Unnamed: 0"].values[0],
                   def_df[def_df[def_df.columns[1]] == game["Away"]]["Unnamed: 0"].values[0]]
            row = pd.Series(row)
            mathcup = mathcup.append(row, ignore_index=True)

        mathcup.columns = pd.MultiIndex.from_product([[home, away], ["offense", "defense"]])
        mathcup.index = stats

        games[away + " @ " + home] = mathcup

    return games


def main():
    season = 2021
    week = 22
    last_number_games = week - 1

    sheet_names = {"1st": "1st Downs", "3rd_Att": "3rd Down Attempts", "3rd_Cmp": "3rd Down Completions",
                   "4th_Att": "4th Down Attempts", "4th_Cmp": "4th Down Completions", "Att": "Passing Attempts",
                   "Cmp": "Passing Completions", "Fg_Att": "Field Goal Attempts", "Fg_Cmp": "Field Goal Completions",
                   "Fum": "Fumbles", "Int": "Interceptions", "Kick_Ret_Yds": "Kick Return Yards",
                   "P_1st": "First Downs Gained By Passing", "Pass_Yds": "Passing Yards", "Pen_Yds": "Penalty Yards",
                   "Poss": "Time of Possession", "Punt_Ret_Yds": "Punt Return Yards", "Punt_Yds": "Punt Yards",
                   "Punts": "Punts", "R_1st": "First Downs By Rushing", "Rush_Ply": "Rushing Plays",
                   "Rush_Yds": "Rushing Yards", "Score": "Scoring", "Total_Ply": "Total Amount of Plays",
                   "Total_Y": "Total Yards", "Sacks": "Sacks", "Sack_Yds": "Total Sack Yards",
                   "Cmp_perc": "Completion Percentage", "TO": "Turnovers", "3rd_perc": "3rd Down Percentage",
                   "4th_perc": "4th Down Percentage", "Fg_perc": "Field Goal Percentage",
                   "Pass_Yds_Per_Cmp": "Yards Per Completion", "Rush_Yds_Per_Carry": "Yards Per Carry",
                   "Yds_Per_Ply": "Yards Per Play", "Yds_Per_Sack": "Yards Per Sack", "Int_Yds": "Interception Yards"}

    # Offensive rankings
    offense = team_ranks_offense(week, season, last_number_games)
    with pd.ExcelWriter("backend/data/rankings/offensive_ranks.xlsx") as writer:
        for col in offense.columns:
            offense[col].to_excel(writer, sheet_name=sheet_names[col])

    # Defensive rankings
    defense = team_ranks_defense(week, season, last_number_games)
    with pd.ExcelWriter("backend/data/rankings/defensive_ranks.xlsx") as writer:
        for col in defense.columns:
            defense[col].to_excel(writer, sheet_name=sheet_names[col])

    # Match ups
    matchups = week_matchups(week)
    with pd.ExcelWriter("backend/data/rankings/game_matchups.xlsx") as writer:
        for matchup in matchups:
            matchups[matchup].to_excel(writer, sheet_name=matchup)


if __name__ == '__main__':
    main()
