# The Defense_Team_Stats Data Frame is structured as follows:
#
# TId = Team Id <String> (COMPOSITE KEY)
# SId = Season Id <String> (COMPOSITE KEY)
# Yds = Total yards <Integer>
# Ply = Total plays <Integer>
# Y/P = Yards per play <Integer>
# TO = Total takeaways <Integer>
# FL = Fumbles lost <Integer>
# 1stD = Opponent total first downs <Integer>
# PCmp = Opponent passes completed <Integer>
# PAtt = Opponent passes attempted <Integer>
# PYds = Opponent passing yards <Integer>
# PTD = Opponent passing touchdowns <Integer>
# Int = Interceptions <Integer>
# NY/A = Net yards gained per opponent pass attempt <Double>
# P1stD = Opponent first downs by passing <Integer>
# RAtt = Opponent rushing attempts <Integer>
# RYds = Opponent rushing yards <Integer>
# RTD = Opponent rushing touchdowns <Integer>
# Y/A = Opponent yards per rushing attempt <Double>
# R1stD = Opponent first downs by rushing <Integer>
# Pen = Number of penalties committed by team <Integer>
# PenYds = Yards from penalties committed by team <Integer>
# Sc% = Percentage of drives ending in an opponent score <Double>
# TO% = Percentage of drives ending in an opponent turnover <Double>
#
#
#   -------------------------------------------------------------------------------------
#  | TId | SId | Yds | Ply | Y/P | TO | FL | 1stD | PCmp | PAtt | PYds | PTD | Int | -->
#  |-----|-----|-----|-----|-----|----|----|------|------|------|------|-----|-----|-----
#  |     |     |     |     |     |    |    |      |      |      |      |     |     |
#
#   ---------------------------------------------------------------------------
#    NY/A | P1stD | RAtt | RYds | RTD | Y/A | R1stD | Pen | PenYds | Sc% | TO% |
#   ------|-------|------|------|-----|-----|-------|-----|--------|-----|-----|
#         |       |      |      |     |     |       |     |        |     |     |
#
import pandas as pd
import Seasons
import Season_Team_Stats


def scrape_defense_team_stats(year):
    df = pd.read_csv("defense_stats/team_defense_" + str(year) + ".csv").drop(['Rk', 'G', '1stPy'], axis=1)

    df.rename({'Cmp': 'PCmp', 'Att': 'PAtt', 'Yds.1': 'PYds', 'TD': 'PTD',
               '1stD.1': 'P1stD', 'Att.1': 'RAtt', 'Yds.2': 'RYds', 'TD.1': 'RTD',
               '1stD.2': 'R1stD', 'Yds.3': 'PenYds'})

    # map team name to TId
    team_ids = []
    for i in df.index.values:
        tid = Seasons.map_team_tid(df.loc[i]["Tm"].strip('+*'))
        team_ids.append(tid)
    df["Tm"] = team_ids
    df.rename(columns={"Tm": "TId"}, inplace=True)

    # add season id column
    df["SId"] = Season_Team_Stats.map_season_sid(year)

    # set index to SId and TId
    df = df.set_index(["SId", "TId"])

    return df


def main():
    # scrape last 5 seasons
    years = [2020, 2019, 2018, 2017, 2016]
    Defense_Team_Stats = pd.DataFrame()
    for year in years:
        season = scrape_defense_team_stats(year)
        Defense_Team_Stats = Defense_Team_Stats.append(season)

    print(Defense_Team_Stats)

    return Defense_Team_Stats


if __name__ == '__main__':
    main()
