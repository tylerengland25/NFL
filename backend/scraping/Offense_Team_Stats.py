# The Offense_Team_Stats Data Frame is structured as follows:
#
# TId = Team Id <String> (COMPOSITE KEY)
# SId = Season Id <String> (COMPOSITE KEY)
# Yds = Total yards <Integer>
# Ply = Total plays <Integer>
# Y/P = Yards per play <Integer>
# TO = Total turnovers <Integer>
# FL = Fumbles lost <Integer>
# 1stD = Total first downs <Integer>
# PCmp = Passes completed <Integer>
# PAtt = Passes attempted <Integer>
# PYds = Passing yards <Integer>
# PTD = Passing touchdowns <Integer>
# Int = Interceptions <Integer>
# NY/A = Net yards gained per pass attempt <Double>
# P1stD = First downs by passing <Integer>
# RAtt = Rushing attempts <Integer>
# RYds = Rushing yards <Integer>
# RTD = Rushing touchdowns <Integer>
# Y/A = Yards per rushing attempt <Double>
# R1stD = First downs by rushing <Integer>
# Pen = Number of penalties <Integer>
# PenYds = Yards from penalties <Integer>
# Sc% = Percentage of drives ending in a score <Double>
# TO% = Percentage of drives ending in a turnover <Double>
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


def scrape_offense_team_stats(year):
    df = pd.read_csv("offense_stats/team_offense_" + str(year) + ".csv").drop(['Rk', 'G', '1stPy'], axis=1)

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
    Offense_Team_Stats = pd.DataFrame()
    for year in years:
        season = scrape_offense_team_stats(year)
        Offense_Team_Stats = Offense_Team_Stats.append(season)

    # print(Offense_Team_Stats)

    return Offense_Team_Stats


if __name__ == '__main__':
    main()
