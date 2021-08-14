# The Season_Team_Stats Data Frame is structured as follows:
#
# TId = Team Id <String> (COMPOSITE KEY)
# SId = Season Id <String> (COMPOSITE KEY)
# W = Number of wins <Integer>
# L = Number of Losses <Integer>
# T = Number of Ties <Integer>
# W-L% = Win/Loss ratio <Double>
# PF = Points for <Integer>
# PA = Points against <Integer>
# PD = Point differential <Integer>
# MoV = Average margin of victory <Double>
# SoS = Strength of schedule <Double>
# SrS = Simple rating system <Double>
# OSrS = Offense simple rating system <Double>
# DSrS = Defense simple rating system <Double>
#
#
#   -----------------------------------------------------------------------------
#  | TId | SId | W | L | T | W-L% | PF | PA | PD | MoV | Sos | SrS | OSrS | DSrS |
#  |-----|-----|---|---|---|------|----|----|----|-----|-----|-----|------|------|
#  |     |     |   |   |   |      |    |    |    |     |     |     |      |      |
#

import pandas as pd
import Seasons


def map_season_sid(year):
    seasons_sid = {"2020": "S1",
                   "2019": "S2",
                   "2018": "S3",
                   "2017": "S4",
                   "2016": "S5"}

    return seasons_sid[str(year)]


def scrape_season_team_stats(year):

    url = "https://www.pro-football-reference.com/years/" + str(year) + "/"

    AFC_df = pd.read_html(url)[0]
    NFC_df = pd.read_html(url)[1]

    # remove divisional rows
    AFC_df = AFC_df.drop([0, 5, 10, 15], axis=0)
    NFC_df = NFC_df.drop([0, 5, 10, 15], axis=0)

    # combine data frames
    df = pd.concat([AFC_df, NFC_df], ignore_index=True)

    # map team name to TId
    team_ids = []
    for i in df.index.values:
        tid = Seasons.map_team_tid(df.loc[i]["Tm"].strip('+*'))
        team_ids.append(tid)
    df["Tm"] = team_ids
    df.rename(columns={"Tm": "TId"}, inplace=True)

    # add season id column
    df["SId"] = map_season_sid(year)

    # set index to SId and TId
    df = df.set_index(["SId", "TId"])

    df = df.fillna(0)

    return df


def main():

    # scrape last 5 seasons
    years = [2020, 2019, 2018, 2017, 2016]
    Season_Team_Stats = pd.DataFrame()
    for year in years:
        season = scrape_season_team_stats(year)
        Season_Team_Stats = Season_Team_Stats.append(season)

    Season_Team_Stats = Season_Team_Stats.fillna(0)

    # print(Season_Team_Stats)

    return Season_Team_Stats


if __name__ == '__main__':
    main()
