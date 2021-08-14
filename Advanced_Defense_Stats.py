# The Defense_Team_Stats Data Frame is structured as follows:
#
# TId = Team Id <String> (COMPOSITE KEY)
# SId = Season Id <String> (COMPOSITE KEY)
# DADOT = Average depth of target when targeted as a defender <Double>
# Air = Total air yards on completions by opponent <Integer>
# YAC = Yards after catch by opponent <Integer>
# Bltz = Total times defense blitz the opponent <Integer>
# Bltz% = Blitz per opponent dropback <Double>
# Hrry = Total QB hurries forced by defense <Integer>
# Hrry% = QB hurries per opponent dropback <Double>
# QBKD = Opposing QB hit the ground after the throw <Integer>
# QBKD% = Knockdowns per opponent pass attempt <Double>
# Sk = Total sacks on season <Integer>
# Prss = Total QB Pressures <Integer>
# Prss% = QB Pressures per opponent dropback <Double>
# MTkl = Missed tackles = <Integer>
#
#
#   -------------------------------------------------------------------------------------------------------
#  | TId | SId | DADOT | Air | YAC | Bltz | Bltz% | Hrry | Hrry% | QBKD | QBKD% | Sk | Prss | Prss% | MTkl |
#  |-----|-----|-------|-----|-----|------|-------|------|-------|------|-------|----|------|-------|------|
#  |     |     |       |     |     |      |       |      |       |      |       |    |      |       |      |
#
#
import pandas as pd
import Seasons
import Season_Team_Stats


def scrape_advanced_defense_stats(year):
    url = "https://www.pro-football-reference.com/years/" + str(year) + "/opp.htm"

    df = pd.read_html(url)[1].drop(['G', 'Att', 'Cmp', 'Yds', 'TD'], axis=1)

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
    # scrape last 3 seasons
    years = [2020, 2019, 2018]
    Advanced_Defense_Stats = pd.DataFrame()
    for year in years:
        season = scrape_advanced_defense_stats(year)
        Advanced_Defense_Stats = Advanced_Defense_Stats.append(season)

    # print(Advanced_Defense_Stats)

    return Advanced_Defense_Stats


if __name__ == '__main__':
    main()
