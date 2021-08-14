# The Seasons Data Frame is structured as follows:
#
# SId = Season Id <String> (PRIMARY KEY)
# Year = Year (YYYY) <Integer>
# SuperBowl = TId of who won that year <String>
#
#   -------------------------
#  | SId | Year |  SuperBowl |
#  |-----|------|------------|
#  |     |      |            |
#  |     |      |            |

import pandas as pd
import Teams
import re


def map_team_tid(team):

    team_names = ["Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
                  "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
                  "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
                  "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
                  "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
                  "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
                  "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
                  "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Football Team"]

    team_ids = Teams.generate_team_ids()

    team_id_dict = {team_names[i]: team_ids[i] for i in range(len(team_ids))}

    team_id_dict["Oakland Raiders"] = team_id_dict["Las Vegas Raiders"]
    team_id_dict["Washington Redskins"] = team_id_dict["Washington Football Team"]
    team_id_dict["San Diego Chargers"] = team_id_dict["Los Angeles Chargers"]

    return team_id_dict[team]


def generate_season_id():
    season_ids = []
    for i in range(1, 6):
        sid = "S" + (str(i))
        season_ids.append(sid)
    return season_ids


def scrape_seasons():

    url = "https://www.pro-football-reference.com/years/"
    num_years = 5

    df = pd.read_html(url)[0]

    # keep past 5 seasons and drop league column
    df = df.loc[range(1, num_years + 1)].drop(["Lg"], axis=1)

    # rename columns
    df.columns = ["Year", "SuperBowl"]

    # obtain Super Bowl winner
    sb_winners = []
    for i in df.index.values:
        txt = df.loc[i]["SuperBowl"]
        sb_winners.append(re.search(" \w+ \w+ \w*", txt).group().strip())
    df["SuperBowl"] = sb_winners

    return df


def clean_data(df):

    # map team names to team ids
    team_ids = []
    for i in df.index.values:
        tid = map_team_tid(df.loc[i]["SuperBowl"])
        team_ids.append(tid)
    df["SuperBowl"] = team_ids

    # set season ids as index
    df["SId"] = generate_season_id()
    df = df.set_index(["SId"])

    return df


def main():

    Seasons = scrape_seasons()

    Seasons = clean_data(Seasons)

    # print(Seasons)

    return Seasons


if __name__ == '__main__':
    main()

