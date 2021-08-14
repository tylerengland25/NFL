# The Teams Data Frame is structured as follows:
#
# TId = Team Id <String> (PRIMARY KEY)
# Name = Team name <String>
# Abbreviation = Team abbreviation <String>
# Conference = Conference <String>
# Division = Division <String>
#
#   ---------------------------------------------------
#  | TId | Name | Abbreviation | Conference | Division |
#  |-----|------|--------------|------------|----------|
#  |     |      |              |            |          |

import pandas as pd


# Generates all team ids starting with "T1" to "T32"
def generate_team_ids():
    team_ids = []
    for i in range(1, 33):
        tid = "T" + (str(i))
        team_ids.append(tid)
    return team_ids


def main():
    Teams = pd.read_csv("nfl_teams.csv")

    team_ids = generate_team_ids()

    Teams['ID'] = team_ids
    Teams.rename({"ID": "TId"}, axis=1, inplace=True)
    Teams = Teams.set_index(["TId"])

    print(Teams)

    return Teams


if __name__ == '__main__':
    main()
