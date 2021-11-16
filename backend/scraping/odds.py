import pandas as pd


def odds_season(filename):
    df = pd.read_excel(filename)
    df["Year"] = filename.split("-")[0][-4:]

    home_df = df[df["VH"] == "H"].reset_index()[["Team", "ML", "Year", "Week"]]
    home_df.rename(columns={"Team": "Home"}, inplace=True)
    away_df = df[df["VH"] == "V"].reset_index()[["Team", "ML", "Year", "Week"]]
    away_df.rename(columns={"Team": "Away"}, inplace=True)
    df = home_df.join(away_df, lsuffix="_h", rsuffix="_a")

    return df


if __name__ == '__main__':
    files = ["backend/data/odds/nfl odds 2010-11.xlsx", "backend/data/odds/nfl odds 2011-12.xlsx",
             "backend/data/odds/nfl odds 2012-13.xlsx", "backend/data/odds/nfl odds 2013-14.xlsx",
             "backend/data/odds/nfl odds 2014-15.xlsx", "backend/data/odds/nfl odds 2015-16.xlsx",
             "backend/data/odds/nfl odds 2016-17.xlsx", "backend/data/odds/nfl odds 2017-18.xlsx",
             "backend/data/odds/nfl odds 2018-19.xlsx", "backend/data/odds/nfl odds 2019-20.xlsx",
             "backend/data/odds/nfl odds 2020-21.xlsx"]

    odds_df = pd.DataFrame()
    for file in files:
        odds_df = odds_df.append(odds_season(file))

    odds_df.drop(["Week_a", "Year_a"], axis=1, inplace=True)
    odds_df.rename(columns={"Year_h": "Year", "Week_h": "Week"}, inplace=True)

    test = odds_df[odds_df["Away"] == "LVRaiders"]

    teams_dict = {'Philadelphia': 'philadelphia-eagles', 'St.Louis': 'st-louis-rams',
                  'TampaBay': 'tampa-bay-buccaneers', 'NYGiants': 'new-york-giants', 'GreenBay': 'green-bay-packers',
                  'Chicago': 'chicago-bears', 'NewEngland': 'new-england-patriots', 'Pittsburgh': 'pittsburgh-steelers',
                  'Houston': 'houston-texans', 'Denver': 'denver-broncos', 'SanFrancisco': 'san-francisco-49ers',
                  'Minnesota': 'minnesota-vikings', 'Washington': 'washington-redskins',
                  'Jacksonville': 'jacksonville-jaguars', 'Tennessee': 'tennessee-titans',
                  'Carolina': 'carolina-panthers', 'KansasCity': 'kansas-city-chiefs', 'Miami': 'miami-dolphins',
                  'Atlanta': 'atlanta-falcons', 'NewOrleans': 'new-orleans-saints', 'Baltimore': 'baltimore-ravens',
                  'Seattle': 'seattle-seahawks', 'SanDiego': 'san-diego-chargers', 'Dallas': 'dallas-cowboys',
                  'Cincinnati': 'cincinnati-bengals', 'NYJets': 'new-york-jets', 'Detroit': 'detroit-lions',
                  'Arizona': 'arizona-cardinals', 'Oakland': 'oakland-raiders', 'Indianapolis': 'indianapolis-colts',
                  'Buffalo': 'buffalo-bills', 'Cleveland': 'cleveland-browns', 'LARams': 'los-angeles-rams',
                  'LAChargers': 'los-angeles-chargers', 'washington': 'washington-football-team',
                  'LasVegas': 'las-vegas-raiders'}

    odds_df["Home"] = odds_df["Home"].apply(lambda x: teams_dict[x])
    odds_df["Away"] = odds_df["Away"].apply(lambda x: teams_dict[x])

    odds_df.to_csv("backend/data/odds/odds.csv")
