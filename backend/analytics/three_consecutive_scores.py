import pandas as pd
from backend.scraping.scores import update_scoring
from backend.scraping.odds import scrape_vegas


def spreads(col1, col2):
    if col1 == "pk":
        col1 = 0
    if col2 == "pk":
        col2 = 0
    if col1 < col2:
        return col1
    else:
        return col2


def totals(col1, col2):
    if col1 == "pk":
        col1 = 0
    if col2 == "pk":
        col2 = 0
    if col1 < col2:
        return col2
    else:
        return col1


def odds_season(filename):
    df = pd.read_excel(filename)
    df["Year"] = filename.split("-")[0][-4:]

    home_df = df[df["VH"] == "H"].reset_index()[["Team", "ML", "Close", "Year", "Week"]]
    home_df.rename(columns={"Team": "Home"}, inplace=True)
    away_df = df[df["VH"] == "V"].reset_index()[["Team", "ML", "Close", "Year", "Week"]]
    away_df.rename(columns={"Team": "Away"}, inplace=True)
    df = home_df.join(away_df, lsuffix="_h", rsuffix="_a")

    df["Spread"] = df.apply(lambda x: spreads(x.Close_h, x.Close_a), axis=1)
    df["Total"] = df.apply(lambda x: totals(x.Close_h, x.Close_a), axis=1)
    df = df.drop(["Close_h", "Close_a"], axis=1)

    return df


def spread_totals():
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

    # Current season
    current_season_odds = pd.read_excel("backend/data/odds/nfl odds 2021-22.xlsx")
    current_season_odds = current_season_odds.drop(["Unnamed: 0"], axis=1)
    odds_df = odds_df.append(current_season_odds, ignore_index=True)

    return odds_df


def round_totals(number, base):
    return base * round(number/base)


def round_spreads(number):
    if number <= 1.5:
        return 0
    elif number < 5:
        return 3
    elif number < 8:
        return 7
    elif number < 12:
        return 10
    elif number < 17:
        return 14
    else:
        return 17


def pivot_table():
    # Load data
    df = pd.read_csv("backend/data/scoring.csv")
    odds = spread_totals()

    # Join data
    odds["Year"] = odds["Year"].astype(int)
    df = pd.merge(df, odds, left_on=["home", "away", "week", "year"], right_on=["Home", "Away", "Week", "Year"])

    # Round spreads and totals
    df["Spread"] = df["Spread"].apply(lambda x: round_spreads(x))
    df["Total"] = df["Total"].apply(lambda x: round_totals(x, 2))

    # Create pivot table for all games
    table = df.groupby(["Spread", "Total"]).agg({"3_consecutive_scores": ["sum", "count"]})
    table["perc"] = table[("3_consecutive_scores", "sum")] / table[("3_consecutive_scores", "count")]
    table["perc"] = table["perc"].apply(lambda x: 100 * round(x, 2))
    table = table.reset_index()
    table.columns = ["spread", "total", "sum", "count", "perc"]

    # Create pivot table for last 5 years of games
    five_years = df[df["Year"] > 2015]
    last_five = five_years.groupby(["Spread", "Total"]).agg({"3_consecutive_scores": ["sum", "count"]})
    last_five["perc"] = last_five[("3_consecutive_scores", "sum")] / last_five[("3_consecutive_scores", "count")]
    last_five["perc"] = last_five["perc"].apply(lambda x: 100 * round(x, 2))
    last_five = last_five.reset_index()
    last_five.columns = ["spread", "total", "sum", "count", "perc"]

    # Create pivot table for current season
    current_season = df[df["Year"] == 2021]
    current = current_season.groupby(["Spread", "Total"]).agg({"3_consecutive_scores": ["sum", "count"]})
    current["perc"] = current[("3_consecutive_scores", "sum")] / current[("3_consecutive_scores", "count")]
    current["perc"] = current["perc"].apply(lambda x: 100 * round(x, 2))
    current = current.reset_index()
    current.columns = ["spread", "total", "sum", "count", "perc"]

    with pd.ExcelWriter("backend/data/analytics/3_consecutive_scores.xlsx") as writer:
        perc_table = pd.pivot_table(table, values="perc", index="total", columns="spread")
        perc_table = perc_table.fillna(0)
        perc_table.to_excel(writer, sheet_name="10 Year Percentages")

        count_table = pd.pivot_table(table, values="count", index="total", columns="spread")
        count_table = count_table.fillna(0)
        count_table.to_excel(writer, sheet_name="10 Year Counts")

        perc_table = pd.pivot_table(last_five, values="perc", index="total", columns="spread")
        perc_table = perc_table.fillna(0)
        perc_table.to_excel(writer, sheet_name="5 Year Percentages")

        count_table = pd.pivot_table(last_five, values="count", index="total", columns="spread")
        count_table = count_table.fillna(0)
        count_table.to_excel(writer, sheet_name="5 Year Counts")

        perc_table = pd.pivot_table(current, values="perc", index="total", columns="spread")
        perc_table = perc_table.fillna(0)
        perc_table.to_excel(writer, sheet_name="Current Season Percentages")

        count_table = pd.pivot_table(current, values="count", index="total", columns="spread")
        count_table = count_table.fillna(0)
        count_table.to_excel(writer, sheet_name="Current Season Counts")


def main():
    update_scoring(2021, 22)
    pivot_table()


if __name__ == '__main__':
    main()
