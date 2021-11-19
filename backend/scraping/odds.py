from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
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


def scrape_excel_files():
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


def scrape_vegas(current_week):
    url = "https://www.lines.com/betting/nfl/odds/moneyline"
    hdr = {
        'User-Agent': """Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) 
        Chrome/92.0.4515.159 Safari/537.36""",
        'Connection': 'close'}
    req = Request(url, headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, "html.parser")

    tags = soup.find_all("div", attrs={"class": "odds-list-panel"})
    tags = [tag.text for tag in tags]
    tags = [game.split("\n") for game in tags]
    week = []
    for game in tags:
        important_info = []
        for info in game:
            if len(info) > 0:
                important_info.append(info)
        week.append(important_info)

    home_teams = [game[2] for game in week]
    away_teams = [game[1] for game in week]
    home_odds = [game[4] for game in week]
    away_odds = [game[3] for game in week]
    df = pd.DataFrame({"Home": home_teams, "Away": away_teams, "ML_h": home_odds, "ML_a": away_odds})
    df["Week"] = current_week
    df["Year"] = 2021

    teams_dict = {"Patriots": "new-england-patriots", "Colts": "indianapolis-colts", "Falcons": "atlanta-falcons",
                  "Bills": "buffalo-bills", "Ravens": "baltimore-ravens", "Bears": "chicago-bears",
                  "Lions": "detroit-lions", "Browns": "cleveland-browns", "Texans": "houston-texans",
                  "Titans": "tennessee-titans", "Packers": "green-bay-packers", "Vikings": "minnesota-vikings",
                  "Dolphins": "miami-dolphins", "Jets": "new-york-jets", "Saints": "new-orleans-saints",
                  "Eagles": "philadelphia-eagles", "Washington": "washington-football-team",
                  "Panthers": "carolina-panthers", "49ers": "san-francisco-49ers", "Jaguars": "jacksonville-jaguars",
                  "Bengals": "cincinnati-bengals", "Raiders": "las-vegas-raiders", "Cowboys": "dallas-cowboys",
                  "Chiefs": "kansas-city-chiefs", "Cardinals": "arizona-cardinals", "Seahawks": "seattle-seahawks",
                  "Steelers": "pittsburgh-steelers", "Chargers": "los-angeles-chargers", "Giants": "new-york-giants",
                  "Buccaneers": "tampa-bay-buccaneers", "Rams": "los-angeles-rams", "Denver": "denver-broncos"}

    df["Home"] = df["Home"].apply(lambda x: teams_dict[x])
    df["Away"] = df["Away"].apply(lambda x: teams_dict[x])

    df.to_csv("backend/data/odds/2021/Week_" + str(current_week) + ".csv")


if __name__ == '__main__':
    scrape_vegas(11)
