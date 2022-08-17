from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import pandas as pd


def odds_season(filename):
    """
    Function: Process excel file

    Input: 
        filename: str

    Output:
        df: DataFrame
    """
    df = pd.read_excel(filename)
    year = int(filename.split(' ')[-1].split('-')[0])

    if year < 2021:

        home_df = df[df["VH"] == "H"].reset_index()[["Team", "ML", "Week"]]
        home_df.rename(
            columns={
                "Team": "home", 'ML': 'ml', 'Week': 'week'
            }, 
            inplace=True
        )

        away_df = df[df["VH"] == "V"].reset_index()[["Team", "ML"]]
        away_df.rename(
            columns={
                "Team": "away", 'ML': 'ml'
            }, 
            inplace=True
        )

        df = pd.merge(
            home_df,
            away_df,
            left_index=True,
            right_index=True,
            suffixes=('_h', '_a')
        )

        df["year"] = year

    return df


def scrape_excel_files():
    """
    Function: Process odds for past seasons

    Input: 
        None

    Output: 
        None
    """
    files = [
        "backend/data/odds/nfl odds 2010-11.xlsx", "backend/data/odds/nfl odds 2011-12.xlsx",
        "backend/data/odds/nfl odds 2012-13.xlsx", "backend/data/odds/nfl odds 2013-14.xlsx",
        "backend/data/odds/nfl odds 2014-15.xlsx", "backend/data/odds/nfl odds 2015-16.xlsx",
        "backend/data/odds/nfl odds 2016-17.xlsx", "backend/data/odds/nfl odds 2017-18.xlsx",
        "backend/data/odds/nfl odds 2018-19.xlsx", "backend/data/odds/nfl odds 2019-20.xlsx",
        "backend/data/odds/nfl odds 2020-21.xlsx", "backend/data/odds/nfl odds 2021-22.xlsx"
    ]

    odds_df = pd.DataFrame()
    for file in files:
        odds_df = odds_df.append(odds_season(file))

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

    odds_df["home"] = odds_df["home"].apply(lambda x: teams_dict.get(x, x))
    odds_df["away"] = odds_df["away"].apply(lambda x: teams_dict.get(x, x))

    odds_df.to_csv("backend/data/odds/odds.csv", index=False)


def scrape_vegas(current_week, current_season):
    """
    Function:
        Scrapes current weeks odds and adds them to the current season odds excel sheet

    Input: 
        current_week: int
        current_season: str

    Output:
        None 
    """

    # Link
    url = "https://www.lines.com/betting/nfl/odds"

    # Rrequest html
    hdr = {
        'User-Agent': """Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) 
        Chrome/92.0.4515.159 Safari/537.36""",
        'Connection': 'close'
    }
    
    # Moneyline page
    req = Request(url, headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, "html.parser")

    # Dataframe variables
    home_teams = []
    away_teams = []
    home_odds = []
    away_odds = []
    spreads = []
    totals = []

    # Scraper
    games = soup.find_all("div", attrs={"class": "odds-list-panel"})
    for game in games:
        # Home team
        home_teams.append(
            game.find_all(
                'div',
                attrs={
                    'class': 'odds-list-team'
                }
            )[1]['title']
        )

        # Away team
        away_teams.append(
            game.find_all(
                'div',
                attrs={
                    'class': 'odds-list-team'
                }
            )[0]['title']
        )

        # Moneyline
        odds = game.find_all(
            'div',
            attrs={
                'class': 'odds-list-panel-col'
            }
        )[2]

        home_odds.append(
            int(
                odds.find_all(
                    'div', 
                    attrs={
                        'class': 'odds-list-val'
                        }
                )[1].text
            )
        )

        away_odds.append(
            int(
                odds.find_all(
                    'div',
                    attrs={
                        'class': 'odds-list-val'
                        }
                )[0].text
            )
        )
        
        # Spread
        spreads.append(
            abs(
                float(
                    game.find_all(
                        'div',
                        attrs={
                            'class': 'odds-list-panel-col'
                            }
                    )[0].find(
                        'div',
                        attrs={
                            'class': 'odds-list-val'
                        }
                    ).text.split(
                        ' '
                    )[0]
                )
            )
        )

        # Totals
        totals.append(
            float(
                game.find_all(
                    'div', 
                    attrs={
                        'class': 'odds-list-panel-col'
                    }
                )[1].find(
                    'div',
                    attrs={
                        'class': 'odds-list-val'
                    }
                ).text.split(
                    ' ',
                )[0][1:]
            )
        )

    # Create dataframe
    df = pd.DataFrame(
        {
            "home": home_teams, "away": away_teams,
            "ml_h": home_odds, "ml_a": away_odds,
            "spread": spreads, "total": totals  
        }
    )
    df["week"] = current_week
    df["year"] = int(current_season.split('-')[0])

    # Change team names
    team_names = {
        'Saints': 'new-orleans-saints', 'Dolphins': 'miami-dolphins', 'Cowboys': 'dallas-cowboys',
        'Washington': 'washington-football-team', 'Raiders': 'las-vegas-raiders', 'Broncos': 'denver-broncos',
        'Chiefs': 'kansas-city-chiefs', 'Steelers': 'pittsburgh-steelers', 'Seahawks': 'seattle-seahawks',
        'Bears': 'chicago-bears', 'Texans': 'houston-texans', 'Chargers': 'los-angeles-chargers',
        'Panthers': 'carolina-panthers', 'Buccaneers': 'tampa-bay-buccaneers',
        'Eagles': 'philadelphia-eagles', 'Giants': 'new-york-giants', 'Jets': 'new-york-jets',
        'Jaguars': 'jacksonville-jaguars', 'Patriots': 'new-england-patriots', 'Bills': 'buffalo-bills',
        'Vikings': 'minnesota-vikings', 'Rams': 'los-angeles-rams', 'Bengals': 'cincinnati-bengals',
        'Ravens': 'baltimore-ravens', 'Falcons': 'atlanta-falcons', 'Lions': 'detroit-lions',
        'Cardinals': 'arizona-cardinals', 'Colts': 'indianapolis-colts', 'Packers': 'green-bay-packers',
        'Browns': 'cleveland-browns', 'Titans': 'tennessee-titans', '49ers': 'san-francisco-49ers'
    }
    df["home"] = df["home"].apply(lambda x: team_names[x])
    df["away"] = df["away"].apply(lambda x: team_names[x])

    odds = pd.read_excel(f"backend/data/odds/nfl odds {current_season}.xlsx")
    odds = odds[odds["week"] != current_week]
    odds = odds.append(df, ignore_index=True)
    odds.to_excel(f"backend/data/odds/nfl odds {current_season}.xlsx", index=False)


if __name__ == '__main__':
    # scrape_excel_files()
    scrape_vegas(1, '2022-23')
