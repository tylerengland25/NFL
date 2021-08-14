import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen


def scrape_schedule(team, year):

    url = "https://www.footballdb.com/teams/nfl/"+team+"/results/"+year
    hdr = {'User-Agent': 'Mozilla/5.0', 'Connection': 'close'}
    req = Request(url, headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, "html.parser")

    dates = []
    home_teams, away_teams = [], []
    Q1_scores_A, Q2_scores_A, Q3_scores_A, Q4_scores_A = [], [], [], []
    Q1_scores_H, Q2_scores_H, Q3_scores_H, Q4_scores_H = [], [], [], []
    away_final_scores, home_final_scores = [], []

    team_ids = {'Arizona': 'T1', 'Atlanta': 'T2', 'Baltimore': 'T3', 'Buffalo': 'T4', 'Carolina': 'T5', 'Chicago': 'T6',
                'Cincinnati': 'T7', 'Cleveland': 'T8', 'Dallas': 'T9', 'Denver': 'T10', 'Detroit': 'T11',
                'Green Bay': 'T12', 'Houston': 'T13', 'Indianapolis': 'T14', 'Jacksonville': 'T15',
                'Kansas City': 'T16', 'Miami': 'T17', 'Minnesota': 'T18', 'New England': 'T19', 'New Orleans': 'T20',
                'NY Giants': 'T21', 'NY Jets': 'T22', 'Las Vegas': 'T23', 'Philadelphia': 'T24', 'Pittsburgh': 'T25',
                'LA Chargers': 'T26', 'LA Rams': 'T29', 'San Francisco': 'T27', 'Seattle': 'T28', 'Tampa Bay': 'T30',
                'Tennessee': 'T31', 'Washington': 'T32', "Oakland": "T23", "San Diego": "T26", "Los Angeles": "T29"}

    team_id = {'arizona cardinals': 'T1', 'atlanta falcons': 'T2', 'baltimore ravens': 'T3', 'buffalo bills': 'T4',
               'carolina panthers': 'T5', 'chicago bears': 'T6', 'cincinnati bengals': 'T7', 'cleveland browns': 'T8',
               'dallas cowboys': 'T9', 'denver broncos': 'T10', 'detroit lions': 'T11', 'green bay packers': 'T12',
               'houston texans': 'T13', 'indianapolis colts': 'T14', 'jacksonville jaguars': 'T15',
               'kansas city chiefs': 'T16', 'miami dolphins': 'T17', 'minnesota vikings': 'T18',
               'new england patriots': 'T19', 'new orleans saints': 'T20', 'new york giants': 'T21',
               'new york jets': 'T22', 'las vegas raiders': 'T23', 'philadelphia eagles': 'T24',
               'pittsburgh steelers': 'T25', 'los angeles chargers': 'T26', 'san francisco 49ers': 'T27',
               'seattle seahawks': 'T28', 'los angeles rams': 'T29', 'tampa bay buccaneers': 'T30',
               'tennessee titans': 'T31', 'washington football team': 'T32', 'oakland raiders': 'T23',
               'washington redskins': 'T32', 'san diego chargers': 'T26'}

    season_id = {"2020": "S1", "2019": "S2", "2018": "S3", "2017": "S4", "2016": "S5"}

    for game in soup.find_all("div", attrs={"class": "fb_component"}):
        # scrape date
        date = game.find("table").find("th", attrs= {"class": "left"}).text
        if len(date.split()) > 2:
            continue
        dates.append(date)
        # scrape home and away team stats
        away_row = game.find("table").find("tr", attrs={"class": "row-visitor center"}).find_all("td")
        home_row = game.find("table").find("tr", attrs={"class": "row-home center"}).find_all("td")

        away_team = away_row[0].a.text
        away_teams.append(team_ids[away_team])

        home_team = home_row[0].a.text
        home_teams.append(team_ids[home_team])

        try:
            Q1_score_A = away_row[1].text
            Q1_score_H = home_row[1].text
            Q2_score_A = away_row[2].text
            Q2_score_H = home_row[2].text
            Q3_score_A = away_row[3].text
            Q3_score_H = home_row[3].text
            Q4_score_A = away_row[4].text
            Q4_score_H = home_row[4].text
            away_final_score = away_row[5].text
            home_final_score = home_row[5].text
        except IndexError:
            Q1_score_A = 0
            Q1_score_H = 0
            Q2_score_A = 0
            Q2_score_H = 0
            Q3_score_A = 0
            Q3_score_H = 0
            Q4_score_A = 0
            Q4_score_H = 0
            away_final_score = 0
            home_final_score = 0

        Q1_scores_A.append(Q1_score_A)
        Q2_scores_A.append(Q2_score_A)
        Q3_scores_A.append(Q3_score_A)
        Q4_scores_A.append(Q4_score_A)
        Q1_scores_H.append(Q1_score_H)
        Q2_scores_H.append(Q2_score_H)
        Q3_scores_H.append(Q3_score_H)
        Q4_scores_H.append(Q4_score_H)
        away_final_scores.append(away_final_score)
        home_final_scores.append(home_final_score)

    df = pd.DataFrame({"Date": dates,
                       "Home": home_teams,
                       "Away": away_teams,
                       "H_Q1": Q1_scores_H,
                       "H_Q2": Q2_scores_H,
                       "H_Q3": Q2_scores_H,
                       "H_Q4": Q4_scores_H,
                       "A_Q1": Q1_scores_A,
                       "A_Q2": Q2_scores_A,
                       "A_Q3": Q3_scores_A,
                       "A_Q4": Q4_scores_A,
                       "H_final": home_final_scores,
                       "A_final": away_final_scores})

    df["SId"] = season_id[year]
    df["Year"] = year
    df["TId"] = team_id[" ".join(team.split('-'))]

    return df


def generate_tuples():
    teams = ["arizona-cardinals", "atlanta-falcons", "carolina-panthers", "chicago-bears", "dallas-cowboys",
             "detroit-lions", "green-bay-packers", "los-angeles-rams", "minnesota-vikings", "new-orleans-saints",
             "new-york-giants", "philadelphia-eagles", "san-francisco-49ers", "seattle-seahawks",
             "tampa-bay-buccaneers", "washington-redskins", "baltimore-ravens", "buffalo-bills",
             "cincinnati-bengals", "cleveland-browns", "denver-broncos", "houston-texans", "indianapolis-colts",
             "jacksonville-jaguars", "kansas-city-chiefs", "oakland-raiders", "los-angeles-chargers",
             "miami-dolphins", "new-england-patriots", "new-york-jets", "pittsburgh-steelers", "tennessee-titans"]

    years = ["2020", "2019", "2018", "2017", "2016"]

    tuples = []

    for year in years:
        for team in teams:
            if year == "2020" and team == "oakland-raiders":
                pair = ("las-vegas-raiders", year)
            elif year == "2016" and team == "los-angeles-chargers":
                pair = ("san-diego-chargers", year)
            elif year == "2020" and team == "washington-redskins":
                pair = ("washington-football-team", year)
            else:
                pair = (team, year)

            tuples.append(pair)

    return tuples


def scrape_all_schedules():
    team_years = generate_tuples()
    schedules = pd.DataFrame()

    for team_year in team_years:
        team = team_year[0]
        year = team_year[1]
        schedule = scrape_schedule(team, year)
        schedules = schedules.append(schedule)

    return schedules


def main():
    df = scrape_all_schedules()

    print(df)


if __name__ == '__main__':
    main()
