from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import pandas as pd


def scrape_game(home, visitor, number, week):
    """
    Scrapes statistics for given game
    :param home: home team separated by - (e.g. kansas-city-chiefs)
    :param visitor: away team separated by - (e.g. houston-texans)
    :param number: unique number associated with date of game
    :param week: week number
    :return: dictionary with game stats
    """
    url = "https://www.footballdb.com/games/boxscore/" \
          + visitor + "-vs-" + home + "-" + number
    hdr = {
        'User-Agent': """Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) 
    Chrome/92.0.4515.159 Safari/537.36""",
        'Connection': 'close'}
    req = Request(url, headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, "html.parser")

    team_stats = soup.find("div", attrs={"id": "divBox_team",
                                         "class": "hidden-xs"})
    tables = team_stats.find_all("tbody")

    stats = {"Year": number[:4], "Week": week, "Home": home, "Away": visitor,
             "H_1st": tables[0].find_all("td")[2].text, "A_1st": tables[0].find_all("td")[1].text,
             "H_R_1st": tables[0].find_all("td")[5].text, "A_R_1st": tables[0].find_all("td")[4].text,
             "H_P_1st": tables[0].find_all("td")[8].text, "A_P_1st": tables[0].find_all("td")[7].text,
             "H_Total_Y": tables[0].find_all("td")[14].text, "A_Total_Y": tables[0].find_all("td")[13].text,
             "H_Rush_Yds": tables[0].find_all("td")[17].text, "A_Rush_Yds": tables[0].find_all("td")[16].text,
             "H_Rush_Ply": tables[0].find_all("td")[20].text, "A_Rush_Ply": tables[0].find_all("td")[19].text,
             "H_Pass_Yds": tables[0].find_all("td")[35].text, "A_Pass_Yds": tables[0].find_all("td")[34].text,
             "H_Att": tables[0].find_all("td")[29].text.split('-')[0],
             "A_Att": tables[0].find_all("td")[28].text.split('-')[0],
             "H_Cmp": tables[0].find_all("td")[29].text.split('-')[1],
             "A_Cmp": tables[0].find_all("td")[28].text.split('-')[1],
             "H_Int": tables[0].find_all("td")[29].text.split('-')[2],
             "A_Int": tables[0].find_all("td")[28].text.split('-')[2],
             "H_Sacks": tables[0].find_all("td")[31].text.split('-')[0],
             "A_Sacks": tables[0].find_all("td")[32].text.split('-')[0],
             "H_Sack_Yds": tables[0].find_all("td")[31].text.split('-')[1],
             "A_Sack_Yds": tables[0].find_all("td")[32].text.split('-')[1],
             "H_Punts": tables[1].find_all("td")[2].text.split('-')[0],
             "A_Punts": tables[1].find_all("td")[1].text.split('-')[0],
             "H_Punt_Yds": tables[1].find_all("td")[2].text.split('-')[1],
             "A_Punt_Yds": tables[1].find_all("td")[1].text.split('-')[1],
             "H_Punt_Ret_Yds": tables[1].find_all("td")[8].text.split('-')[1],
             "A_Punt_Ret_Yds": tables[1].find_all("td")[7].text.split('-')[1],
             "H_Kick_Ret_Yds": tables[1].find_all("td")[11].text.split('-')[1],
             "A_Kick_Ret_Yds": tables[1].find_all("td")[10].text.split('-')[1],
             "H_Int_Yds": tables[1].find_all("td")[14].text.split('-')[1],
             "A_Int_Yds": tables[1].find_all("td")[13].text.split('-')[1],
             "H_Pen_Yds": tables[1].find_all("td")[17].text.split('-')[1],
             "A_Pen_Yds": tables[1].find_all("td")[16].text.split('-')[1],
             "H_Fum": tables[1].find_all("td")[19].text.split('-')[1],
             "A_Fum": tables[1].find_all("td")[20].text.split('-')[1],
             "H_Fg_Att": tables[1].find_all("td")[23].text.split('-')[1],
             "A_Fg_Att": tables[1].find_all("td")[22].text.split('-')[1],
             "H_Fg_Cmp": tables[1].find_all("td")[23].text.split('-')[0],
             "A_Fg_Cmp": tables[1].find_all("td")[22].text.split('-')[0],
             "H_3rd_Att": tables[1].find_all("td")[26].text.split('-')[1],
             "A_3rd_Att": tables[1].find_all("td")[25].text.split('-')[1],
             "H_3rd_Cmp": tables[1].find_all("td")[26].text.split('-')[0],
             "A_3rd_Cmp": tables[1].find_all("td")[25].text.split('-')[0],
             "H_4th_Att": tables[1].find_all("td")[29].text.split('-')[1],
             "A_4th_Att": tables[1].find_all("td")[28].text.split('-')[1],
             "H_4th_Cmp": tables[1].find_all("td")[29].text.split('-')[0],
             "A_4th_Cmp": tables[1].find_all("td")[28].text.split('-')[0],
             "H_Total_Ply": tables[1].find_all("td")[32].text, "A_Total_Ply": tables[1].find_all("td")[31].text,
             "H_Poss": tables[1].find_all("td")[38].text, "A_Poss": tables[1].find_all("td")[37].text
             }

    return stats


def scrape_season(year):
    url = "https://www.footballdb.com/games/index.html?lg=NFL&yr=" + year
    hdr = {
        'User-Agent': """Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) 
        Chrome/92.0.4515.159 Safari/537.36""",
        'Connection': 'close'}
    req = Request(url, headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, "html.parser")

    season_stats = pd.DataFrame()

    weeks = soup.find_all("tbody")
    for week in range(len(weeks)):
        games = weeks[week].find_all("tr")
        for game in games:
            link = game.find_all("td")[6].a["href"]
            away = link.split('/')[-1].split('vs')[0][:-1]
            home = "-".join(link.split('/')[-1].split('vs')[1].split('-')[:-1])[1:]
            number = link.split('/')[-1].split('vs')[1].split('-')[-1]
            game_stats = scrape_game(home, away, number, week + 1)
            game_stats["H_Score"] = game.find_all("td")[4].text
            game_stats["A_Score"] = game.find_all("td")[2].text
            season_stats = season_stats.append(game_stats, ignore_index=True)

    return season_stats


def main():
    ten_years = pd.DataFrame()
    for season in range(2010, 2021):
        season_stats = scrape_season(str(season))
        ten_years = pd.concat([ten_years, season_stats], ignore_index=True)

    ten_years.to_csv("backend/data/weekly_stats.csv")



if __name__ == '__main__':
    main()
