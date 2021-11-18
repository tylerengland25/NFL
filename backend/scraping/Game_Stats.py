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

    stats = {"Week": week, "Home": home, "Away": visitor,
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


def scrape_season(year, num_weeks):
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
    for week in range(10, num_weeks):
        print(week)
        if week == 10:
            print("debug")
        games = weeks[week].find_all("tr")
        for game in games:
            link = game.find_all("td")[6].a
            link = game.find_all("td")[6].a["href"]
            away = link.split('/')[-1].split('vs')[0][:-1]
            home = "-".join(link.split('/')[-1].split('vs')[1].split('-')[:-1])[1:]
            number = link.split('/')[-1].split('vs')[1].split('-')[-1]
            game_stats = scrape_game(home, away, number, week + 1)
            game_stats["H_Score"] = game.find_all("td")[4].text
            game_stats["A_Score"] = game.find_all("td")[2].text
            game_stats["Year"] = year
            season_stats = season_stats.append(game_stats, ignore_index=True)

    return season_stats


def convert_poss(time):
    """
    Assumes time is minutes and seconds separated by :
    :param time: String "MM:SS"
    :return: Integer of seconds of possession
    """
    min_sec = time.split(":")
    mins = int(min_sec[0])
    secs = int(min_sec[1])
    return (mins * 60) + secs


def last_5_games(filename1, filename2):
    """
    Selects last 5 games of stats for each game
    :return:
    """
    # Load data
    home_df = pd.read_csv(filename1)
    away_df = pd.read_csv(filename1)

    home_df.rename(
        columns=lambda col: col.lower()[2:]
        if "H_" in col or len(col.split('_')) == 1
        else col.lower()[2:] + "_def",
        inplace=True
    )
    home_df.rename(columns={"ek": "week", "ar": "year", "me": "team", "ay": "opponent"}, inplace=True)
    home_df["home_away"] = 1

    away_df.rename(
        columns=lambda col: col.lower()[2:]
        if "A_" in col or len(col.split('_')) == 1
        else col.lower()[2:] + "_def",
        inplace=True
    )
    away_df.rename(columns={"ek": "week", "ar": "year", "ay": "team", "me": "opponent"}, inplace=True)
    away_df["home_away"] = 0

    # merge home and away team data into one dataframe
    teams_df = pd.concat([home_df, away_df], ignore_index=True, sort=True)
    season_length = teams_df.groupby(["team", "year"])["week"].max().reset_index()
    season_length.rename(columns={"week": "season_length"}, inplace=True)
    teams_df = pd.merge(teams_df, season_length,
                        how="left",
                        on=["team", "year"])

    # Feature engineering
    teams_df["cmp_pct"] = teams_df["cmp"] / teams_df["att"]
    teams_df["cmp_pct_def"] = teams_df["cmp_def"] / teams_df["att_def"]
    teams_df["3rd_pct"] = teams_df["3rd_cmp"] / teams_df["3rd_att"]
    teams_df["3rd_pct_def"] = teams_df["3rd_cmp_def"] / teams_df["3rd_att_def"]
    teams_df["4th_pct"] = teams_df["4th_cmp"] / teams_df["4th_att"]
    teams_df["4th_pct_def"] = teams_df["4th_cmp_def"] / teams_df["4th_att_def"]
    teams_df["fg_pct"] = teams_df["fg_cmp"] / teams_df["fg_att"]
    teams_df["yds_per_rush"] = teams_df["rush_yds"] / teams_df["rush_ply"]
    teams_df["yds_per_rush_def"] = teams_df["rush_yds_def"] / teams_df["rush_ply_def"]
    teams_df["yds_per_att"] = teams_df["pass_yds"] / teams_df["att"]
    teams_df["yds_per_att_def"] = teams_df["pass_yds_def"] / teams_df["att_def"]
    teams_df["yds_per_ply"] = teams_df["total_y"] / teams_df["total_ply"]
    teams_df["yds_per_ply_def"] = teams_df["total_y_def"] / teams_df["total_ply_def"]
    teams_df["pts_per_ply"] = teams_df["score"] / teams_df["total_ply"]
    teams_df["pts_per_ply_def"] = teams_df["score_def"] / teams_df["total_ply_def"]
    teams_df["punts_per_ply"] = teams_df["punts"] / teams_df["total_ply"]
    teams_df["punts_per_ply_def"] = teams_df["punts_def"] / teams_df["total_ply"]
    teams_df["pts_per_yd"] = teams_df["score"] / teams_df["total_y"]
    teams_df["pts_per_yd_def"] = teams_df["score_def"] / teams_df["total_y_def"]
    teams_df = teams_df.fillna(0)

    # dataframe for last 5 games
    last_five = teams_df.copy()
    for i in range(1, 6):
        week_i = teams_df.copy()
        week_i["week"] = week_i["week"] + i
        week_i.loc[week_i["week"] > week_i["season_length"], "year"] = \
            week_i.loc[week_i["week"] > week_i["season_length"], "year"] + 1
        week_i.loc[week_i["week"] > week_i["season_length"], "week"] = \
            week_i.loc[week_i["week"] > week_i["season_length"], "week"] \
            - week_i.loc[week_i["week"] > week_i["season_length"], "season_length"]

        last_five = pd.merge(last_five, week_i,
                             how="left",
                             on=["team", "week", "year"],
                             suffixes=["", "_" + str(i)])

    last_five.fillna(0, inplace=True)
    last_five.to_csv(filename2)


def main():
    ten_years = pd.DataFrame()
    for season in range(2010, 2021):
        print(season)
        season_stats = scrape_season(str(season), 21)
        ten_years = pd.concat([ten_years, season_stats], ignore_index=True)

    ten_years.to_csv("backend/data/weekly_stats.csv")
    last_5_games("backend/data/weekly_stats.csv", "backend/data/last_five.csv")


def current_season(current):
    season = scrape_season(current, 11)
    past_seasons = pd.read_csv("backend/data/weekly_stats.csv")
    df = pd.concat([past_seasons, season], ignore_index=True)
    df.to_csv("backend/data/current_season_stats.csv")
    last_5_games("backend/data/current_season_stats.csv", "backend/data/current_last_five_games.csv")


if __name__ == '__main__':
    current_season("2021")
