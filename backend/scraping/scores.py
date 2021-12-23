from urllib.request import urlopen, Request
import pandas as pd
from bs4 import BeautifulSoup


def scrape_game(link):
    url = "https://www.footballdb.com" + link
    hdr = {
        'User-Agent': """Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) 
    Chrome/92.0.4515.159 Safari/537.36""",
        'Connection': 'close'}
    req = Request(url, headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, "html.parser")

    away = link.split('/')[-1].split('vs')[0][:-1]
    home = "-".join(link.split('/')[-1].split('vs')[1].split('-')[:-1])[1:]
    tables = soup.find_all("table")

    # Scrape scores of each quarter and final
    quarter_scores = tables[0]
    home_scores = quarter_scores.find_all("tr")[2].find_all("td")
    away_scores = quarter_scores.find_all("tr")[1].find_all("td")
    scores = {"home": home, "away": away,
              "q1_h": home_scores[1].text, "q1_a": away_scores[1].text,
              "q2_h": home_scores[2].text, "q2_a": away_scores[2].text,
              "q3_h": home_scores[3].text, "q3_a": away_scores[3].text,
              "q4_h": home_scores[4].text, "q4_a": away_scores[4].text,
              "final_h": home_scores[5].text, "final_a": away_scores[5].text,
              "3_consecutive_scores": 0}

    # Scrape True of False if there were three consecutive scores
    scoring_plays = tables[1].find_all("tr")[1:]
    last_score = scoring_plays[0].find("td").text
    num_scores = 1
    for score in scoring_plays[1:]:
        current_score = score.find("td").text
        if last_score == current_score:
            num_scores += 1
        elif current_score != "2nd Quarter" and current_score != "3rd Quarter" and current_score != "4th Quarter":
            num_scores = 1
            last_score = current_score

        if num_scores == 3:
            scores["3_consecutive_scores"] = 1

    return scores


def scrape_season(year):
    url = "https://www.footballdb.com/games/index.html?lg=NFL&yr=" + year
    hdr = {
        'User-Agent': """Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) 
        Chrome/92.0.4515.159 Safari/537.36""",
        'Connection': 'close'}
    req = Request(url, headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, "html.parser")

    consecutive_games = pd.DataFrame()

    weeks = soup.find_all("tbody")
    for week in range(len(weeks)):
        print("\t" + str(week))
        games = weeks[week].find_all("tr")
        for game in games:
            link = game.find_all("td")[6]
            if link is None or link.text == "--":
                pass
            else:
                link = link.a["href"]
                scores = scrape_game(link)
                scores["year"] = int(year)
                scores["week"] = week + 1
                consecutive_games = consecutive_games.append(scores, ignore_index=True)

    return consecutive_games


def update_scoring():
    df = pd.read_csv("backend/data/scoring.csv")
    df = df.drop(["Unnamed: 0"], axis=1)
    scoring_df = scrape_season(str(2021))
    df = df.append(scoring_df, ignore_index=True)
    df = df.drop_duplicates()
    df.to_csv("backend/data/scoring.csv")


def main():
    df = pd.DataFrame()
    for year in range(2010, 2022):
        print(year)
        scoring_df = scrape_season(str(year))
        df = df.append(scoring_df, ignore_index=True)

    df.to_csv("backend/data/scoring.csv")


if __name__ == '__main__':
    main()
