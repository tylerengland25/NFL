import pandas as pd
from bs4 import BeautifulSoup
import requests


def scrape_game(home, visitor, number):
    """
    Scrapes statistics for given game
    :param home: home team separated by - (e.g. kansas-city-chiefs)
    :param visitor: away team separated by - (e.g. houston-texans)
    :param number: unique number associated with date of game
    :return:
    """
    url = "https://www.footballdb.com/games/boxscore/" \
          + visitor + "-vs-" + home + "-" + number
    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")

    


def main():
    visitor = "houston-texans"
    home = "kansas-city-chiefs"
    number = "2020091001"
    scrape_game(home, visitor, number)


if __name__ == '__main__':
    main()
