import pandas as pd
from bs4 import BeautifulSoup, Comment
from urllib.request import urlopen


def scrape_game(game_info, dfs):
    """
    Function: 
        Scrapes schdeule
        Writes following DataFrames to CSV files:
            ~ scores

    Input:
        game_info: dict(
            str: str, 
            str: str, 
            str: str, 
            str: str,
            str: str, 
            str: str
        )
        dfs: dict(str: DataFrame)

    Output:
        None
    """
    # Connect
    url = f"https://www.pro-football-reference.com{game_info['href']}"
    html = urlopen(url)
    soup = BeautifulSoup(html, features="lxml")

    # Set tables
    date = soup.find('div', attrs={'class': 'scorebox_meta'}).find('div').text

    game_info['date'] = date

    # Print matchup tp track progress
    print(f"\t\t{game_info['away']} @ {game_info['home']}, {game_info['date'].strip()}")

    dfs['schedule'] = dfs['schedule'].append(game_info, ignore_index=True)


def scrape_week(href, season, dfs):
    """
    Function: 
        Scrapes week schedule.
        Writes following DataFrames to CSV files:
            ~ schedule

    Input:
        href: str
        season: int
        dfs: dict(str: DataFrame)

    Output:
        None
    """
    # Print week to track progress
    week = href.split('/')[-1].split('_')[-1].split('.')[0]
    print(f'\tWeek: {week}')
    
    # Connect
    url = f'https://www.pro-football-reference.com{href}'
    html = urlopen(url)
    soup = BeautifulSoup(html, features="lxml")

    # Iterate over each game
    games = soup.find('div', attrs={'class': 'game_summaries'}).find_all('div')
    for game in games:
        game_info = {
            'week': week,

            'season': season,

            'away': game.find(
                'table', 
                attrs={'class': 'teams'}
            ).find_all('tr')[1].find('td').text,

            'home': game.find(
                'table', 
                attrs={'class': 'teams'}
            ).find_all('tr')[2].find('td').text,

            'href': game.find(
                'table', 
                attrs={'class': 'teams'}
            ).find_all('tr')[1].find(
                'td', 
                attrs={'class': 'right gamelink'}
            ).find('a')['href']
        }

        scrape_game(game_info, dfs)


def scrape_season(season, dfs):
    """
    Function: 
        Scrapes season schedules since 2010.
        Writes following DataFrames to CSV files:
            ~ schedule

    Input:
        season: int
        dfs: dict(str: DataFrame)

    Output:
        None
    """
    # Print statement to track progress
    print(f'Season: {season}')

    # Connect
    url = f'https://www.pro-football-reference.com/years/{season}/'
    html = urlopen(url)
    soup = BeautifulSoup(html, features="lxml")
    
    # Week links
    week_hrefs = {a['href'] for a in soup.find_all('a') if '/week_' in a['href']}

    # Iterate of each week
    for href in week_hrefs:
        scrape_week(href, season, dfs)


def main():
    """
    Function:
        Scrapes season schedules since 2010.
        Writes following DataFrames to CSV files:
            ~ schedule

    Input:
        None
        
    Output:
        None
    """
    # Initialize DataFrames
    dfs = {'schedule': pd.DataFrame()}

    # Iterate over each season
    for season in range(2010, 2023):
        scrape_season(season, dfs) # Append season to data structure

    # Write DataFrames to CSV files
    for df in dfs:
        dfs[df].to_csv(f'backend/data/games/{df}.csv', index=False)


if __name__ == '__main__':
    main()