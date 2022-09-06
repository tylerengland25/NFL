import pandas as pd
from bs4 import BeautifulSoup, Comment
from urllib.request import urlopen


def scrape_scores(scores, three_straight, game_info, dfs):
    """
    Function: 
        Scrape scores

    Input:
        scores: tag
        three_straight: tag
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
    df = {
        'date': [game_info['date'], game_info['date']],
        'week': [game_info['week'], game_info['week']],
        'season': [game_info['season'], game_info['season']],
        'team': [game_info['away'], game_info['home']],
        'opponent': [game_info['home'], game_info['away']],
        'home_field': [False, True]
    }
    for quarter in scores.find_all('th')[2:]:
        df[quarter.text] = []
    for row in scores.find('tbody').find_all('tr'):
        for score in zip(scores.find_all('th')[2:], row.find_all('td')[2:]):
            df[score[0].text].append(score[1].text)

    # Three straight scores
    three_straight_scores = False
    last_score = three_straight.find('tbody').find_all('td', attrs={'data-stat': 'team'})[0].text
    count = 1
    for team in three_straight.find('tbody').find_all('td', attrs={'data-stat': 'team'})[1:]:
        if team.text == last_score:
            count += 1
            last_score = team.text
            if count >= 3:
                three_straight_scores = True
        else:
            count = 1
            last_score = team.text
    
    df['3_straight'] = [three_straight_scores, three_straight_scores]
    
    dfs['scores'] = dfs['scores'].append(pd.DataFrame(df), ignore_index=True)


def scrape_game(game_info, dfs):
    """
    Function: 
        Scrapes game data since 2010. Data includes quarterly scores and if there 
        was 3 consecutive scores.
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
    # Print matchup tp track progress
    print(f"\t\t{game_info['away']} @ {game_info['home']}, {game_info['date'].strip()}")

    # Connect
    url = f"https://www.pro-football-reference.com{game_info['href']}"
    html = urlopen(url)
    soup = BeautifulSoup(html, features="lxml")

    # Set tables
    quarter_scores = soup.find('table', attrs={'class': 'linescore nohover stats_table no_freeze'})
    three_straight = soup.find('table', attrs={'class': 'stats_table', 'id': 'scoring'})

    # Look at comments as well
    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment_soup = BeautifulSoup(comment, 'html.parser')
        if quarter_scores is None: # Scores
            quarter_scores = comment_soup.find('table', attrs={'class': 'linescore nohover stats_table no_freeze'})
        if three_straight is None: # Three Consectutive Scores
            three_straight = comment_soup.find('table', attrs={'class': 'stats_table', 'id': 'scoring'})
    
    # Scrape data for each table
    scrape_scores(quarter_scores, three_straight, game_info, dfs)


def scrape_week(href, season, dfs):
    """
    Function: 
        Scrapes game data since 2010. Data includes quarterly scores and if there 
        was 3 consecutive scores.
        Writes following DataFrames to CSV files:
            ~ scores

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
    games = soup.find_all('div', attrs={'class': 'game_summary expanded nohover'})
    for game in games:
        game_info = {
            'date': game.find(
                'table',
                attrs={'class': 'teams'}
            ).find_all('tr')[0].find('td').text.strip('\n'),

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
        Scrapes game data since 2010. Data includes quarterly scores and if there 
        was 3 consecutive scores.
        Writes following DataFrames to CSV files:
            ~ scores

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
        Scrapes game data since 2010. Data includes quarterly scores and if there 
        was 3 consecutive scores.
        Writes following DataFrames to CSV files:
            ~ scores

    Input:
        None
        
    Output:
        None
    """
    # Initialize DataFrames
    dfs = {'scores': pd.DataFrame()}

    # Iterate over each season
    for season in range(2010, 2022):
        scrape_season(season, dfs) # Append season to data structure

    # Write DataFrames to CSV files
    for df in dfs:
        dfs[df].to_csv(f'backend/data/games/{df}.csv', index=False)


if __name__ == '__main__':
    main()