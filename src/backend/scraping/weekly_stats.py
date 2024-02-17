import pandas as pd
from bs4 import BeautifulSoup, Comment
from urllib.request import urlopen


def scrape_home_away_tag(tag, game_info, type, drives=False):
    """
    Function: 
        Scrape tag

    Input:
        tag: tag
        game_info: dict(
            str: str, 
            str: str, 
            str: str, 
            str: str,
            str: str,
            str: str
        )

    Output:
        df: DataFrame
    """
    labels = [th['data-stat'] for th in tag.find('thead').find_all('tr')[-1].find_all('th')]
    labels = ['date', 'week', 'season', 'team', 'opponent', 'home_field'] + labels
    df = {label: [] for label in labels}
    if drives:
        df['pass'] = []
        df['rush'] = []
        df['penalty'] = []

    team = game_info['away'] if type == 'away' else game_info['home']
    opponent = game_info['home'] if type == 'away' else game_info['away']
    home = False if type == 'away' else True
    
    for tr in tag.find('tbody').find_all('tr'):
        if tr.get('class', None) is None:
            df['date'].append(game_info['date'])
            df['week'].append(game_info['week'])
            df['season'].append(game_info['season'])
            df['team'].append(team)
            df['opponent'].append(opponent)
            df['home_field'].append(home)
            for td in tr.find_all(['th', 'td']):
                if td['data-stat'] != 'team':
                    df[td['data-stat']].append(td.text)
                if td['data-stat'] == 'play_count_tip' and drives:
                    play_counts= [span.split(' ') for span in td.find('span')['tip'].split(', ')]
                    for play_count in play_counts:
                        df[play_count[1].lower()].append(play_count[0])
                        
        elif drives:
            df['date'].append(game_info['date'])
            df['week'].append(game_info['week'])
            df['season'].append(game_info['season'])
            df['team'].append(team)
            df['opponent'].append(opponent)
            df['home_field'].append(home)
            for td in tr.find_all(['th', 'td']):
                if td['data-stat'] != 'team':
                    df[td['data-stat']].append(td.text)
                if td['data-stat'] == 'play_count_tip':
                    play_counts= [span.split(' ') for span in td.find('span')['tip'].split(', ')]
                    for play_count in play_counts:
                        df[play_count[1].lower()].append(play_count[0])

    return pd.DataFrame(df)


def scrape_tag(tag, game_info):
    """
    Function: 
        Scrape tag

    Input:
        tag: tag
        game_info: dict(
            str: str, 
            str: str, 
            str: str, 
            str: str,
            str: str,
            str: str
        )

    Output:
        df: DataFrame
    """
    if tag is None:
        return pd.DataFrame()
        
    labels = [th['data-stat'] for th in tag.find('thead').find_all('tr')[-1].find_all('th')]
    labels = ['date', 'week', 'season', 'team', 'opponent', 'home_field'] + labels
    df = {label: [] for label in labels}

    team = game_info['away']
    opponent = game_info['home']
    home = False
    for tr in tag.find('tbody').find_all('tr'):
        if tr.get('class', None) is None:
            df['date'].append(game_info['date'])
            df['week'].append(game_info['week'])
            df['season'].append(game_info['season'])
            df['team'].append(team)
            df['opponent'].append(opponent)
            df['home_field'].append(home)
            for td in tr.find_all(['th', 'td']):
                if td['data-stat'] != 'team':
                    df[td['data-stat']].append(td.text) 
        else:
            team = game_info['home']
            opponent = game_info['away']
            home = True

    return pd.DataFrame(df)


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
    first_score_team = three_straight.find('tbody').find_all('td', attrs={'data-stat': 'team'})[0].text
    first_score_quarter = three_straight.find('tbody').find_all('th', attrs={'data-stat': 'quarter'})[0].text
    first_score_time = three_straight.find('tbody').find_all('td', attrs={'data-stat': 'time'})[0].text
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
    df['first_score_team'] = [first_score_team, first_score_team]
    df['first_score_quarter'] = [first_score_quarter, first_score_quarter]
    df['first_score_time'] = [first_score_time, first_score_time]
    
    dfs['scores'] = dfs['scores'].append(pd.DataFrame(df), ignore_index=True)


def scrape_team_stats(team_stats, game_info, dfs):
    """
    Function: 
        Scrape team stats

    Input:
        team_stats: tag
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

    rows = team_stats.find('tbody').find_all('tr')
    df['1st'] = [td.text for td in rows[0].find_all('td')]
    df['rush_att'] = [td.text.split('-')[0] for td in rows[1].find_all('td')]
    df['rush_yds'] = [td.text.split('-')[1] for td in rows[1].find_all('td')]
    df['rush_tds'] = [td.text.split('-')[2] for td in rows[1].find_all('td')]
    df['cmp'] = [td.text.split('-')[0] for td in rows[2].find_all('td')]
    df['att'] = [td.text.split('-')[1] for td in rows[2].find_all('td')]
    df['pass_yds'] = [td.text.split('-')[2] for td in rows[2].find_all('td')]
    df['pass_tds'] = [td.text.split('-')[3] for td in rows[2].find_all('td')]
    df['ints'] = [td.text.split('-')[4] for td in rows[2].find_all('td')]
    df['sacks'] = [td.text.split('-')[0] for td in rows[3].find_all('td')]
    df['sack_yds'] = [td.text.split('-')[1] for td in rows[3].find_all('td')]
    df['net_pass_yds'] = [td.text for td in rows[4].find_all('td')]
    df['total_yds'] = [td.text for td in rows[5].find_all('td')]
    df['fum'] = [td.text.split('-')[0] for td in rows[6].find_all('td')]
    df['fum_lost'] = [td.text.split('-')[1] for td in rows[6].find_all('td')]
    df['to'] = [td.text for td in rows[7].find_all('td')]
    df['pen'] = [td.text.split('-')[0] for td in rows[8].find_all('td')]
    df['pen_yds'] = [td.text.split('-')[1] for td in rows[8].find_all('td')]
    df['3rd_att'] = [td.text.split('-')[0] for td in rows[9].find_all('td')]
    df['3rd_cmp'] = [td.text.split('-')[1] for td in rows[9].find_all('td')]
    df['4th_att'] = [td.text.split('-')[0] for td in rows[10].find_all('td')]
    df['4th_cmp'] = [td.text.split('-')[1] for td in rows[10].find_all('td')]
    df['poss'] = [td.text for td in rows[11].find_all('td')]
    
    dfs['team_stats'] = dfs['team_stats'].append(pd.DataFrame(df), ignore_index=True)


def scrape_player_offense(player_offense, game_info, dfs):
    """
    Function: 
        Scrape offensive player stats

    Input:
        player_offense: tag
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
    
    df = scrape_tag(player_offense, game_info)

    dfs['player_offense'] = dfs['player_offense'].append(df, ignore_index=True)


def scrape_player_defense(player_defense, game_info, dfs):
    """
    Function: 
        Scrape defensive player stats

    Input:
        player_defense: tag
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
    
    df = scrape_tag(player_defense, game_info)

    dfs['player_defense'] = dfs['player_defense'].append(df, ignore_index=True)


def scrape_returns(returns, game_info, dfs):
    """
    Function: 
        Scrape returning stats

    Input:
        returns: tag
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
    
    df = scrape_tag(returns, game_info)

    dfs['returns'] = dfs['returns'].append(df, ignore_index=True)


def scrape_kicking(kicking, game_info, dfs):
    """
    Function: 
        Scrape kicking stats

    Input:
        kicking: tag
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
    
    df = scrape_tag(kicking, game_info)

    dfs['kicking'] = dfs['kicking'].append(df, ignore_index=True)


def scrape_starters(home_starters, away_starters, game_info, dfs):
    """
    Function: 
        Scrape starters

    Input:
        home_starters: tag
        away_starters: tag
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
    
    home_df = scrape_home_away_tag(home_starters, game_info, 'home')
    away_df = scrape_home_away_tag(away_starters, game_info, 'away')
    
    dfs['starters'] = dfs['starters'].append(home_df, ignore_index=True)
    dfs['starters'] = dfs['starters'].append(away_df, ignore_index=True)


def scrape_drives(home_drives, away_drives, game_info, dfs):
    """
    Function: 
        Scrape drives

    Input:
        home_drives: tag
        away_drives: tag
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
    
    home_df = scrape_home_away_tag(home_drives, game_info, 'home', drives=True)
    away_df = scrape_home_away_tag(away_drives, game_info, 'away', drives=True)
    
    dfs['drives'] = dfs['drives'].append(home_df, ignore_index=True)
    dfs['drives'] = dfs['drives'].append(away_df, ignore_index=True)


def scrape_game(game_info, dfs):
    
    """
    Function: 
        Scrape game. Data includes team and player stats.
        Following DataFrames:
            ~ scores
            ~ team_stats
            ~ player_offense
            ~ player_defense
            ~ returns
            ~ kicking
            ~ starters
            ~ drives

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
    scores = soup.find('table', attrs={'class': 'linescore nohover stats_table no_freeze'})
    three_straight = soup.find('table', attrs={'class': 'stats_table', 'id': 'scoring'})
    team_stats = soup.find('table', attrs={'id': 'team_stats'})
    player_offense = soup.find('table', attrs={'id': 'player_offense'})
    player_defense = soup.find('table', attrs={'id': 'player_defense'})
    returns = soup.find('table', attrs={'id': 'returns'})
    kicking = soup.find('table', attrs={'id': 'kicking'})
    home_starters = soup.find('table', attrs={'id': 'home_starters'})
    away_starters = soup.find('table', attrs={'id': 'vis_starters'})
    home_drives = soup.find('table', attrs={'id': 'home_drives'})
    away_drives = soup.find('table', attrs={'id': 'vis_drives'})

    # Look at comments as well
    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment_soup = BeautifulSoup(comment, 'html.parser')
        if scores is None: # Scores
            scores = comment_soup.find('table', attrs={'class': 'linescore nohover stats_table no_freeze'})
        if three_straight is None: # Three Consectutive Scores
            three_straight = comment_soup.find('table', attrs={'class': 'stats_table', 'id': 'scoring'})
        if team_stats is None: # Team stats
            team_stats = comment_soup.find('table', attrs={'id': 'team_stats'})
        if player_offense is None: # Player offense
            player_offense = comment_soup.find('table', attrs={'id': 'player_offense'})
        if player_defense is None: # Player defense
            player_defense = comment_soup.find('table', attrs={'id': 'player_defense'})
        if returns is None: # Returns
            returns = comment_soup.find('table', attrs={'id': 'returns'})
        if kicking is None: # Kicking
            kicking = comment_soup.find('table', attrs={'id': 'kicking'})
        if home_starters is None: # Starters
            home_starters = comment_soup.find('table', attrs={'id': 'home_starters'})
        if away_starters is None: # Starters
            away_starters = comment_soup.find('table', attrs={'id': 'vis_starters'})
        if home_drives is None: # Drives
            home_drives = comment_soup.find('table', attrs={'id': 'home_drives'})
        if away_drives is None: # Drives
            away_drives = comment_soup.find('table', attrs={'id': 'vis_drives'})
    
    # Scrape data for each table
    scrape_scores(scores, three_straight, game_info, dfs)
    scrape_team_stats(team_stats, game_info, dfs)
    scrape_player_offense(player_offense, game_info, dfs)
    scrape_player_defense(player_defense, game_info, dfs)
    scrape_returns(returns, game_info, dfs)
    scrape_kicking(kicking, game_info, dfs)
    scrape_starters(home_starters, away_starters, game_info, dfs)
    scrape_drives(home_drives, away_drives, game_info, dfs)


def scrape_week(href, season, dfs):
    """
    Function: 
        Scrapes games in week. Data includes team and player stats.
        Following DataFrames:
            ~ scores
            ~ team_stats
            ~ player_offense
            ~ player_defense
            ~ returns
            ~ kicking
            ~ starters
            ~ drives

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
        Scrapes games in season. Data includes team and player stats.
        Following DataFrames:
            ~ scores
            ~ team_stats
            ~ player_offense
            ~ player_defense
            ~ returns
            ~ kicking
            ~ starters
            ~ drives

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
        Scrapes game data since 2010. Data includes team and player stats.
        Writes following DataFrames to CSV files:
            ~ scores
            ~ team_stats
            ~ player_offense
            ~ player_defense
            ~ returns
            ~ kicking
            ~ starters
            ~ drives

    Input:
        None
        
    Output:
        None
    """
    # Initialize DataFrames
    dfs = {
        'scores': pd.DataFrame(),
        'team_stats': pd.DataFrame(),
        'player_offense': pd.DataFrame(), 
        'player_defense': pd.DataFrame(),
        'returns': pd.DataFrame(), 
        'kicking': pd.DataFrame(), 
        'starters': pd.DataFrame(),
        'drives': pd.DataFrame()
    }

    # Iterate over each season
    for season in range(2010, 2022):
        scrape_season(season, dfs) # Append season to data structure

    # Write DataFrames to CSV files
    for df in dfs:
        dfs[df].to_csv(f'backend/data/games/{df}.csv', index=False)


if __name__ == '__main__':
    main()