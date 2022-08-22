from re import X
from turtle import left
import pandas as pd
import time
import numpy as np


def timeis(func):
    """
    Function:
        Wrapper to time execution

    Input:
        func: Function
    
    Output:
        None
    """
    def wrap(*args, **kwargs):
        print(f'Executing {func.__name__} ...')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'\tDone: {(end - start):.4f}s')
        return result

    return wrap


def load_team_stats():
    """
    Function: 
        Loads following files:
            ~ team_stats.csv
        
        Returned DataFrame should have each entry be a team's performance.
            ~ opponent and home field advantage listed
        Index is date, home, away.

    Input:
        None

    Output:
        df: DataFrame
    """
    df = pd.read_csv('backend/data/games/team_stats.csv')

    # Clean data
    df['date'] = pd.to_datetime(df['date']) # Date data type
    df['poss'] = df['poss'].apply(lambda x: round(int(x.split(':')[0]) + int(x.split(':')[1]) / 60, 2))
    df['away'] = np.where(df['home_field'], df['opponent'], df['team'])
    df['home'] = np.where(df['home_field'], df['team'], df['opponent'])
    df.set_index(
        keys=['date', 'home', 'away'], 
        inplace=True, 
        drop=True
    )

    return df


def load_player_offense():
    """
    Function: 
        Loads following files:
            ~ player_offense.csv
        
        Groups by team and date.
        Returned DataFrame should have each entry be a team's performance.
            ~ opponent and home field advantage listed
        Index is date, home, away.

    Input:
        None

    Output:
        df: DataFrame
    """
    df = pd.read_csv('backend/data/games/player_offense.csv')
    
    # Clean data
    df['date'] = pd.to_datetime(df['date'])
    df.fillna(0, inplace=True)

    # Groupby
    df = df.groupby(
        ['date', 'team', 'opponent', 'home_field', 'week', 'season']
    ).aggregate(
        {
            'pass_cmp': np.sum, 'pass_att': np.sum, 
            'pass_yds': np.sum, 'pass_td': np.sum, 'pass_int': np.sum, 
            'pass_sacked': np.sum, 'pass_sacked_yds': np.sum, 
            'pass_long': np.max, 'pass_rating': np.max, 
            'rush_att': [np.sum, np.count_nonzero], 'rush_yds': np.sum, 
            'rush_td': [np.sum, np.count_nonzero], 'rush_long': np.max, 
            'rec': [np.sum, np.count_nonzero], 'rec_yds': np.sum, 
            'rec_td': [np.sum, np.count_nonzero], 'rec_long': np.max, 
            'fumbles': [np.sum, np.count_nonzero], 'fumbles_lost': np.sum
        }
    ).reset_index()

    df.columns = [f'{col[0]}_count' if col[1] == 'count_nonzero' else f'{col[0]}' for col in df.columns]

    df['away'] = np.where(df['home_field'], df['opponent'], df['team'])
    df['home'] = np.where(df['home_field'], df['team'], df['opponent'])
    df.set_index(keys=['date', 'home', 'away'], inplace=True, drop=True)

    return df


def load_player_defense():
    """
    Function: 
        Loads following files:
            ~ player_defense.csv
        
        Groups by team and date.
        Returned DataFrame should have each entry be a team's performance.
            ~ opponent and home field advantage listed
        Index is date, home, away.

    Input:
        None

    Output:
        df: DataFrame
    """
    df = pd.read_csv('backend/data/games/player_defense.csv')
    
    # Clean data
    df['date'] = pd.to_datetime(df['date'])
    df.fillna(0, inplace=True)

    # Groupby
    df = df.groupby(
        ['date', 'team', 'opponent', 'home_field', 'week', 'season']
    ).aggregate(
        {
            'def_int': [np.sum, np.count_nonzero], 'def_int_yds': np.sum, 'def_int_td': np.sum, 'def_int_long': np.max, 
            'pass_defended': [np.sum, np.count_nonzero], 'sacks': [np.sum, np.count_nonzero], 
            'tackles_solo': [np.sum, np.count_nonzero], 'tackles_assists': np.sum, 
            'tackles_loss': np.sum, 'qb_hits': [np.sum, np.count_nonzero], 
            'fumbles_rec': np.sum, 'fumbles_rec_yds': np.sum, 
            'fumbles_rec_td': np.sum, 'fumbles_forced': [np.sum, np.count_nonzero]
        }
    ).reset_index()

    df.columns = [f'{col[0]}_count' if col[1] == 'count_nonzero' else f'{col[0]}' for col in df.columns]

    df['away'] = np.where(df['home_field'], df['opponent'], df['team'])
    df['home'] = np.where(df['home_field'], df['team'], df['opponent'])
    df.set_index(keys=['date', 'home', 'away'], inplace=True, drop=True)

    return df


def load_returns():
    """
    Function: 
        Loads following files:
            ~ returns.csv
        
        Groups by team and date.
        Returned DataFrame should have each entry be a team's performance.
            ~ opponent and home field advantage listed
        Index is date, home, away.

    Input:
        None

    Output:
        df: DataFrame
    """
    df = pd.read_csv('backend/data/games/returns.csv')

    # Clean data
    df['date'] = pd.to_datetime(df['date'])
    df.fillna(0, inplace=True)

    # Groupby
    df = df.groupby(
        ['date', 'team', 'opponent', 'home_field', 'week', 'season']
    ).aggregate(
        {
            'kick_ret': np.sum, 'kick_ret_yds': np.sum,
            'kick_ret_td': np.sum, 'kick_ret_long': np.max, 
            'punt_ret': np.sum, 'punt_ret_yds': np.sum,
            'punt_ret_td': np.sum, 'punt_ret_long': np.max
        }
    ).reset_index()

    df['away'] = np.where(df['home_field'], df['opponent'], df['team'])
    df['home'] = np.where(df['home_field'], df['team'], df['opponent'])
    df.set_index(keys=['date', 'home', 'away'], inplace=True, drop=True)

    return df


def load_kicking():
    """
    Function: 
        Loads following files:
            ~ kicking.csv
        
        Groups by team and date.
        Returned DataFrame should have each entry be a team's performance.
            ~ opponent and home field advantage listed
        Index is date, home, away.

    Input:
        None

    Output:
        df: DataFrame
    """
    df = pd.read_csv('backend/data/games/kicking.csv')

    # Clean data
    df['date'] = pd.to_datetime(df['date'])
    df.fillna(0, inplace=True)

    # Groupby
    df = df.groupby(
        ['date', 'team', 'opponent', 'home_field', 'week', 'season']
    ).aggregate(
        {
            'xpm': np.sum, 'xpa': np.sum, 
            'fgm': np.sum, 'fga': np.sum, 
            'punt': np.sum, 'punt_yds': np.sum, 'punt_long': np.max
        }
    ).reset_index()

    df['away'] = np.where(df['home_field'], df['opponent'], df['team'])
    df['home'] = np.where(df['home_field'], df['team'], df['opponent'])
    df.set_index(keys=['date', 'home', 'away'], inplace=True, drop=True)

    return df


def merge_offensive(team_stats, offense, defense):
    """
    Function:
        Merges a team's offensive stats, taking into account the opposing defense QB hits, sacks, etc.
    
    Input:
        team_stats: DataFrame
        offense: DataFrame
        defense: DataFrame

    Output:
        df: DataFrame
    """
    # Merge offense
    df = pd.merge(
        team_stats[
            [
                'team', '1st', 'sacks', 'sack_yds', 'net_pass_yds', 'total_yds', 'to', 
                'pen', 'pen_yds', '3rd_att', '3rd_cmp', '4th_att', '4th_cmp', 'poss'
            ]
        ],
        offense, 
        right_index=True,
        left_index=True
    )
    df = df[df['team_x'] == df['team_y']].drop(['team_y'], axis=1).rename({'team_x': 'team'}, axis=1)

    # Merge opponent's defense
    df = pd.merge(
        df,
        defense[['team', 'def_int_td', 'pass_defended', 'tackles_loss', 'qb_hits', 'fumbles_rec_td']],
        right_index=True,
        left_index=True
    )
    df = df[df['opponent'] == df['team_y']].drop(['team_y'], axis=1).rename({'team_x': 'team'}, axis=1)

    return df


def merge_defensive(team_stats, defense, offense):
    """
    Function:
        Merges a team's defensive stats, taking into account the opposing offense rushing and passing.
    
    Input:
        team_stats: DataFrame
        defense: DataFrame
        offense: DataFrame

    Output:
        df: DataFrame
    """
    # Merge defense to opposing offense
    df = pd.merge(
        defense,
        team_stats[
            [
                'team', '1st', 'net_pass_yds', 'total_yds', 'to', 
                '3rd_att', '3rd_cmp', '4th_att', '4th_cmp', 
            ]
        ],
        right_index=True,
        left_index=True
    )
    df = df[df['team_y'] == df['opponent']].drop(['team_y'], axis=1).rename({'team_x': 'team'}, axis=1)

    # Merge defense to opposing offense
    df = pd.merge(
        df,
        offense[
            [
                'team', 'pass_cmp', 'pass_att', 'pass_yds', 'pass_td', 'pass_long', 'pass_rating', 
                'rush_att', 'rush_yds', 'rush_td', 'rush_long', 
                'rec', 'rec_yds', 'rec_td', 'rec_long'
            ]
        ],
        left_index=True,
        right_index=True
    )
    df = df[df['team_y'] == df['opponent']].drop(['team_y'], axis=1).rename({'team_x': 'team'}, axis=1)

    return df


def merge_special_teams(returns, kicking):
    """
    Function:
        Merges a team's special team stats, taking into account the opposing returning and kicking.
    
    Input:
        returns: DataFrame
        kicking: DataFrame

    Output:
        df: DataFrame
    """
    # Merge returning and kicking
    df = pd.merge(
        kicking,
        returns[
            [
                'team', 'kick_ret', 'kick_ret_yds', 'kick_ret_td', 'kick_ret_long', 
                'punt_ret', 'punt_ret_yds', 'punt_ret_td', 'punt_ret_long'
            ]
        ],
        how='left',
        left_index=True,
        right_index=True
    )
    df = df[df['team_x'] == df['team_y']].drop(['team_y'], axis=1).rename({'team_x': 'team'}, axis=1)
    
    # Merge fielding 
    df = pd.merge(
        df,
        returns[
            [
                'team', 'kick_ret', 'kick_ret_yds', 'kick_ret_td', 'kick_ret_long', 
                'punt_ret', 'punt_ret_yds', 'punt_ret_td', 'punt_ret_long'
            ]
        ],
        how='left',
        left_index=True,
        right_index=True,
        suffixes=('', '_fielding')
    )
    df = df[df['team_fielding'] == df['opponent']].drop(['team_fielding'], axis=1)
    
    # Merge blocking
    df = pd.merge(
        df,
        kicking[['team', 'xpm', 'xpa', 'fgm', 'fga']],
        how='left',
        left_index=True,
        right_index=True,
        suffixes=('', '_block')
    )
    df = df[df['team_block'] == df['opponent']].drop(['team_block'], axis=1)

    return df


def merge_squads(offensive_df, defensive_df, special_teams):
    """
    Function:
        Merges a team's offensive, defensive and special teams stats.
        Returns a DataFrame with each entry being a team's performance in a game.
    
    Input:
        offensive_df: DataFrame
        defensive_df: DataFrame
        special_teams: DataFrame

    Output:
        df: DataFrame
    """
    # Add team to index
    offensive_df.set_index(['team'], append=True, inplace=True)
    offensive_df.drop(['home_field', 'week', 'season'], axis=1, inplace=True)
    defensive_df.set_index(['team'], append=True, inplace=True)
    defensive_df.drop(['home_field', 'week', 'season'], axis=1, inplace=True)
    special_teams.set_index(['team'], append=True, inplace=True)
    special_teams.drop(['home_field', 'week', 'season'], axis=1, inplace=True)

    # Merge
    df = pd.merge(
        offensive_df,
        defensive_df.drop(['opponent'], axis=1),
        left_index=True,
        right_index=True,
        how='left',
        suffixes=('_off', '_def')
    )
    df = pd.merge(
        df,
        special_teams.drop(['opponent'], axis=1),
        left_index=True, 
        right_index=True,
        how='left'
    )

    return df


@timeis
def load_data():
    """
    Function: 
        Loads following files:
            ~ kicking.csv
            ~ player_offense.csv
            ~ player_defense.csv
            ~ returns.csv
            ~ team_stats.csv
        
        Merge offensive stats.
        Merge defensive stats.
        Merge special team stats.
        
        Returned DataFrame should have each entry be a game with team's offensive, defensive, and 
        special team statistics.

    Input:
        None

    Output:
        df: DataFrame
    """
    # Load data
    team_stats = load_team_stats()
    offense = load_player_offense()
    defense = load_player_defense()
    returns = load_returns()
    kicking = load_kicking()

    # Merge data by squads
    offensive_df = merge_offensive(team_stats, offense, defense)
    defensive_df = merge_defensive(team_stats, defense, offense)
    special_teams = merge_special_teams(returns, kicking)

    # Merge data
    df = merge_squads(offensive_df, defensive_df, special_teams)

    return df


def sma(df, bin):
    """
    Function: 
        Simple moving average.
        
    Input:
        df: DataFrame
        bin: int

    Output:
        df: DataFrame
    """
    # Index
    sorted_index = df.sort_index(level=['team', 'date']).index

    df = df.sort_index(
        level=['team', 'date']
    ).groupby(
        ['team']
    ).rolling(
        window=bin, 
        min_period=bin,
        closed='left'
    ).mean()
    df.index = sorted_index

    return df


def cma(df, bin):
    """
    Function: 
        Cumulative moving average.
        
    Input:
        df: DataFrame
        bin: int

    Output:
        df: DataFrame
    """
    # Index
    sorted_index = df.sort_index(level=['team', 'date']).index

    df = df.sort_index(
        level=['team', 'date']
    ).groupby(
        ['team']
    ).shift(
        periods=1
    ).groupby(
        ['team']
    ).expanding( 
        min_periods=bin
    ).mean()
    df.index = sorted_index

    return df


def ema(df, bin):
    """
    Function: 
        Exponential moving average.
        
    Input:
        df: DataFrame
        bin: int

    Output:
        df: DataFrame
    """
    # Index
    sorted_index = df.sort_index(level=['team', 'date']).index

    df = df.sort_index(
        level=['team', 'date']
    ).groupby(
        ['team']
    ).shift(
        periods=1
    ).groupby(
        ['team']
    ).ewm(
        span=bin,
        min_periods=bin,
    ).mean()
    df.index = sorted_index

    return df


def merge_averages(sma, ema, cma):
    """
    Function: 
        Merge together each moving average

        Retrurn DataFrame so that each entry contains a sma, ema, and cma for each stat.
        
    Input:
        sma: DataFrame
        ema: DataFrame
        cma: DataFrame

    Output:
        df: DataFrame
    """
    df = pd.merge(
        sma,
        ema, 
        left_index=True,
        right_index=True,
        suffixes=('_sma', '_ema')
    )

    cma.columns = [f'{col}_cma' for col in cma.columns]
    df = pd.merge(
        df,
        cma, 
        left_index=True,
        right_index=True
    )

    df = pd.merge(
        ema,
        cma, 
        left_index=True,
        right_index=True,
        suffixes=('_ema', '_cma')
    )

    return df


def feature_engineer(df):
    """
    Function: 
        Feature engineer following stats:
            ~ cmp_perc_off (sma, ema, cma)
            ~ cmp_perc_def (sma, ema, cma)
            ~ pass_yds_per_att_off (sma, ema, cma)
            ~ pass_yds_per_att_def (sma, ema, cma)
            ~ rush_yds_per_att_off (sma, ema, cma)
            ~ rush_yds_per_att_def (sma, ema, cma)
            ~ rec_yds_per_rec_off (sma, ema, cma)
            ~ rec_yds_per_rec_def (sma, ema, cma)
            ~ punt_yds_per_ret (sma, ema, cma)
            ~ kick_yds_per_ret (sma, ema, cma)
            ~ punt_yds_per_ret_fielding (sma, ema, cma)
            ~ kick_yds_per_ret_fielding (sma, ema, cma)
            ~ punt_yds_per_punt (sma, ema, cma)
            ~ xp_perc (sma, ema, cma)
            ~ xp_perc_block (sma, ema, cma)
            ~ fg_perc (sma, ema, cma)
            ~ fg_perc_block (sma, ema, cma)

    Input:
        df: DataFrame

    Output:
        df: DataFrame
    """
    for ma in ['sma', 'ema', 'cma']:
        
        for team in ['off', 'def']:
            df[f'cmp_perc_{team}_{ma}'] = df[f'pass_cmp_{team}_{ma}'] / df[f'pass_att_{team}_{ma}']
            df[f'pass_yds_per_att_{team}_{ma}'] = df[f'pass_yds_{team}_{ma}'] / df[f'pass_att_{team}_{ma}']
            df[f'rush_yds_per_att_{team}_{ma}'] = df[f'rush_yds_{team}_{ma}'] / df[f'rush_att_{team}_{ma}']
            df[f'rec_yds_per_rec_{team}_{ma}'] = df[f'rec_yds_{team}_{ma}'] / df[f'rec_{team}_{ma}']
        
        df[f'punt_yds_per_ret_{ma}'] = df[f'punt_ret_yds_{ma}'] / df[f'punt_ret_{ma}']
        df[f'kick_yds_per_ret_{ma}'] = df[f'kick_ret_yds_{ma}'] / df[f'kick_ret_{ma}']
        df[f'punt_yds_per_ret_fielding_{ma}'] = df[f'punt_ret_yds_fielding_{ma}'] / df[f'punt_ret_fielding_{ma}']
        df[f'kick_yds_per_ret_{ma}'] = df[f'kick_ret_yds_fielding_{ma}'] / df[f'kick_ret_fielding_{ma}']
        df[f'punt_yds_per_punt_{ma}'] = df[f'punt_yds_{ma}'] / df[f'punt_{ma}']
        df[f'xp_perc_{ma}'] = df[f'xpm_{ma}'] / df[f'xpa_{ma}']
        df[f'xp_perc_block_{ma}'] = df[f'xpm_block_{ma}'] / df[f'xpa_block_{ma}']
        df[f'fg_perc_{ma}'] = df[f'fgm_{ma}'] / df[f'fga_{ma}']
        df[f'fg_perc_block_{ma}'] = df[f'fgm_block_{ma}'] / df[f'fga_block_{ma}']

            
    return df


def merge_matchup(df):
    """
    Function: 
        Retrurn DataFrame so that each entry contains both home and away stats. 
        
    Input:
        df: DataFrame

    Output:
        df: DataFrame
    """
    # Remove team from index
    df.reset_index('team', inplace=True)

    # Home field advantage
    df['home_field'] = np.where(df.index.get_level_values(1) == df['team'], 1, 0)

    # Merge
    df = pd.merge(df,df, left_index=True, right_index=True)
    df = df[df['team_x'] != df['team_y']].drop(['team_x', 'team_y'], axis=1)
    df = df.groupby(df.index).first()
    
    return df


@timeis
def preprocess(X_df):
    """
    Function: 
        Preprocess each game so that each entry consists of a team's 5 game sma, 5 game ema,
        and season average.

        Retrurn DataFrame so that each entry contains both home and away stats. 
        
    Input:
        X_df: DataFrame

    Output:
        df: DataFrame
    """
    # 5 game SMA
    sma_df = sma(X_df, 5)

    # Season CMA
    cma_df = cma(X_df, 5)

    # 5 game EMA
    ema_df = ema(X_df, 5)

    # Merge SMA, EMA, CMA
    df = merge_averages(sma_df, ema_df, cma_df)

    # Feature Engineeer
    df = feature_engineer(df)

    # Merge matchup
    df = merge_matchup(df)

    # Deal with nan's
    df.dropna(axis=0, inplace=True)

    return df

    
@timeis
def load_target_data():
    """
    Function: 
        Retrurn DataFrame so that each entry contains winner represented as a binary:
            home_win: 1
            away_win: 2
        
    Input:
        None

    Output:
        df: DataFrame
    """
    # Load scores
    df = pd.read_csv('backend/data/games/scores.csv')
    df.columns = [col.lower() for col in df.columns]

    # Clean data
    df['date'] = pd.to_datetime(df['date'])
    df['home'] = np.where(df['home_field'], df['team'], df['opponent'])
    df['away'] = np.where(df['home_field'], df['opponent'], df['team'])
    df.set_index(['date', 'home', 'away'], inplace=True, drop=True)
    
    # Merge home and away scores
    df = pd.merge(
        df[df['home_field'] == 1],
        df[df['home_field'] == 0],
        left_index=True,
        right_index=True,
        suffixes=('_h', '_a')
    )

    # Outcome
    df['y'] = (df['final_h'] > df['final_a']).apply(int)
    return df[['y']]


@timeis
def merge_x_y(X_df, y_df):
    """
    Function: 
        Retrurn DataFrame so that each entry contains input variables and target variables
        
    Input:
        X_df: DataFrame
        y_df: DataFrame

    Output:
        df: DataFrame
    """
    y_df.index = [(index[0].date(), index[1], index[2]) for index in y_df.index]
    X_df.index = [(index[0].date(), index[1], index[2]) for index in X_df.index]
    
    df = pd.merge(
        X_df,
        y_df,
        left_index=True,
        right_index=True,
        how='left'
    )

    return df


def main():
    """
    Function: 
        Preprocess a team's stats. 
        Focuses on last 5 performances weighted/unweighted and seasonal average.

    Input:
        None

    Output:
        df: DataFrame
    """
    # Load X data
    X_df = load_data()

    # Preprocess data
    X_df = preprocess(X_df)

    # Load y data
    y_df = load_target_data()

    # Merge X and y
    df = merge_x_y(X_df, y_df)

    return df
    

if __name__ == '__main__':
    main()