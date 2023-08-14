import sys
sys.path.append('/home/tylerengland/NFL/')
sys.path.append('/home/tylerengland/NFL/backend/simulation/')

import numpy as np
import nfl_data_py as nfl
from backend.simulation.simulation import Game
from statistics import mean
from sklearn.metrics import mean_squared_error, r2_score


class BackTest():
    def __init__(self, num_sims):
        self.num_sims = num_sims
        self.boxscores = {}
        self.predictions = {}

    def load_data(self):
        backtest_years = [2022]
        pbp = nfl.import_pbp_data(backtest_years, downcast=True, cache=False, alt_path=None)

        # Set pandas display options
        df = pbp.loc[~pbp['posteam_type'].isna(), :].copy()
        df.fillna(0, inplace=True)

        # Score columns
        df.loc[:, 'home_score'] = df.loc[:, 'total_home_score'].astype(int)
        df.loc[:, 'away_score'] = df.loc[:, 'total_away_score'].astype(int)

        # Passing columns
        df.loc[:, 'posteam_type'] = df.loc[:, 'posteam_type'].map({'home': 1, 'away': 0}).astype(int)
        df.loc[:, 'home_pass_yards'] = np.where((df['pass_attempt'].astype(int)) & (df['posteam_type']), df['yards_gained'], 0).copy()
        df.loc[:, 'away_pass_yards'] = np.where((df['pass_attempt'].astype(int)) & (~df['posteam_type']), df['yards_gained'], 0)
        df.loc[:, 'home_pass_atts'] = np.where((df['pass_attempt'].astype(int)) & (df['posteam_type']), 1, 0)
        df.loc[:, 'away_pass_atts'] = np.where((df['pass_attempt'].astype(int)) & (~df['posteam_type']), 1, 0)
        df.loc[:, 'home_pass_tds'] = np.where((df['pass_touchdown'].astype(int)) & (df['posteam_type']), 1, 0)
        df.loc[:, 'away_pass_tds'] = np.where((df['pass_touchdown'].astype(int)) & (~df['posteam_type']), 1, 0)
        df.loc[:, 'home_pass_ints'] = np.where((df['interception'].astype(int)) & (df['posteam_type']), 1, 0)
        df.loc[:, 'away_pass_ints'] = np.where((df['interception'].astype(int)) & (~df['posteam_type']), 1, 0)

        # Rushing columns
        df.loc[:, 'home_rush_yards'] = np.where((df['rush_attempt'].astype(int)) & (df['posteam_type']), df['yards_gained'], 0)
        df.loc[:, 'away_rush_yards'] = np.where((df['rush_attempt'].astype(int)) & (~df['posteam_type']), df['yards_gained'], 0)
        df.loc[:, 'home_rush_atts'] = np.where((df['rush_attempt'].astype(int)) & (df['posteam_type']), 1, 0)
        df.loc[:, 'away_rush_atts'] = np.where((df['rush_attempt'].astype(int)) & (~df['posteam_type']), 1, 0)
        df.loc[:, 'home_rush_tds'] = np.where((df['rush_touchdown'].astype(int)) & (df['posteam_type']), 1, 0)
        df.loc[:, 'away_rush_tds'] = np.where((df['rush_touchdown'].astype(int)) & (~df['posteam_type']), 1, 0)
        df.loc[:, 'home_rush_fums'] = np.where((df['fumble'].astype(int)) & (df['posteam_type']), 1, 0)
        df.loc[:, 'away_rush_fums'] = np.where((df['fumble'].astype(int)) & (~df['posteam_type']), 1, 0)

        # Defensive columns
        df.loc[:, 'home_def_ints'] = np.where((df['interception'].astype(int)) & (~df['posteam_type']), 1, 0)
        df.loc[:, 'away_def_ints'] = np.where((df['interception'].astype(int)) & (df['posteam_type']), 1, 0)
        df.loc[:, 'home_def_fums'] = np.where((df['fumble'].astype(int)) & (~df['posteam_type']), 1, 0)
        df.loc[:, 'away_def_fums'] = np.where((df['fumble'].astype(int)) & (df['posteam_type']), 1, 0)

        # Special teams columns
        df.loc[:, 'home_punt_atts'] = np.where((df['punt_attempt'].astype(int)) & (df['posteam_type']), 1, 0)
        df.loc[:, 'away_punt_atts'] = np.where((df['punt_attempt'].astype(int)) & (~df['posteam_type']), 1, 0)
        df.loc[:, 'home_fg_atts'] = np.where((df['field_goal_attempt'].astype(int)) & (df['posteam_type']), 1, 0)
        df.loc[:, 'away_fg_atts'] = np.where((df['field_goal_attempt'].astype(int)) & (~df['posteam_type']), 1, 0)
        df.loc[:, 'home_fg_made'] = np.where((df['field_goal_result'] == 'made') & (df['posteam_type']), 1, 0)
        df.loc[:, 'away_fg_made'] = np.where((df['field_goal_result'] == 'made') & (~df['posteam_type']), 1, 0)

        game_cols = [
            'game_id', 'game_date', 'home_team', 'away_team', 'season_type', 'week', 'season',
        ]

        # Aggregate stats
        df = (
            df.groupby(game_cols)
            .agg(
                {
                    'home_score': 'max', 'away_score': 'max',
                    'home_pass_yards': 'sum', 'away_pass_yards': 'sum',
                    'home_pass_atts': 'sum', 'away_pass_atts': 'sum',
                    'home_pass_tds': 'sum', 'away_pass_tds': 'sum',
                    'home_pass_ints': 'sum', 'away_pass_ints': 'sum',
                    'home_rush_yards': 'sum', 'away_rush_yards': 'sum',
                    'home_rush_atts': 'sum', 'away_rush_atts': 'sum',
                    'home_rush_tds': 'sum', 'away_rush_tds': 'sum',
                    'home_rush_fums': 'sum', 'away_rush_fums': 'sum',
                    'home_def_ints': 'sum', 'away_def_ints': 'sum',
                    'home_def_fums': 'sum', 'away_def_fums': 'sum',
                    'home_punt_atts': 'sum', 'away_punt_atts': 'sum',
                    'home_fg_atts': 'sum', 'away_fg_atts': 'sum',
                    'home_fg_made': 'sum', 'away_fg_made': 'sum'
                }
            )
            .reset_index()
        )

        # Calculate spread and total
        df['spread'] = df['home_score'] - df['away_score']
        df['total'] = df['home_score'] + df['away_score']

        return df

    def update_boxscores(self, boxscore):
        for stat in boxscore:
            if stat not in self.boxscores:
                self.boxscores[stat] = [boxscore[stat]]
            else:
                self.boxscores[stat].append(boxscore[stat])

    def aggregate_boxscores(self):
        for stat in self.boxscores:
            if stat not in self.predictions:
                self.predictions[stat] = [mean(self.boxscores[stat])]
            else:
                self.predictions[stat].append(mean(self.boxscores[stat]))
        self.boxscores = {}

    def evaluate(self):
        for stat in self.predictions:
            actual = self.actuals[stat]
            pred = self.predictions[stat]
            mse = mean_squared_error(actual, pred)
            r2 = r2_score(actual_scores, sim_scores)
            
            print(f"Team: {team}")
            print(f"Actual Scores: {actual_scores}")
            print(f"Simulated Scores: {sim_scores}")
            print(f"MSE: {mse}, R2: {r2}")
            print("====================")

    def run(self):
        games = self.load_data()

        for index, game in games.iterrows():
            # Extract game data
            home_team = game['home_team']
            away_team = game['away_team']

            # Simulate game and update boxscores
            game_sim = Game(home_team, away_team, verbose=False)
            for _ in range(self.num_sims):
                boxscore = game_sim.simulate_game()
                self.update_boxscores(boxscore)

            # Aggregate boxscores
            self.aggregate_boxscores()
        
        
        self.evaluate()

if __name__ == '__main__':
    backtest = BackTest(num_sims=5000)
    backtest.run()
