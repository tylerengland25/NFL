import sys
sys.path.append('/home/tylerengland/NFL/')
sys.path.append('/home/tylerengland/NFL/backend/simulation/')

import json
import pandas as pd
from backend.simulation.simulation import Game
from statistics import mean
from sklearn.metrics import mean_squared_error, mean_absolute_error


class BackTest():
    def __init__(self, num_sims, save_path):
        self.num_sims = num_sims
        self.boxscores = {}
        self.predictions = {}
        self.save_path = save_path

        # Print settings
        self.print_counter = 0

    def load_data(self):
        df = pd.read_csv('/home/tylerengland/NFL/backend/test/data.csv')
        self.data = df

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
        resutls = {}
        for stat in self.predictions:
            actuals = self.data[stat]
            preds = self.predictions[stat]
            rmse = mean_squared_error(actuals, preds, squared=False)
            mae = mean_absolute_error(actuals, preds)
            
            # Headers and states
            headers = ["Stat", "RMSE", "MAE"]
            data = [stat, str(round(rmse, 2)), str(round(mae, 2))]

            # Set column widths
            preset_widths = [20, 7, 7]
            column_widths = [max(len(header), width) for header, width in zip(headers, preset_widths)]

            # Header row
            header_row = " | ".join(header.ljust(width) for header, width in zip(headers, column_widths))
            data_row = " | ".join(state.ljust(width) for state, width in zip(data, column_widths))

            # Print the header ever 15 plays
            if self.print_counter % 15 == 0:
                print("-" * len(header_row))
                print(header_row)
                print("-" * len(header_row))
            print(data_row)
            self.print_counter += 1

            # Save results
            resutls[stat] = {
                "rmse": rmse,
                "mae": mae
            }
        json.dump(resutls, open(self.save_path, 'w'), indent=4)

    def run(self):
        self.load_data()

        for index, game in self.data.iterrows():
            print(game['game_id'])
            # Extract game data
            home_team = game['home_team']
            away_team = game['away_team']

            # Simulate game and update boxscores
            for _ in range(self.num_sims):
                game_sim = Game(home_team, away_team, verbose=False)
                boxscore = game_sim.simulate_game()
                boxscore['spread'] = boxscore['home_score'] - boxscore['away_score']
                boxscore['total'] = boxscore['home_score'] + boxscore['away_score']
                self.update_boxscores(boxscore)

            # Aggregate boxscores
            self.aggregate_boxscores()
                
        self.evaluate()

if __name__ == '__main__':
    filename = 'test___00001'
    backtest = BackTest(num_sims=1, save_path = f'/home/tylerengland/NFL/backend/test/backtests/{filename}.json')
    backtest.run()
