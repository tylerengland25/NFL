import sys
import pandas as pd
import numpy as np
sys.path.append('c:\\Users\\tyler\\OneDrive\\Documents\\Python\\NFL')
from backend.preprocess.preprocess import main as load_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
import warnings
import pickle


def load_odds():
    """
    Function:
        Load odds for every game:
            ~ odds.csv
    
    Input:
        None
        
    Output:
        None
    """
    # Load and clean odds
    odds = pd.read_csv('backend/data/odds/odds.csv')
    odds['home'] = odds['home'].apply(lambda x: ' '.join([word.capitalize() for word in x.split('-')]))
    odds['away'] = odds['away'].apply(lambda x: ' '.join([word.capitalize() for word in x.split('-')]))
    odds.set_index(['home', 'away', 'week', 'year'], inplace=True, drop=True)
    
    # Load scores to merge date
    scores = pd.read_csv('backend/data/games/scores.csv')
    scores['date'] = pd.to_datetime(scores['date'])
    scores['home'] = np.where(scores['home_field'], scores['team'], scores['opponent'])
    scores['away'] = np.where(scores['home_field'], scores['opponent'], scores['team'])
    scores.set_index(['home', 'away', 'week', 'season'], inplace=True, drop=True)

    # Merge odds and date
    odds = pd.merge(odds, scores[['date']], left_index=True, right_index=True)
    odds.reset_index(inplace=True, drop=False)
    odds.set_index(['date', 'home', 'away', 'week', 'season'], inplace=True, drop=True)
    odds = odds.groupby(odds.index).first()
    
    return odds[['ml_h', 'ml_a']]


def risk_management(diff):
    """
    Function:
        Manage risk
    
    Input:
        diff: float
        fav: boolean
        odds: int
        
    Output:
        unit: float
    """
    if diff <= 0:
        return None
    elif 0 < diff <= .05:
        return .25
    elif .05 < diff <= .10:
        return .5
    elif .10 < diff <= .15:
        return 1
    elif .15 < diff <= .20:
        return 1.5
    elif .20 < diff <= .25:
        return 2
    elif .25 < diff <= .30:
        return 2.5
    elif .35 < diff <= .40:
        return 3
    elif .40 < diff:
        return 3.5


def calculate_profit(y_test, y_pred, y_prob):
    """
    Function:
        Calculate profit of algorithm using risk management system.
    
    Input:
        y_test: DataFrame
        y_pred: np.array
        y_prob: np.array
        
    Output:
        None
    """
    # Load odds
    odds = load_odds()

    # Merge y_test
    odds.index = [(index[0].date(), index[1], index[2], index[3], index[4]) for index in odds.index]
    df = pd.merge(y_test, odds, left_index=True, right_index=True, how='left')
    df['y_pred'] = y_pred
    df['y_prob_a'] = [prob[0] for prob in y_prob]
    df['y_prob_h'] = [prob[1] for prob in y_prob]
    df.dropna(axis=0, inplace=True)

    # Calculate profit for every pick
    df['potential'] = np.where(df['y'], df['ml_h'], df['ml_a'])
    df['potential'] = np.where(df['potential'] < 0, -1 / (df['potential'] / 100), 1 * (df['potential'] / 100))
    df['profit'] = np.where(df['y'] == df['y_pred'], df['potential'], -1)

    profit = df['profit'].sum()
    correct = df[df['y'] == df['y_pred']]['y_pred'].count()
    wrong = df[df['y'] != df['y_pred']]['y_pred'].count()

    print(f'\tProfit: {round(profit)} Units')
    print(f'\tAccuracy: {round(correct / (correct + wrong) * 100, 2)}%')
    print(f'\tCount: {correct + wrong}')


    # Calculate profit for risk management
    df['h_fav'] = np.where(df['ml_h'] < 0, 1, 0)
    df['a_fav'] = np.where(df['ml_a'] < 0, 1, 0)
    df['prob_a'] = df['ml_a'].apply(lambda x: abs(x) / (abs(x) + 100) if x < 0 else 100 / (x + 100))
    df['prob_h'] = df['ml_h'].apply(lambda x: abs(x) / (abs(x) + 100) if x < 0 else 100 / (x + 100))
    df['h_diff'] = df['y_prob_h'] - df['prob_h']
    df['a_diff'] = df['y_prob_a'] - df['prob_a']
    df['pick_diff'] = np.where(df['y_pred'], df['h_diff'], df['a_diff'])
    df['pick_fav'] = np.where(df['y_pred'], df['h_fav'], df['a_fav'])
    df['pick_odds'] = np.where(df['y_pred'], df['ml_h'], df['ml_a'])
    df['pick_odds'] = df['pick_odds'].apply(lambda x: 50 * round(x / 50))
    df['pick_diff'] = df['pick_diff'].apply(lambda x: .05 * round(x / .05))
    df['risk_correct'] = np.where(df['y_pred'] == df['y'], 1, 0)
    df['risk_unit'] = df.apply(lambda x: risk_management(x.pick_diff), axis=1)
    df['risk_profit'] = np.where(df['risk_correct'], df['potential'] * df['risk_unit'], -1 * df['risk_unit'])
    
    # Save 2021 to csv
    df_2021 = df[['ml_h', 'ml_a', 'y', 'y_pred', 'risk_unit', 'risk_profit']].copy()
    df_2021.rename({'y': 'outcome', 'y_pred': 'prediction', 'risk_profit': 'profit'}, axis=1, inplace=True)
    df_2021['date'] = [index[0] for index in df_2021.index]
    df_2021['home'] = [index[1] for index in df_2021.index]
    df_2021['away'] = [index[2] for index in df_2021.index]
    df_2021['week'] = [index[3] for index in df_2021.index]
    df_2021['season'] = [index[4] for index in df_2021.index]
    df_2021['outcome'] = np.where(df_2021['outcome'], df_2021['home'], df_2021['away'])
    df_2021['prediction'] = np.where(df_2021['prediction'], df_2021['home'], df_2021['away'])
    df_2021 = df_2021[['date', 'home', 'ml_h', 'away', 'ml_a', 'outcome', 'prediction', 'risk_unit', 'profit', 'week', 'season']]
    df_2021.to_csv('backend/data/predictions/2021_nn.csv', index=False)

    df = df[df['pick_diff'] > 0]
    risk_df = df.groupby(['pick_fav']).aggregate({'risk_profit': 'sum', 'risk_correct': ['sum', 'count']})
    risk_df['accuracy'] = risk_df[('risk_correct', 'sum')] / risk_df[('risk_correct', 'count')]
    risk_df.to_csv('backend/data/risk_management.csv')
    print(f"\tRisk Profit: {round(risk_df[('risk_profit', 'sum')].sum())} Units")
    print(f"\tRisk Accuracy: {round(risk_df[('risk_correct', 'sum')].sum() / risk_df[('risk_correct', 'count')].sum() * 100, 2)}%")
    print(f"\tRisk Count: {risk_df[('risk_correct', 'count')].sum()}")
    print(f'\tBreakdown: \n{risk_df.to_string()}')


def nn():
    """
    Function:
        Create a MLPClassifier using preprocessed data to determine the winner of a game.
        Standardizes data using StandardScaler.
        Selects features using ....
    
    Input:
        None
    
    Output:
        None
    """
    # Load data
    df = load_data()
    
    last_season = df[[index[0].year >= 2021 for index in df.index]]
    df = df[[index[0].year < 2021 for index in df.index]]
    X = df.drop(['y'], axis=1)
    y = df[['y']]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

    # Pipeline
    pipe = Pipeline(
        [
            ('scaler', StandardScaler()),
            (
                'nn', 
                MLPClassifier(
                    random_state=1, 
                    hidden_layer_sizes=(300, 200, 100, ),
                    activation='identity'
                )
            )
        ]
    )

    # Fit and Score
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)

    # Calculate profit
    print(f'\nNN Model: ')
    calculate_profit(y_test, y_pred, y_prob)

    # Calculate season profit
    print(f'\nSeason 2021: ')
    calculate_profit(last_season[['y']], pipe.predict(last_season.drop(['y'], axis=1)), pipe.predict_proba(last_season.drop(['y'], axis=1)))

    # Save model
    with open('modeling/models/nn.pkl','wb') as f:
        pickle.dump(pipe, f)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    nn()