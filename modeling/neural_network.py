import sys
import pandas as pd
import numpy as np
from datetime import date
sys.path.append('c:\\Users\\tyler\\OneDrive\\Documents\\Python\\NFL')
from backend.preprocess.preprocess import main as load_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
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
    odds.rename({'year': 'season'}, axis=1, inplace=True)
    odds.set_index(['home', 'away', 'week', 'season'], inplace=True, drop=True)
    
    # Load scores to merge date
    scores = pd.read_csv('backend/data/games/scores.csv')
    scores['date'] = pd.to_datetime(scores['date'])
    scores['home'] = np.where(scores['home_field'], scores['team'], scores['opponent'])
    scores['away'] = np.where(scores['home_field'], scores['opponent'], scores['team'])
    scores.set_index(['home', 'away', 'week', 'season'], inplace=True, drop=True)

    # Merge odds and date
    odds = pd.merge(odds, scores[['date']], left_index=True, right_index=True)
    odds.reset_index(inplace=True, drop=False)
    odds.drop_duplicates(['date', 'home', 'away', 'week', 'season'], inplace=True)
    odds.set_index(['date', 'home', 'away', 'week', 'season'], inplace=True, drop=True)
    
    return odds[['ml_h', 'ml_a']]


def risk_management(diff, odds):
    """
    Function:
        Manage risk
    
    Input:
        diff: float
        
    Output:
        unit: float
    """
    if -.20 < diff <= .20 :
        return abs(diff) * abs(odds) / 100
    else:
        return None


def calculate_profit(y_test, y_pred, y_prob, scores):
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
    df = pd.merge(y_test, odds, left_index=True, right_index=True, how='left')
    df['y_pred'] = y_pred
    df['y_prob_a'] = [prob[0] for prob in y_prob]
    df['y_prob_h'] = [prob[1] for prob in y_prob]
    df.dropna(axis=0, inplace=True)

    # Every Pick
    df['potential'] = np.where(df['y'], df['ml_h'], df['ml_a'])
    df['potential'] = np.where(df['potential'] < 0, -1 / (df['potential'] / 100), 1 * (df['potential'] / 100))
    df['profit'] = np.where(df['y'] == df['y_pred'], df['potential'], -1)
    scores['profit'] = df['profit'].sum()
    scores['hit'] = df[df['y'] == df['y_pred']]['y_pred'].count()
    scores['placed'] = df['y'].count()

    # Risk Management
    df['h_fav'] = np.where(df['ml_h'] < 0, 1, 0)
    df['a_fav'] = np.where(df['ml_a'] < 0, 1, 0)
    df['prob_a'] = df['ml_a'].apply(lambda x: abs(x) / (abs(x) + 100) if x < 0 else 100 / (x + 100))
    df['prob_h'] = df['ml_h'].apply(lambda x: abs(x) / (abs(x) + 100) if x < 0 else 100 / (x + 100))
    df['h_diff'] = df['y_prob_h'] - df['prob_h']
    df['a_diff'] = df['y_prob_a'] - df['prob_a']
    df['pick_diff'] = np.where(df['y_pred'], df['h_diff'], df['a_diff'])
    df['pick_fav'] = np.where(df['y_pred'], df['h_fav'], df['a_fav'])
    df['pick_odds'] = np.where(df['y_pred'], df['ml_h'], df['ml_a'])
    df['risk_correct'] = np.where(df['y_pred'] == df['y'], 1, 0)
    df['risk_unit'] = df.apply(lambda x: risk_management(x.pick_diff, x.pick_odds), axis=1)
    df['risk_profit'] = np.where(df['risk_correct'], df['potential'] * df['risk_unit'], -1 * df['risk_unit'])
    df.dropna(subset=['risk_unit'], axis=0, inplace=True)
    scores['profit_risk'] = df['risk_profit'].sum()
    scores['hit_risk'] = df['risk_correct'].sum()
    scores['placed_risk'] = df['risk_correct'].count()
    df.to_csv('backend/data/predictions/2021_nn.csv')


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
    df = pd.read_csv('c:\\Users\\tyler\\OneDrive\\Documents\\Python\\NFL\\backend\\preprocess\\preprocess.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index(['date', 'home', 'away', 'week', 'season'], inplace=True)
    
    X = df.drop(['y'], axis=1)
    y = df[['y']]

    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
    X_train = X[X.index.get_level_values(4) < 2021]
    X_test = X[X.index.get_level_values(4) >= 2021]
    y_train = y[y.index.get_level_values(4) < 2021]
    y_test = y[y.index.get_level_values(4) >= 2021]

    # Pipeline
    pipe = Pipeline(
        [
            ('scaler', StandardScaler()),
            (
                'nn', 
                MLPClassifier(
                    random_state=1, 
                    hidden_layer_sizes=(200,),
                    activation='tanh',
                )
            )
        ]
    )

    # Fit and Score
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)

    # Calculate accuracy and profit
    k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
    print(f'NN Model: ')
    print(f'\tAccuracy:\n\t\t{round(pipe.score(X_test, y_test) * 100)}%')
    scores = cross_val_score(pipe, X_test, y_test, cv=k_fold)
    print(
        f"""\tk-Fold (5):
        \tMean: {round(scores.mean() * 100)}%
        \tStd: {round(scores.std() * 100)}%"""
    )
    scores = {
        'profit': None, 'hit': None, 'placed': None, 
        'profit_risk': None, 'hit_risk': None, 'placed_risk': None
    }
    calculate_profit(y_test, y_pred, y_prob, scores)
    print(
        f"""\tEvery Bet:
        \tProfit: {round(scores['profit'], 2)}u 
        \tAccuracy: {round(scores['hit'] / scores['placed'] * 100)}%
        \tPlaced: {round(scores['placed'])}"""
    )
    print(
        f"""\tRisk Management:
        \tProfit: {round(scores['profit_risk'], 2)}u 
        \tAccuracy: {round(scores['hit_risk'] / scores['placed_risk'] * 100)}%
        \tPlaced: {round(scores['placed_risk'])} ({round(scores['placed_risk'] / scores['placed'] * 100)}%)"""
    )

    # Save model
    with open('modeling/models/nn.pkl','wb') as f:
        pickle.dump(pipe, f)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    nn()