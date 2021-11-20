import random
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from backend.scraping.Game_Stats import convert_poss

import matplotlib.pyplot as plt


def convert_odds(odds):
    if odds < 0:
        odds = odds * -1
        return odds / (100 + odds)
    else:
        return 100 / (100 + odds)


def calc_profit(stake, odds):
    if odds < 0:
        odds = odds * -1
        return stake / (odds / 100)
    else:
        return stake * (odds / 100)


def load_data_classifier():
    """
    Loads data and splits into X and Y training and testing sets
    :return:
    """
    x_df = pd.read_csv("backend/data/last_five.csv")
    x_df.fillna(0, inplace=True)
    y_df = pd.read_csv("backend/data/weekly_stats.csv")
    odds = pd.read_csv("backend/data/odds/odds.csv")

    df = pd.merge(y_df, x_df,
                  how="left",
                  left_on=["Home", "Week", "Year"],
                  right_on=["team", "week", "year"])

    df = pd.merge(df, x_df,
                  how="left",
                  left_on=["Away", "Week", "Year"],
                  right_on=["team", "week", "year"])

    df = df[(df["Week"] > 5) | (df["Year"] != 2010)]

    x_cols = []
    for col in df.columns:
        if "poss" in col:
            df[col] = df[col].apply(lambda x: convert_poss(x) if ":" in x else x)
            x_cols.append(col)
        elif col[-4:] in ["_1_x", "_1_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * 3
        elif col[-4:] in ["_2_x", "_2_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * 3
        elif col[-4:] in ["_3_x", "_3_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * 2

        elif col[-4:] in ["_4_x", "_4_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * 2
        elif col[-4:] in ["_5_x", "_5_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * 2

    # Standardized X values
    # x_cols = ["3rd_att", "3rd_cmp", "3rd_att_def", "3rd_cmp_def", "4th_att", "4th_cmp", "4th_att_def", "4th_cmp_def",
    #           "cmp", "cmp_def", "poss", "total_y", "total_y_def"]
    x_cols = ["cmp_pct", "cmp_pct_def", "3rd_pct", "3rd_pct_def", "4th_pct", "4th_pct_def", "fg_pct", "yds_per_rush",
              "yds_per_rush_def", "yds_per_att", "yds_per_att_def", "yds_per_ply", "yds_per_ply_def", "poss",
              "pass_yds", "pass_yds_def", "pen_yds", "pen_yds_def", "punts", "punts_def", "rush_yds", "rush_yds_def",
              "sacks", "sacks_def", "score", "score_def", "cmp", "cmp_def", "total_y", "total_y_def", "pts_per_ply",
              "pts_per_ply_def", "punts_per_ply", "punts_per_ply_def", "pts_per_yd", "pts_per_yd_def"]

    final_cols = []
    for col in x_cols:
        for i in range(1, 6):
            final_cols.append(col + "_" + str(i) + "_x")
            final_cols.append(col + "_" + str(i) + "_y")
    X = df[final_cols]
    X.astype(float)
    scaler = StandardScaler()
    X_standardized = pd.DataFrame(scaler.fit_transform(X))

    # Y values
    df = pd.merge(df, odds.iloc[:, 1:],
                  how="left",
                  left_on=["Home", "Away", "Week", "Year"],
                  right_on=["Home", "Away", "Week", "Year"])
    df["win_lose"] = df["H_Score"] - df["A_Score"]
    df["win_lose"] = df["win_lose"] > 0
    df["win_lose"] = df["win_lose"].astype(int)
    y = df[["win_lose", "ML_h", "ML_a"]]

    return train_test_split(X_standardized, y, test_size=.2, random_state=17)


def winner_analysis(odds):
    odds["winner_predict_prob"] = np.where(odds["outcome"], odds["home_win_prob"], odds["away_win_prob"])
    odds["prediction"] = odds["winner_predict_prob"].apply(lambda x: round(x, 1))
    odds["winner_actual_prob"] = np.where(odds["outcome"], odds["Home_odds_actual"], odds["Away_odds_actual"])
    odds["winner_ML"] = np.where(odds["outcome"], odds["ML_h"], odds["ML_a"])
    odds["winner_divergence"] = np.where(odds["outcome"], odds["home_divergence"], odds["away_divergence"])
    odds["divergence"] = odds["winner_divergence"].apply(lambda x: round(x, 1))
    odds["winner_payout"] = odds["winner_ML"].apply(lambda x: calc_profit(100, x))
    odds = odds[["prediction", "winner_actual_prob", "divergence", "winner_ML", "winner_payout"]]
    favorites = odds[odds["winner_ML"] < 0]
    underdogs = odds[odds["winner_ML"] > 0]

    favorites = favorites.groupby(["prediction", "divergence"]).agg({"winner_ML": "count", "winner_payout": "sum"})
    underdogs = underdogs.groupby(["prediction", "divergence"]).agg({"winner_ML": "count", "winner_payout": "sum"})
    # print("Winner Favorites:")
    # print(favorites)
    # print("\n\n Winner Underdogs:")
    # print(underdogs)
    return favorites, underdogs


def loser_analysis(odds):
    odds["loser_predict_prob"] = np.where(odds["outcome"], odds["away_win_prob"], odds["home_win_prob"])
    odds["prediction"] = odds["loser_predict_prob"].apply(lambda x: round(x, 1))
    odds["loser_actual_prob"] = np.where(odds["outcome"], odds["Away_odds_actual"], odds["Home_odds_actual"])
    odds["loser_ML"] = np.where(odds["outcome"], odds["ML_a"], odds["ML_h"])
    odds["loser_divergence"] = np.where(odds["outcome"], odds["away_divergence"], odds["home_divergence"])
    odds["divergence"] = odds["loser_divergence"].apply(lambda x: round(x, 1))
    odds["loser_payout"] = -100
    odds = odds[["prediction", "loser_actual_prob", "divergence", "loser_ML", "loser_payout"]]
    favorites = odds[odds["loser_ML"] < 0]
    underdogs = odds[odds["loser_ML"] > 0]
    favorites = favorites.groupby(["prediction", "divergence"]).agg({"loser_ML": "count", "loser_payout": "sum"})
    underdogs = underdogs.groupby(["prediction", "divergence"]).agg({"loser_ML": "count", "loser_payout": "sum"})
    # print("Loser Favorites:")
    # print(favorites)
    # print("\n\nLoser Underdogs:")
    # print(underdogs)
    return favorites, underdogs


def odds_analysis(probabilities, actual_odds, y_test):
    odds = pd.DataFrame(probabilities, columns=["away_win_prob", "home_win_prob"])
    odds = odds.join(actual_odds)
    odds["outcome"] = y_test.reset_index().drop(["index"], axis=1)
    odds["home_divergence"] = odds["home_win_prob"] - odds["Home_odds_actual"]
    odds["away_divergence"] = odds["away_win_prob"] - odds["Away_odds_actual"]
    odds = odds.dropna(axis=0)
    winner_favorites, winner_underdogs = winner_analysis(odds)
    loser_favorites, loser_underdogs = loser_analysis(odds)
    favorites = pd.merge(winner_favorites, loser_favorites,
                         left_on=["prediction", "divergence"], right_on=["prediction", "divergence"])
    favorites["num_hits"] = favorites["winner_ML"] - favorites["loser_ML"]
    favorites["profit"] = favorites["winner_payout"] + favorites["loser_payout"]
    underdogs = pd.merge(winner_underdogs, loser_underdogs,
                         left_on=["prediction", "divergence"], right_on=["prediction", "divergence"])
    underdogs["num_hits"] = underdogs["winner_ML"] - underdogs["loser_ML"]
    underdogs["profit"] = underdogs["winner_payout"] + underdogs["loser_payout"]
    favorites = favorites[favorites["profit"] > 100]
    underdogs = underdogs[underdogs["profit"] > 100]
    print("Favorites:")
    print(favorites)
    print("Underdogs:")
    print(underdogs)
    odds["favorite"] = np.where(odds["ML_h"] < 0, 1, 0)
    odds["underdog"] = np.where(odds["ML_h"] > 0, 1, 0)
    odds["fav_win_prob"] = np.where(odds["ML_h"] < 0,
                                    round(odds["home_win_prob"], 1),
                                    round(odds["away_win_prob"], 1))
    odds["under_win_prob"] = np.where(odds["ML_h"] > 0,
                                      round(odds["home_win_prob"], 1),
                                      round(odds["away_win_prob"], 1))
    odds["fav_div"] = np.where(odds["ML_h"] < 0,
                               round(odds["home_divergence"], 1),
                               round(odds["away_divergence"], 1))
    odds["under_div"] = np.where(odds["ML_h"] > 0,
                                 round(odds["home_divergence"], 1),
                                 round(odds["away_divergence"], 1))
    predicted_outcome = []
    for index, row in odds.iterrows():
        if (row["under_win_prob"], row["under_div"]) in list(underdogs.index):
            predicted_outcome.append(row["underdog"])
        elif (row["fav_win_prob"], row["fav_div"]) in list(favorites.index):
            predicted_outcome.append(row["favorite"])
        else:
            predicted_outcome.append(None)
    odds["predicted_outcome"] = predicted_outcome
    print(list(favorites.index))
    print(list(underdogs.index))
    return odds.dropna(axis=0)


def performance():
    svm = pickle.load(open('backend/modeling/models/svm.pkl', 'rb'))
    # Load data
    X_train, X_test, y_train, y_test = load_data_classifier()
    actual_odds = y_test[["ML_h", "ML_a"]].reset_index().drop(["index"], axis=1)
    actual_odds["Home_odds_actual"] = actual_odds["ML_h"].apply(lambda x: convert_odds(x))
    actual_odds["Away_odds_actual"] = actual_odds["ML_a"].apply(lambda x: convert_odds(x))
    y_test = y_test["win_lose"]
    svm_prob = svm.predict_proba(X_test)
    bets = odds_analysis(svm_prob, actual_odds, y_test)
    bets["potential_payout"] = np.where(bets["predicted_outcome"],
                                        bets["ML_h"].apply(lambda x: calc_profit(100, x)),
                                        bets["ML_a"].apply(lambda x: calc_profit(100, x)))
    bets["payout"] = np.where(bets["outcome"] == bets["predicted_outcome"], bets["potential_payout"], -100)
    num_hit = bets[bets["outcome"] == bets["predicted_outcome"]]["outcome"].count()
    num_placed = bets["outcome"].count()
    profit = bets["payout"].sum()
    print("Performance:")
    print("\tBets hit: {}".format(num_hit))
    print("\tBets placed: {}".format(num_placed))
    print("\tAccuracy: {}%".format(round(num_hit / num_placed * 100)))
    print("\tProfit: ${}".format(round(profit)))


def main_classifier():
    # Load data
    X_train, X_test, y_train, y_test = load_data_classifier()
    y_train = y_train["win_lose"]
    actual_odds = y_test[["ML_h", "ML_a"]].reset_index().drop(["index"], axis=1)
    actual_odds["Home_odds_actual"] = actual_odds["ML_h"].apply(lambda x: convert_odds(x))
    actual_odds["Away_odds_actual"] = actual_odds["ML_a"].apply(lambda x: convert_odds(x))
    y_test = y_test["win_lose"]

    # Support Vector Machine
    svm = SVC(probability=True)
    svm.fit(X_train, y_train.values.ravel())
    svm_predictions = svm.predict(X_test)
    print("Support Vector Machine report:")
    print(classification_report(y_test, svm_predictions))
    pickle.dump(svm, open('backend/modeling/models/svm.pkl', 'wb'))


if __name__ == '__main__':
    performance()
