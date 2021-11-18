import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from backend.scraping.Game_Stats import convert_poss


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


def odds_calculations(probabilities, actual_odds, y_test):
    odds = pd.DataFrame(probabilities, columns=["Away_odds_predict", "Home_odds_predict"])
    odds = odds.join(actual_odds)
    odds["outcome"] = y_test.reset_index().drop(["index"], axis=1)
    odds["predicted_winner"] = np.where(odds["Home_odds_predict"] >= odds["Away_odds_predict"],
                                        odds["Home_odds_predict"],
                                        odds["Away_odds_predict"])
    odds["odds_actual"] = np.where(odds["Home_odds_predict"] >= odds["Away_odds_predict"],
                                   odds["Home_odds_actual"],
                                   odds["Away_odds_actual"])
    odds["ML"] = np.where(odds["Home_odds_predict"] >= odds["Away_odds_predict"],
                          odds["ML_h"],
                          odds["ML_a"])
    odds["potential_payout"] = odds["ML"].apply(lambda x: calc_profit(100, x))
    odds["divergence"] = odds["predicted_winner"] - odds["odds_actual"]
    odds["place_bet"] = (odds["predicted_winner"] >= 0.7) | (odds["divergence"] >= 0.1)
    odds = odds[odds["place_bet"]]
    odds["payout"] = np.where(odds["outcome"] == 1, odds["potential_payout"], -100)
    profit = odds["payout"].sum()
    print("SVM result:")
    print("\tBets placed: {}".format(odds.shape[0]))
    print("\tBets hit: {}".format(odds["outcome"].sum()))
    print("\tAccuracy: {}%".format(round(odds["outcome"].sum() / odds.shape[0] * 100)))
    print("\tProfit: ${}".format(profit))


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
    svm_prob = svm.predict_proba(X_test)
    odds_calculations(svm_prob, actual_odds, y_test)


if __name__ == '__main__':
    main_classifier()
