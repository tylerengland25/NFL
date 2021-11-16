import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from backend.scraping.Game_Stats import convert_poss

from sklearn.model_selection import KFold


def convert_odds(odds):
    if odds < 0:
        odds = odds * -1
        return odds/(100 + odds)
    else:
        return 100/(100 + odds)


def calc_profit(stake, odds):
    if odds < 0:
        odds = odds * -1
        return stake/(odds/100)
    else:
        return stake * (odds/100)


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
    df = pd.merge(df, odds,
                  how="left",
                  left_on=["Home", "Away", "Week", "Year"],
                  right_on=["Home", "Away", "Week", "Year"])
    df["win_lose"] = df["H_Score"] - df["A_Score"]
    df["win_lose"] = df["win_lose"] > 0
    df["win_lose"] = df["win_lose"].astype(int)
    y = df[["win_lose", "ML_h", "ML_a"]]

    return train_test_split(X_standardized, y, test_size=.2, random_state=17)


def classifier_baseline_nn():
    """
    Creates a fully connected neural network with the input layer having as many nodes as the input, and an
    output layer
    :return:
    """
    # Create model
    model = Sequential()
    model.add(Dense(360, input_dim=360, kernel_initializer='normal', activation="relu"))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def classifier_nn_1():
    """
        Creates a fully connected neural network with the three hidden layers
        :return:
        """
    # Create model
    model = Sequential()
    model.add(Dense(360, input_dim=360, kernel_initializer='normal', activation="relu"))
    model.add(Dense(360, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def classifier_nn_2():
    """
    Creates a fully connected neural network with the three hidden layers
    :return:
    """
    # Create model
    model = Sequential()
    model.add(Dense(360, input_dim=360, kernel_initializer='normal', activation="relu"))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def classifier_nn_3():
    """
    Creates a fully connected neural network with the three hidden layers
    :return:
    """
    # Create model
    model = Sequential()
    model.add(Dense(360, input_dim=360, kernel_initializer='normal', activation="relu"))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main_classifier():
    # Load data
    X_train, X_test, y_train, y_test = load_data_classifier()
    y_train = y_train["win_lose"]
    actual_odds = y_test[["ML_h", "ML_a"]].reset_index().drop(["index"], axis=1)
    actual_odds["Home_odds_actual"] = actual_odds["ML_h"].apply(lambda x: convert_odds(x))
    actual_odds["Away_odds_actual"] = actual_odds["ML_a"].apply(lambda x: convert_odds(x))
    y_test = y_test["win_lose"]

    # # baseline nn
    # accuracy = []
    # loss = []
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #     baseline_model = classifier_baseline_nn()
    #     baseline_model.fit(X_train, y_train, epochs=50, batch_size=40, verbose=0)
    #     scores = baseline_model.evaluate(X_test, y_test)
    #     loss.append(scores[0])
    #     accuracy.append(scores[1])
    # print("Baseline NN report:")
    # print("\tAvg loss: {}".format(sum(loss)/len(loss)))
    # print("\tAvg accuracy: {}".format(round(sum(accuracy) / len(accuracy), 2) * 100))
    #
    # # deeper nn_1
    # accuracy = []
    # loss = []
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #     baseline_model = classifier_nn_1()
    #     baseline_model.fit(X_train, y_train, epochs=50, batch_size=40, verbose=0)
    #     scores = baseline_model.evaluate(X_test, y_test)
    #     loss.append(scores[0])
    #     accuracy.append(scores[1])
    # print("NN 1 report:")
    # print("\tAvg loss: {}".format(sum(loss) / len(loss)))
    # print("\tAvg accuracy: {}".format(round(sum(accuracy) / len(accuracy), 2) * 100))
    #
    # # deeper nn_2
    # accuracy = []
    # loss = []
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #     baseline_model = classifier_nn_2()
    #     baseline_model.fit(X_train, y_train, epochs=50, batch_size=40, verbose=0)
    #     scores = baseline_model.evaluate(X_test, y_test)
    #     loss.append(scores[0])
    #     accuracy.append(scores[1])
    # print("NN 2 report:")
    # print("\tAvg loss: {}".format(sum(loss) / len(loss)))
    # print("\tAvg accuracy: {}".format(round(sum(accuracy) / len(accuracy), 2) * 100))
    #
    # # deeper nn_3
    # accuracy = []
    # loss = []
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #     baseline_model = classifier_nn_3()
    #     baseline_model.fit(X_train, y_train, epochs=50, batch_size=40, verbose=0)
    #     scores = baseline_model.evaluate(X_test, y_test)
    #     loss.append(scores[0])
    #     accuracy.append(scores[1])
    # print("NN 3 report:")
    # print("\tAvg loss: {}".format(sum(loss) / len(loss)))
    # print("\tAvg accuracy: {}".format(round(sum(accuracy) / len(accuracy), 2) * 100))

    # Support Vector Machine
    svm = SVC(probability=True)
    svm.fit(X_train, y_train.values.ravel())
    svm_predictions = svm.predict(X_test)
    print("Support Vector Machine report:")
    print(classification_report(y_test, svm_predictions))
    svm_prob = svm.predict_proba(X_test)
    svm_odds = pd.DataFrame(svm_prob, columns=["Away_odds_predict", "Home_odds_predict"])
    svm_odds = svm_odds.join(actual_odds)
    svm_odds["outcome"] = y_test.reset_index().drop(["index"], axis=1)
    # Straight
    svm_odds["my_bet_straight"] = svm_odds["Home_odds_predict"] >= svm_odds["Away_odds_predict"]
    svm_odds["my_bet_straight"] = svm_odds["my_bet_straight"].astype(int)
    svm_odds["result_straight"] = svm_odds["my_bet_straight"] == svm_odds["outcome"]
    svm_odds["result_straight"] = svm_odds["result_straight"].astype(int)
    # Divergence
    svm_odds["home_odds_diff"] = svm_odds["Home_odds_predict"] - svm_odds["Home_odds_actual"]
    svm_odds["away_odds_diff"] = svm_odds["Away_odds_predict"] - svm_odds["Away_odds_actual"]
    svm_odds["my_bet_divergence"] = svm_odds["home_odds_diff"] >= svm_odds["away_odds_diff"]
    svm_odds["my_bet_divergence"] = svm_odds["my_bet_divergence"].astype(int)
    svm_odds["result_divergence"] = svm_odds["my_bet_divergence"] == svm_odds["outcome"]
    svm_odds["result_divergence"] = svm_odds["result_divergence"].astype(int)
    # Print result
    print("Support Vector Machine betting report:")
    print("\tDivergence betting:")
    print("\t\tBets that hit: {}".format(svm_odds["result_divergence"].sum()))
    print("\t\tNumber of bets: {}".format(svm_odds["result_divergence"].count()))
    print("\t\tAccuracy: {}%".format(
        round(
            svm_odds["result_divergence"].sum() / svm_odds["result_divergence"].count() * 100
        )
    ))
    print("\tStraight betting:")
    print("\t\tBets that hit: {}".format(svm_odds["result_straight"].sum()))
    print("\t\tNumber of bets: {}".format(svm_odds["result_straight"].count()))
    print("\t\tAccuracy: {}%".format(
        round(
            svm_odds["result_straight"].sum() / svm_odds["result_straight"].count() * 100
        )
    ))


if __name__ == '__main__':
    main_classifier()
