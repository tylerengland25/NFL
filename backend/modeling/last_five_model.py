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
    y = df[["win_lose", "ML_h", "ML_a", "Year", "Week"]]

    return train_test_split(X_standardized, y, test_size=.2, random_state=17)


def performance():
    svm = pickle.load(open('backend/modeling/models/svm.pkl', 'rb'))
    # Load data
    X_train, X_test, y_train, y_test = load_data_classifier()
    # Predict
    predictions = pd.DataFrame(svm.predict_proba(X_test), columns=["away_predict_prob", "home_predict_prob"])
    # Join Vegas odds
    odds = pd.merge(predictions, y_test.reset_index().drop("index", axis=1), left_index=True, right_index=True)
    odds = odds.rename(columns={"win_lose": "outcome"})
    odds = odds.dropna(axis=0).reset_index().drop(["index"], axis=1)
    # Feature engineer probabilities, potential payouts, and divergences
    odds["away_actual_prob"] = odds["ML_a"].apply(lambda x: convert_odds(x))
    odds["home_actual_prob"] = odds["ML_h"].apply(lambda x: convert_odds(x))
    odds["away_divergence"] = odds["away_predict_prob"] - odds["away_actual_prob"]
    odds["away_divergence"] = odds["away_divergence"].apply(lambda x: round(x, 1))
    odds["home_divergence"] = odds["home_predict_prob"] - odds["home_actual_prob"]
    odds["home_divergence"] = odds["home_divergence"].apply(lambda x: round(x, 1))
    odds["home_predict_prob"] = odds["home_predict_prob"].apply(lambda x: round(x, 1))
    odds["away_predict_prob"] = odds["away_predict_prob"].apply(lambda x: round(x, 1))
    odds["potential_payout"] = np.where(odds["outcome"],
                                        odds["ML_h"].apply(lambda x: calc_profit(100, x)),
                                        odds["ML_a"].apply(lambda x: calc_profit(100, x)))
    # Exploratory analysis
    home_index, away_index = exploratory_analysis(odds)

    # Predict outcome
    odds["predict_outcome"] = predict(odds)
    odds = odds.dropna(axis=0)
    # Calculate payout
    odds["potential_payout"] = np.where(odds["outcome"],
                                        odds["ML_h"].apply(lambda x: calc_profit(100, x)),
                                        odds["ML_a"].apply(lambda x: calc_profit(100, x)))
    odds["payout"] = np.where(odds["predict_outcome"] == odds["outcome"], odds["potential_payout"], -100)
    performance_by_week = print_performance(odds)
    return performance_by_week


def predict(odds):
    # Predict outcome
    predict_outcome = []
    for index, row in odds.iterrows():
        if (.6 <= row["home_predict_prob"] < .7) and (-.2 <= row["home_divergence"] < .2):
            predict_outcome.append(1)
        elif (.6 <= row["away_predict_prob"] < .7) and (-.2 <= row["away_divergence"] < .1):
            predict_outcome.append(0)
        elif (.5 <= row["home_predict_prob"] < .6) and (-.05 <= row["home_divergence"] < .1):
            predict_outcome.append(1)
        elif (.5 <= row["away_predict_prob"] < .6) and (-.3 <= row["away_divergence"] < .4):
            predict_outcome.append(0)
        elif (.4 <= row["home_predict_prob"] < .5) and (-.2 <= row["home_divergence"] < .1):
            predict_outcome.append(1)
        elif (.4 <= row["away_predict_prob"] < .5) and (0 <= row["home_divergence"]):
            predict_outcome.append(0)
        elif (.2 <= row["home_predict_prob"] < .4) and (-.1 <= row["home_divergence"]):
            predict_outcome.append(1)
        elif (.2 <= row["away_predict_prob"] < .4) and (-.1 <= row["away_divergence"] < .1):
            predict_outcome.append(0)
        else:
            predict_outcome.append(None)
    return predict_outcome


def exploratory_analysis(odds):
    odds["home_payout"] = np.where(odds["outcome"], odds["potential_payout"], -100)
    odds["away_payout"] = np.where(odds["outcome"], -100, odds["potential_payout"])
    # Home analysis
    home = odds.groupby(["home_divergence", "home_predict_prob"]).agg({"outcome": ["sum", "count"],
                                                                       "home_payout": ["sum"]})
    home["perc"] = home[('outcome', 'sum')] / home[('outcome', 'count')]
    home = home[(home[('home_payout', 'sum')] > 0)]
    # Away analysis
    away = odds.groupby(["away_divergence", "away_predict_prob"]).agg({"outcome": ["sum", "count"],
                                                                       "away_payout": ["sum"]})
    away["perc"] = (away[('outcome', 'count')] - away[('outcome', 'sum')]) / away[('outcome', 'count')]
    away = away[(away[('away_payout', 'sum')] > 0)]

    return home.index, away.index


def print_performance(odds):
    hits = odds[odds["outcome"] == odds["predict_outcome"]]["outcome"].count()
    total = odds["outcome"].count()
    profit = odds["payout"].sum()
    print("Performance:")
    print("\tBets hit: {}".format(hits))
    print("\tTotal bets: {}".format(total))
    print("\tAccuracy: {}%".format(round(hits / total * 100)))
    print("\tProfit: ${}".format(round(profit)))
    print("Performance by week:")
    print(odds.groupby(["Week"])["payout"].sum())
    return odds.groupby(["Week"])["payout"].sum()


def main_classifier():
    # Load data
    X_train, X_test, y_train, y_test = load_data_classifier()
    y_train = y_train["win_lose"]
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
