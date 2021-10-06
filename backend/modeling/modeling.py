import pandas as pd
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
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, median_absolute_error
from backend.scraping.Game_Stats import convert_poss


def load_data_classifier():
    """
    Loads data and splits into X and Y training and testing sets
    :return:
    """
    df = pd.read_csv("backend/data/weekly_stats.csv")
    df["Win_Loss"] = (df["H_Score"] - df["A_Score"]) > 0
    df["Win_Loss"] = df["Win_Loss"].astype(int)
    df['A_Poss'] = df['A_Poss'].apply(lambda x: convert_poss(x))
    df['H_Poss'] = df['H_Poss'].apply(lambda x: convert_poss(x))

    y_cols = ["Win_Loss"]
    X_cols = ['A_Att', 'A_Cmp', 'A_Fum', 'A_Int', 'A_Pass_Yds', 'A_Rush_Yds', 'A_Total_Ply', 'A_Total_Y',
              'H_Att', 'H_Cmp', 'H_Fum', 'H_Int', 'H_Pass_Yds', 'H_Rush_Yds', 'H_Total_Ply', 'H_Total_Y']
    # ['A_1st', 'A_3rd_Att', 'A_3rd_Cmp', 'A_4th_Att',
    #  'A_4th_Cmp', 'A_Att', 'A_Cmp', 'A_Fg_Att', 'A_Fg_Cmp', 'A_Fum', 'A_Int',
    #  'A_Int_Yds', 'A_Kick_Ret_Yds', 'A_P_1st', 'A_Pass_Yds', 'A_Pen_Yds',
    #  'A_Poss', 'A_Punt_Ret_Yds', 'A_Punt_Yds', 'A_Punts', 'A_R_1st',
    #  'A_Rush_Ply', 'A_Rush_Yds', 'A_Sack_Yds', 'A_Sacks',
    #  'A_Total_Ply', 'A_Total_Y', 'H_1st', 'H_3rd_Att', 'H_3rd_Cmp',
    #  'H_4th_Att', 'H_4th_Cmp', 'H_Att', 'H_Cmp', 'H_Fg_Att', 'H_Fg_Cmp',
    #  'H_Fum', 'H_Int', 'H_Int_Yds', 'H_Kick_Ret_Yds', 'H_P_1st',
    #  'H_Pass_Yds', 'H_Pen_Yds', 'H_Poss', 'H_Punt_Ret_Yds', 'H_Punt_Yds',
    #  'H_Punts', 'H_R_1st', 'H_Rush_Ply', 'H_Rush_Yds', 'H_Sack_Yds',
    #  'H_Sacks', 'H_Total_Ply', 'H_Total_Y']

    # Standardized X values
    X = df[X_cols]
    X.astype(float)
    X_standardized = pd.DataFrame(StandardScaler().fit_transform(X))

    # Y values
    y = df[y_cols]

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=.2, random_state=17)

    return X_train, X_test, y_train, y_test


def load_data_regression_attempts():
    X_df = pd.read_csv('backend/data/aggregated_stats.csv')
    y_df = pd.read_csv('backend/data/weekly_stats.csv')

    X = X_df[["Att", "Total_Ply", "Poss", "Pass_Yds", "Rush_Yds", "Rush_Ply", "Cmp",
              "Opponent", "Home", "Away", "Week", "Year"]]
    X_opp = X_df[["Opp_Att", "Opp_Total_Ply", "Opp_Poss", "Opp_Pass_Yds", "Opp_Rush_Yds", "Opp_Rush_Ply", "Opp_Cmp",
                  "Team", "Home", "Away", "Week", "Year"]]
    X_opp.columns = ["Opp_Def_Att", "Opp_Def_Total_Ply", "Opp_Def_Poss",
                     "Opp_Def_Pass_Yds", "Opp_Def_Rush_Yds", "Opp_Def_Rush_Ply", "Opp_Def_Cmp",
                     "Team", "Home", "Away", "Week", "Year"]

    X = pd.merge(X, X_opp,
                 left_on=["Opponent", "Home", "Away", "Week", "Year"],
                 right_on=["Team", "Home", "Away", "Week", "Year"])

    y = y_df[["H_Att", "A_Att", "Home", "Away", "Week", "Year"]]

    df_merged = pd.merge(y, X, left_on=["Home", "Away", "Week", "Year"], right_on=["Home", "Away", "Week", "Year"])
    df = pd.DataFrame()
    for index, row in df_merged.iterrows():
        new_row = {"Att": row["Att"], "Total_Ply": row["Total_Ply"], "Poss": row["Poss"], "Pass_Yds": row["Pass_Yds"],
                   "Rush_Yds": row["Rush_Yds"], "Rush_Ply": row["Rush_Ply"], "Cmp": row["Cmp"],
                   "Opp_Def_Att": row["Opp_Def_Att"], "Opp_Def_Total_Ply": row["Opp_Def_Total_Ply"],
                   "Opp_Def_Poss": row["Opp_Def_Poss"], "Opp_Def_Pass_Yds": row["Opp_Def_Pass_Yds"],
                   "Opp_Def_Rush_Yds": row["Opp_Def_Rush_Yds"], "Opp_Def_Rush_Ply": row["Opp_Def_Rush_Ply"],
                   "Opp_Def_Cmp": row["Opp_Def_Cmp"]}
        if row["Opponent"] == row["Away"]:
            new_row["y_Att"] = row["H_Att"]
        else:
            new_row["y_Att"] = row["A_Att"]
        df = df.append(new_row, ignore_index=True)

    X = df[["Att", "Total_Ply", "Poss", "Pass_Yds", "Rush_Yds", "Rush_Ply", "Cmp",
            "Opp_Def_Att", "Opp_Def_Total_Ply", "Opp_Def_Poss", "Opp_Def_Pass_Yds",
            "Opp_Def_Rush_Yds", "Opp_Def_Rush_Ply", "Opp_Def_Cmp"]]

    X_standardized = pd.DataFrame(StandardScaler().fit_transform(X))
    y = df[["y_Att"]]

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=.2, random_state=17)

    return X_train, X_test, y_train, y_test


def load_data_regression_completions():
    X_df = pd.read_csv('backend/data/aggregated_stats.csv')
    y_df = pd.read_csv('backend/data/weekly_stats.csv')

    X = X_df[["Att", "Total_Ply", "Poss", "Pass_Yds", "Rush_Yds", "Rush_Ply", "Cmp",
              "Opponent", "Home", "Away", "Week", "Year"]]
    X_opp = X_df[["Opp_Att", "Opp_Total_Ply", "Opp_Poss", "Opp_Pass_Yds", "Opp_Rush_Yds", "Opp_Rush_Ply", "Opp_Cmp",
                  "Team", "Home", "Away", "Week", "Year"]]
    X_opp.columns = ["Opp_Def_Att", "Opp_Def_Total_Ply", "Opp_Def_Poss",
                     "Opp_Def_Pass_Yds", "Opp_Def_Rush_Yds", "Opp_Def_Rush_Ply", "Opp_Def_Cmp",
                     "Team", "Home", "Away", "Week", "Year"]

    X = pd.merge(X, X_opp,
                 left_on=["Opponent", "Home", "Away", "Week", "Year"],
                 right_on=["Team", "Home", "Away", "Week", "Year"])

    y = y_df[["H_Cmp", "A_Cmp", "Home", "Away", "Week", "Year"]]

    df_merged = pd.merge(y, X, left_on=["Home", "Away", "Week", "Year"], right_on=["Home", "Away", "Week", "Year"])
    df = pd.DataFrame()
    for index, row in df_merged.iterrows():
        new_row = {"Att": row["Att"], "Total_Ply": row["Total_Ply"], "Poss": row["Poss"], "Pass_Yds": row["Pass_Yds"],
                   "Rush_Yds": row["Rush_Yds"], "Rush_Ply": row["Rush_Ply"], "Cmp": row["Cmp"],
                   "Opp_Def_Att": row["Opp_Def_Att"], "Opp_Def_Total_Ply": row["Opp_Def_Total_Ply"],
                   "Opp_Def_Poss": row["Opp_Def_Poss"], "Opp_Def_Pass_Yds": row["Opp_Def_Pass_Yds"],
                   "Opp_Def_Rush_Yds": row["Opp_Def_Rush_Yds"], "Opp_Def_Rush_Ply": row["Opp_Def_Rush_Ply"],
                   "Opp_Def_Cmp": row["Opp_Def_Cmp"]}
        if row["Opponent"] == row["Away"]:
            new_row["y_Cmp"] = row["H_Cmp"]
        else:
            new_row["y_Cmp"] = row["A_Cmp"]
        df = df.append(new_row, ignore_index=True)

    X = df[["Att", "Total_Ply", "Poss", "Pass_Yds", "Rush_Yds", "Rush_Ply", "Cmp",
            "Opp_Def_Att", "Opp_Def_Total_Ply", "Opp_Def_Poss", "Opp_Def_Pass_Yds",
            "Opp_Def_Rush_Yds", "Opp_Def_Rush_Ply", "Opp_Def_Cmp"]]

    X_standardized = pd.DataFrame(StandardScaler().fit_transform(X))
    y = df[["y_Cmp"]]

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=.2, random_state=17)

    return X_train, X_test, y_train, y_test


def load_data_regression_fumbles():
    X_df = pd.read_csv('backend/data/aggregated_stats.csv')
    y_df = pd.read_csv('backend/data/weekly_stats.csv')

    X = X_df[["Total_Ply", "Poss", "Rush_Ply", "Att", "Opp_Sacks", "Fum",
              "Opponent", "Home", "Away", "Week", "Year"]]
    X_opp = X_df[["Opp_Total_Ply", "Opp_Poss", "Opp_Rush_Ply", "Opp_Att", "Sacks", "Opp_Fum",
                  "Team", "Home", "Away", "Week", "Year"]]
    X_opp.columns = ["Opp_Def_Total_Ply", "Opp_Def_Poss", "Opp_Def_Rush_Ply", "Opp_Def_Att",
                     "Opp_Def_Sacks", "Opp_Def_Fum",
                     "Team", "Home", "Away", "Week", "Year"]

    X = pd.merge(X, X_opp,
                 left_on=["Opponent", "Home", "Away", "Week", "Year"],
                 right_on=["Team", "Home", "Away", "Week", "Year"])

    y = y_df[["H_Fum", "A_Fum", "Home", "Away", "Week", "Year"]]

    df_merged = pd.merge(y, X, left_on=["Home", "Away", "Week", "Year"], right_on=["Home", "Away", "Week", "Year"])
    df = pd.DataFrame()
    for index, row in df_merged.iterrows():
        new_row = {"Total_Ply": row["Total_Ply"], "Poss": row["Poss"], "Rush_Ply": row["Rush_Ply"], "Att": row["Att"],
                   "Opp_Sacks": row["Opp_Sacks"], "Fum": row["Fum"], "Opp_Def_Total_Ply": row["Opp_Def_Total_Ply"],
                   "Opp_Def_Poss": row["Opp_Def_Poss"], "Opp_Def_Rush_Ply": row["Opp_Def_Rush_Ply"],
                   "Opp_Def_Att": row["Opp_Def_Att"], "Opp_Def_Sacks": row["Opp_Def_Sacks"],
                   "Opp_Def_Fum": row["Opp_Def_Fum"]}
        if row["Opponent"] == row["Away"]:
            new_row["y_Fum"] = row["H_Fum"]
        else:
            new_row["y_Fum"] = row["A_Fum"]
        df = df.append(new_row, ignore_index=True)

    X = df[["Total_Ply", "Poss", "Rush_Ply", "Att", "Opp_Sacks", "Fum", "Opp_Def_Total_Ply", "Opp_Def_Poss",
            "Opp_Def_Rush_Ply", "Opp_Def_Att", "Opp_Def_Sacks", "Opp_Def_Fum"]]

    X_standardized = pd.DataFrame(StandardScaler().fit_transform(X))
    y = df[["y_Fum"]]

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=.2, random_state=17)

    return X_train, X_test, y_train, y_test


def load_data_regression_interceptions():
    X_df = pd.read_csv('backend/data/aggregated_stats.csv')
    y_df = pd.read_csv('backend/data/weekly_stats.csv')

    X = X_df[["Att", "Total_Ply", "Poss", "Pass_Yds", "Cmp", "Int", "Opp_Sacks",
              "Opponent", "Home", "Away", "Week", "Year"]]
    X_opp = X_df[["Opp_Att", "Opp_Total_Ply", "Opp_Poss", "Opp_Pass_Yds", "Opp_Cmp", "Opp_Int", "Sacks",
                  "Team", "Home", "Away", "Week", "Year"]]
    X_opp.columns = ["Opp_Def_Att", "Opp_Def_Total_Ply", "Opp_Def_Poss", "Opp_Def_Pass_Yds", "Opp_Def_Cmp",
                     "Opp_Def_Int", "Opp_Def_Sacks",
                     "Team", "Home", "Away", "Week", "Year"]

    X = pd.merge(X, X_opp,
                 left_on=["Opponent", "Home", "Away", "Week", "Year"],
                 right_on=["Team", "Home", "Away", "Week", "Year"])

    y = y_df[["H_Int", "A_Int", "Home", "Away", "Week", "Year"]]

    df_merged = pd.merge(y, X, left_on=["Home", "Away", "Week", "Year"], right_on=["Home", "Away", "Week", "Year"])
    df = pd.DataFrame()
    for index, row in df_merged.iterrows():
        new_row = {"Att": row["Att"], "Total_Ply": row["Total_Ply"], "Poss": row["Poss"], "Pass_Yds": row["Pass_Yds"],
                   "Cmp": row["Cmp"], "Int": row["Int"], "Opp_Sacks": row["Opp_Sacks"],
                   "Opp_Def_Att": row["Opp_Def_Att"], "Opp_Def_Total_Ply": row["Opp_Def_Total_Ply"],
                   "Opp_Def_Poss": row["Opp_Def_Poss"], "Opp_Def_Pass_Yds": row["Opp_Def_Pass_Yds"],
                   "Opp_Def_Cmp": row["Opp_Def_Cmp"], "Opp_Def_Int": row["Opp_Def_Int"],
                   "Opp_Def_Sacks": row["Opp_Def_Sacks"]}
        if row["Opponent"] == row["Away"]:
            new_row["y_Int"] = row["H_Int"]
        else:
            new_row["y_Int"] = row["A_Int"]
        df = df.append(new_row, ignore_index=True)

    X = df[["Att", "Total_Ply", "Poss", "Pass_Yds", "Cmp", "Int", "Opp_Sacks", "Opp_Def_Att", "Opp_Def_Total_Ply",
            "Opp_Def_Poss", "Opp_Def_Pass_Yds", "Opp_Def_Cmp", "Opp_Def_Int", "Opp_Def_Sacks"]]

    X_standardized = pd.DataFrame(StandardScaler().fit_transform(X))
    y = df[["y_Int"]]

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=.2, random_state=17)

    return X_train, X_test, y_train, y_test


def load_data_regression_passing_yds():
    X_df = pd.read_csv('backend/data/aggregated_stats.csv')
    y_df = pd.read_csv('backend/data/weekly_stats.csv')

    X = X_df[["Att", "Total_Ply", "Poss", "Pass_Yds", "Cmp", "Int", "Opp_Sacks", "Rush_Ply", "Rush_Yds", "P_1st",
              "R_1st",
              "Opponent", "Home", "Away", "Week", "Year"]]
    X_opp = X_df[["Opp_Att", "Opp_Total_Ply", "Opp_Poss", "Opp_Pass_Yds", "Opp_Cmp", "Opp_Int", "Sacks", "Opp_Rush_Ply",
                  "Opp_Rush_Yds", "Opp_P_1st", "Opp_R_1st",
                  "Team", "Home", "Away", "Week", "Year"]]
    X_opp.columns = ["Opp_Def_Att", "Opp_Def_Total_Ply", "Opp_Def_Poss", "Opp_Def_Pass_Yds", "Opp_Def_Cmp",
                     "Opp_Def_Int", "Opp_Def_Sacks", "Opp_Def_Rush_Ply", "Opp_Def_Rush_Yds", "Opp_Def_P_1st",
                     "Opp_Def_R_1st",
                     "Team", "Home", "Away", "Week", "Year"]

    X = pd.merge(X, X_opp,
                 left_on=["Opponent", "Home", "Away", "Week", "Year"],
                 right_on=["Team", "Home", "Away", "Week", "Year"])

    y = y_df[["H_Pass_Yds", "A_Pass_Yds", "Home", "Away", "Week", "Year"]]

    df_merged = pd.merge(y, X, left_on=["Home", "Away", "Week", "Year"], right_on=["Home", "Away", "Week", "Year"])
    df = pd.DataFrame()
    for index, row in df_merged.iterrows():
        new_row = {"Att": row["Att"], "Total_Ply": row["Total_Ply"], "Poss": row["Poss"], "Pass_Yds": row["Pass_Yds"],
                   "Cmp": row["Cmp"], "Int": row["Int"], "Opp_Sacks": row["Opp_Sacks"], "Rush_Ply": row["Rush_Ply"],
                   "Rush_Yds": row["Rush_Yds"], "P_1st": row["P_1st"], "R_1st": row["R_1st"],
                   "Opp_Def_Att": row["Opp_Def_Att"], "Opp_Def_Total_Ply": row["Opp_Def_Total_Ply"],
                   "Opp_Def_Poss": row["Opp_Def_Poss"], "Opp_Def_Pass_Yds": row["Opp_Def_Pass_Yds"],
                   "Opp_Def_Cmp": row["Opp_Def_Cmp"], "Opp_Def_Int": row["Opp_Def_Int"],
                   "Opp_Def_Sacks": row["Opp_Def_Sacks"], "Opp_Def_Rush_Ply": row["Opp_Def_Rush_Ply"],
                   "Opp_Def_Rush_Yds": row["Opp_Def_Rush_Yds"], "Opp_Def_P_1st": row["Opp_Def_P_1st"],
                   "Opp_Def_R_1st": row["Opp_Def_R_1st"]}
        if row["Opponent"] == row["Away"]:
            new_row["y_Pass_Yds"] = row["H_Pass_Yds"]
        else:
            new_row["y_Pass_Yds"] = row["A_Pass_Yds"]
        df = df.append(new_row, ignore_index=True)

    X = df[["Att", "Total_Ply", "Poss", "Pass_Yds", "Cmp", "Int", "Opp_Sacks", "Rush_Ply", "Rush_Yds", "P_1st",
            "R_1st", "Opp_Def_Att", "Opp_Def_Total_Ply", "Opp_Def_Poss", "Opp_Def_Pass_Yds", "Opp_Def_Cmp",
            "Opp_Def_Int", "Opp_Def_Sacks", "Opp_Def_Rush_Ply", "Opp_Def_Rush_Yds", "Opp_Def_P_1st",
            "Opp_Def_R_1st"]]

    X_standardized = pd.DataFrame(StandardScaler().fit_transform(X))
    y = df[["y_Pass_Yds"]]

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=.2, random_state=17)

    return X_train, X_test, y_train, y_test


def load_data_regression_rushing_yds():
    X_df = pd.read_csv('backend/data/aggregated_stats.csv')
    y_df = pd.read_csv('backend/data/weekly_stats.csv')

    X = X_df[["Att", "Total_Ply", "Poss", "Pass_Yds", "Cmp", "Int", "Opp_Sacks", "Rush_Ply", "Rush_Yds", "P_1st",
              "R_1st",
              "Opponent", "Home", "Away", "Week", "Year"]]
    X_opp = X_df[["Opp_Att", "Opp_Total_Ply", "Opp_Poss", "Opp_Pass_Yds", "Opp_Cmp", "Opp_Int", "Sacks", "Opp_Rush_Ply",
                  "Opp_Rush_Yds", "Opp_P_1st", "Opp_R_1st",
                  "Team", "Home", "Away", "Week", "Year"]]
    X_opp.columns = ["Opp_Def_Att", "Opp_Def_Total_Ply", "Opp_Def_Poss", "Opp_Def_Pass_Yds", "Opp_Def_Cmp",
                     "Opp_Def_Int", "Opp_Def_Sacks", "Opp_Def_Rush_Ply", "Opp_Def_Rush_Yds", "Opp_Def_P_1st",
                     "Opp_Def_R_1st",
                     "Team", "Home", "Away", "Week", "Year"]

    X = pd.merge(X, X_opp,
                 left_on=["Opponent", "Home", "Away", "Week", "Year"],
                 right_on=["Team", "Home", "Away", "Week", "Year"])

    y = y_df[["H_Rush_Yds", "A_Rush_Yds", "Home", "Away", "Week", "Year"]]

    df_merged = pd.merge(y, X, left_on=["Home", "Away", "Week", "Year"], right_on=["Home", "Away", "Week", "Year"])
    df = pd.DataFrame()
    for index, row in df_merged.iterrows():
        new_row = {"Att": row["Att"], "Total_Ply": row["Total_Ply"], "Poss": row["Poss"], "Pass_Yds": row["Pass_Yds"],
                   "Cmp": row["Cmp"], "Int": row["Int"], "Opp_Sacks": row["Opp_Sacks"], "Rush_Ply": row["Rush_Ply"],
                   "Rush_Yds": row["Rush_Yds"], "P_1st": row["P_1st"], "R_1st": row["R_1st"],
                   "Opp_Def_Att": row["Opp_Def_Att"], "Opp_Def_Total_Ply": row["Opp_Def_Total_Ply"],
                   "Opp_Def_Poss": row["Opp_Def_Poss"], "Opp_Def_Pass_Yds": row["Opp_Def_Pass_Yds"],
                   "Opp_Def_Cmp": row["Opp_Def_Cmp"], "Opp_Def_Int": row["Opp_Def_Int"],
                   "Opp_Def_Sacks": row["Opp_Def_Sacks"], "Opp_Def_Rush_Ply": row["Opp_Def_Rush_Ply"],
                   "Opp_Def_Rush_Yds": row["Opp_Def_Rush_Yds"], "Opp_Def_P_1st": row["Opp_Def_P_1st"],
                   "Opp_Def_R_1st": row["Opp_Def_R_1st"]}
        if row["Opponent"] == row["Away"]:
            new_row["y_Rush_Yds"] = row["H_Rush_Yds"]
        else:
            new_row["y_Rush_Yds"] = row["A_Rush_Yds"]
        df = df.append(new_row, ignore_index=True)

    X = df[["Att", "Total_Ply", "Poss", "Pass_Yds", "Cmp", "Int", "Opp_Sacks", "Rush_Ply", "Rush_Yds", "P_1st",
            "R_1st", "Opp_Def_Att", "Opp_Def_Total_Ply", "Opp_Def_Poss", "Opp_Def_Pass_Yds", "Opp_Def_Cmp",
            "Opp_Def_Int", "Opp_Def_Sacks", "Opp_Def_Rush_Ply", "Opp_Def_Rush_Yds", "Opp_Def_P_1st",
            "Opp_Def_R_1st"]]

    X_standardized = pd.DataFrame(StandardScaler().fit_transform(X))
    y = df[["y_Rush_Yds"]]

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=.2, random_state=17)

    return X_train, X_test, y_train, y_test


def load_data_regression_total_ply():
    X_df = pd.read_csv('backend/data/aggregated_stats.csv')
    y_df = pd.read_csv('backend/data/weekly_stats.csv')

    X = X_df[["Total_Ply", "Poss",
              "Opponent", "Home", "Away", "Week", "Year"]]
    X_opp = X_df[["Opp_Total_Ply", "Opp_Poss",
                  "Team", "Home", "Away", "Week", "Year"]]
    X_opp.columns = ["Opp_Def_Total_Ply", "Opp_Def_Poss",
                     "Team", "Home", "Away", "Week", "Year"]

    X = pd.merge(X, X_opp,
                 left_on=["Opponent", "Home", "Away", "Week", "Year"],
                 right_on=["Team", "Home", "Away", "Week", "Year"])

    y = y_df[["H_Total_Ply", "A_Total_Ply", "Home", "Away", "Week", "Year"]]

    df_merged = pd.merge(y, X, left_on=["Home", "Away", "Week", "Year"], right_on=["Home", "Away", "Week", "Year"])
    df = pd.DataFrame()
    for index, row in df_merged.iterrows():
        new_row = {"Total_Ply": row["Total_Ply"], "Poss": row["Poss"],
                   "Opp_Def_Total_Ply": row["Opp_Def_Total_Ply"], "Opp_Def_Poss": row["Opp_Def_Poss"]}
        if row["Opponent"] == row["Away"]:
            new_row["y_Total_Ply"] = row["H_Total_Ply"]
        else:
            new_row["y_Total_Ply"] = row["A_Total_Ply"]
        df = df.append(new_row, ignore_index=True)

    X = df[["Total_Ply", "Poss", "Opp_Def_Total_Ply", "Opp_Def_Poss", ]]

    X_standardized = pd.DataFrame(StandardScaler().fit_transform(X))
    y = df[["y_Total_Ply"]]

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=.2, random_state=17)

    return X_train, X_test, y_train, y_test


def load_data_regression_total_yds():
    X_df = pd.read_csv('backend/data/aggregated_stats.csv')
    y_df = pd.read_csv('backend/data/weekly_stats.csv')

    X = X_df[["Att", "Total_Ply", "Poss", "Pass_Yds", "Cmp", "Int", "Opp_Sacks", "Rush_Ply", "Rush_Yds", "P_1st",
              "R_1st",
              "Opponent", "Home", "Away", "Week", "Year"]]
    X_opp = X_df[["Opp_Att", "Opp_Total_Ply", "Opp_Poss", "Opp_Pass_Yds", "Opp_Cmp", "Opp_Int", "Sacks", "Opp_Rush_Ply",
                  "Opp_Rush_Yds", "Opp_P_1st", "Opp_R_1st",
                  "Team", "Home", "Away", "Week", "Year"]]
    X_opp.columns = ["Opp_Def_Att", "Opp_Def_Total_Ply", "Opp_Def_Poss", "Opp_Def_Pass_Yds", "Opp_Def_Cmp",
                     "Opp_Def_Int", "Opp_Def_Sacks", "Opp_Def_Rush_Ply", "Opp_Def_Rush_Yds", "Opp_Def_P_1st",
                     "Opp_Def_R_1st",
                     "Team", "Home", "Away", "Week", "Year"]

    X = pd.merge(X, X_opp,
                 left_on=["Opponent", "Home", "Away", "Week", "Year"],
                 right_on=["Team", "Home", "Away", "Week", "Year"])

    y = y_df[["H_Total_Y", "A_Total_Y", "Home", "Away", "Week", "Year"]]

    df_merged = pd.merge(y, X, left_on=["Home", "Away", "Week", "Year"], right_on=["Home", "Away", "Week", "Year"])
    df = pd.DataFrame()
    for index, row in df_merged.iterrows():
        new_row = {"Att": row["Att"], "Total_Ply": row["Total_Ply"], "Poss": row["Poss"], "Pass_Yds": row["Pass_Yds"],
                   "Cmp": row["Cmp"], "Int": row["Int"], "Opp_Sacks": row["Opp_Sacks"], "Rush_Ply": row["Rush_Ply"],
                   "Rush_Yds": row["Rush_Yds"], "P_1st": row["P_1st"], "R_1st": row["R_1st"],
                   "Opp_Def_Att": row["Opp_Def_Att"], "Opp_Def_Total_Ply": row["Opp_Def_Total_Ply"],
                   "Opp_Def_Poss": row["Opp_Def_Poss"], "Opp_Def_Pass_Yds": row["Opp_Def_Pass_Yds"],
                   "Opp_Def_Cmp": row["Opp_Def_Cmp"], "Opp_Def_Int": row["Opp_Def_Int"],
                   "Opp_Def_Sacks": row["Opp_Def_Sacks"], "Opp_Def_Rush_Ply": row["Opp_Def_Rush_Ply"],
                   "Opp_Def_Rush_Yds": row["Opp_Def_Rush_Yds"], "Opp_Def_P_1st": row["Opp_Def_P_1st"],
                   "Opp_Def_R_1st": row["Opp_Def_R_1st"]}
        if row["Opponent"] == row["Away"]:
            new_row["y_Total_Y"] = row["H_Total_Y"]
        else:
            new_row["y_Total_Y"] = row["A_Total_Y"]
        df = df.append(new_row, ignore_index=True)

    X = df[["Att", "Total_Ply", "Poss", "Pass_Yds", "Cmp", "Int", "Opp_Sacks", "Rush_Ply", "Rush_Yds", "P_1st",
            "R_1st", "Opp_Def_Att", "Opp_Def_Total_Ply", "Opp_Def_Poss", "Opp_Def_Pass_Yds", "Opp_Def_Cmp",
            "Opp_Def_Int", "Opp_Def_Sacks", "Opp_Def_Rush_Ply", "Opp_Def_Rush_Yds", "Opp_Def_P_1st",
            "Opp_Def_R_1st"]]

    X_standardized = pd.DataFrame(StandardScaler().fit_transform(X))
    y = df[["y_Total_Y"]]

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=.2, random_state=17)

    return X_train, X_test, y_train, y_test


def classifier_baseline_nn():
    """
    Creates a fully connected neural network with the input layer having as many nodes as the input, and an
    output layer
    :return:
    """
    # Create model
    model = Sequential()
    model.add(Dense(16, input_dim=16, kernel_initializer='normal', activation="relu"))
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
    model.add(Dense(16, input_dim=16, kernel_initializer='normal', activation="relu"))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
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
    model.add(Dense(16, input_dim=16, kernel_initializer='normal', activation="relu"))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
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
    model.add(Dense(16, input_dim=16, kernel_initializer='normal', activation="relu"))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main_classifier():
    # Load data
    X_train, X_test, y_train, y_test = load_data_classifier()

    # baseline nn
    baseline_model = classifier_baseline_nn()
    baseline_model.fit(X_train, y_train, epochs=50, batch_size=40, verbose=0)
    baseline_predictions = baseline_model.predict(X_test)
    baseline_predictions = [1 if x > 0.5 else 0 for x in baseline_predictions]
    print("Baseline NN report:")
    print(classification_report(y_test, baseline_predictions))

    # deeper nn_1
    nn_1 = classifier_nn_1()
    nn_1.fit(X_train, y_train, epochs=50, batch_size=40, verbose=0)
    nn_1_predictions = nn_1.predict(X_test)
    nn_1_predictions = [1 if x > 0.5 else 0 for x in nn_1_predictions]
    print("Deeper NN_1 report:")
    print(classification_report(y_test, nn_1_predictions))

    # deeper nn_2
    nn_2 = classifier_nn_2()
    nn_2.fit(X_train, y_train, epochs=50, batch_size=40, verbose=0)
    nn_2_predictions = nn_2.predict(X_test)
    nn_2_predictions = [1 if x > 0.5 else 0 for x in nn_2_predictions]
    print("Deeper NN_2 report:")
    print(classification_report(y_test, nn_2_predictions))

    # deeper nn_3
    nn_3 = classifier_nn_3()
    nn_3.fit(X_train, y_train, epochs=50, batch_size=40, verbose=0)
    nn_3_predictions = nn_3.predict(X_test)
    nn_3_predictions = [1 if x > 0.5 else 0 for x in nn_3_predictions]
    print("Deeper NN_3 report:")
    print(classification_report(y_test, nn_3_predictions))

    # Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train.values.ravel())
    gnb_predicitions = gnb.predict(X_test)
    print("Gaussian Navie Bayes:")
    print(classification_report(y_test, gnb_predicitions))

    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train.values.ravel())
    logreg_predicitons = logreg.predict(X_test)
    print("Logistic Regression report:")
    print(classification_report(y_test, logreg_predicitons))

    # Support Vector Machine
    svm = SVC()
    svm.fit(X_train, y_train.values.ravel())
    svm_predictions = svm.predict(X_test)
    print("Support Vector Machine report:")
    print(classification_report(y_test, svm_predictions))

    # Decision Tree
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train.values.ravel())
    tree_predictions = tree.predict(X_test)
    print("Decision Tree report:")
    print(classification_report(y_test, tree_predictions))

    # Random Forest
    forest = RandomForestClassifier()
    forest.fit(X_train, y_train.values.ravel())
    forest_predicions = forest.predict(X_test)
    print("Random Forest report:")
    print(classification_report(y_test, forest_predicions))


def attempts_regression_model():
    X_train, X_test, y_train, y_test = load_data_regression_attempts()

    # Linear regression model
    print("Attempts Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, predictions)))
    print("Mean Squared Error: {}".format(mean_squared_error(y_test, predictions)))
    print("Median Absolute Error: {}".format(median_absolute_error(y_test, predictions)))


def completions_regression_model():
    X_train, X_test, y_train, y_test = load_data_regression_completions()

    # Linear regression model
    print("Completions Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, predictions)))
    print("Mean Squared Error: {}".format(mean_squared_error(y_test, predictions)))
    print("Median Absolute Error: {}".format(median_absolute_error(y_test, predictions)))


def fumbles_regression_model():
    X_train, X_test, y_train, y_test = load_data_regression_fumbles()

    # Linear regression model
    print("Fumbles Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, predictions)))
    print("Mean Squared Error: {}".format(mean_squared_error(y_test, predictions)))
    print("Median Absolute Error: {}".format(median_absolute_error(y_test, predictions)))


def interceptions_regression_model():
    X_train, X_test, y_train, y_test = load_data_regression_interceptions()

    # Linear regression model
    print("Interceptions Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, predictions)))
    print("Mean Squared Error: {}".format(mean_squared_error(y_test, predictions)))
    print("Median Absolute Error: {}".format(median_absolute_error(y_test, predictions)))


def passing_yds_regression_model():
    X_train, X_test, y_train, y_test = load_data_regression_passing_yds()

    # Linear regression model
    print("Passing Yards Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, predictions)))
    print("Mean Squared Error: {}".format(mean_squared_error(y_test, predictions)))
    print("Median Absolute Error: {}".format(median_absolute_error(y_test, predictions)))


def rushing_yds_regression_model():
    X_train, X_test, y_train, y_test = load_data_regression_rushing_yds()

    # Linear regression model
    print("Rushing Yards Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, predictions)))
    print("Mean Squared Error: {}".format(mean_squared_error(y_test, predictions)))
    print("Median Absolute Error: {}".format(median_absolute_error(y_test, predictions)))


def total_ply_regression_model():
    X_train, X_test, y_train, y_test = load_data_regression_total_ply()

    # Linear regression model
    print("Total Plays Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, predictions)))
    print("Mean Squared Error: {}".format(mean_squared_error(y_test, predictions)))
    print("Median Absolute Error: {}".format(median_absolute_error(y_test, predictions)))


def total_yds_regression_model():
    X_train, X_test, y_train, y_test = load_data_regression_total_yds()

    # Linear regression model
    print("Total Yards Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, predictions)))
    print("Mean Squared Error: {}".format(mean_squared_error(y_test, predictions)))
    print("Median Absolute Error: {}".format(median_absolute_error(y_test, predictions)))


if __name__ == '__main__':
    total_yds_regression_model()
