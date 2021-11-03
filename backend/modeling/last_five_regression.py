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


def load_data(stat):
    """
    Loads data and splits into X and Y training and testing sets
    :return:
    """
    x_df = pd.read_csv("backend/data/last_five.csv")
    x_df.fillna(0, inplace=True)
    y_df = pd.read_csv("backend/data/weekly_stats.csv")

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
        elif col[-4:] in ["_1_x", "_1_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * 2.5
        elif col[-4:] in ["_2_x", "_2_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * 2.5
        elif col[-4:] in ["_3_x", "_3_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * .2
        elif col[-4:] in ["_4_x", "_4_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * .2
        elif col[-4:] in ["_5_x", "_5_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * 1

    # Standardized X values
    X = df[x_cols]
    X.astype(float)
    scaler = StandardScaler()
    X_standardized = pd.DataFrame(scaler.fit_transform(X))

    # Y values
    df["win_lose"] = df["H_Score"] - df["A_Score"]
    df["win_lose"] = df["win_lose"] > 0
    y = df[stat].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=.2, random_state=17)

    return X_train, X_test, y_train, y_test


def attempts_regression_model():
    X_train, X_test, y_train, y_test = load_data("H_Att")

    # Linear regression model
    print("Attempts Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = np.ndarray.tolist(predictions)
    df = pd.DataFrame({"predicted": predictions, "actual": y_test.tolist()})
    df["error"] = df["predicted"] - df["actual"]
    df["percent_error"] = (df["error"].divide(df["actual"])).abs()
    print("\tAverage percent error: {}".format(100 * df["percent_error"].mean()))
    print("\tMedian percent error: {}".format(100 * df["percent_error"].median()))
    pickle.dump(model, open("backend/modeling/attempts_model.pkl", "wb"))


def completions_regression_model():
    X_train, X_test, y_train, y_test,  = load_data("H_Cmp")

    # Linear regression model
    print("Completions Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    df = pd.DataFrame({"predicted": predictions, "actual": y_test.tolist()})
    df = pd.DataFrame({"predicted": predictions, "actual": y_test["H_Cmp"].tolist()})
    df["error"] = df["predicted"] - df["actual"]
    df["percent_error"] = (df["error"].divide(df["actual"])).abs()
    print("\tAverage percent error: {}".format(100 * df["percent_error"].mean()))
    print("\tMedian percent error: {}".format(100 * df["percent_error"].median()))
    pickle.dump(model, open("backend/modeling/completions_model.pkl", "wb"))


def fumbles_regression_model():
    X_train, X_test, y_train, y_test,  = load_data("H_Fum")

    # Linear regression model
    print("Fumbles Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    df = pd.DataFrame({"predicted": predictions, "actual": y_test.tolist()})
    df = pd.DataFrame({"predicted": predictions, "actual": y_test["H_Fum"].tolist()})
    df["error"] = df["predicted"] - df["actual"]
    df["percent_error"] = (df["error"].divide(df["actual"])).abs()
    print("\tAverage percent error: {}".format(100 * df["percent_error"].mean()))
    print("\tMedian percent error: {}".format(100 * df["percent_error"].median()))
    pickle.dump(model, open("backend/modeling/fumbles_model.pkl", "wb"))


def interceptions_regression_model():
    X_train, X_test, y_train, y_test,  = load_data("H_Int")

    # Linear regression model
    print("Interceptions Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    df = pd.DataFrame({"predicted": predictions, "actual": y_test.tolist()})
    df = pd.DataFrame({"predicted": predictions, "actual": y_test["H_Int"].tolist()})
    df["error"] = df["predicted"] - df["actual"]
    df["percent_error"] = (df["error"].divide(df["actual"])).abs()
    print("\tAverage percent error: {}".format(100 * df["percent_error"].mean()))
    print("\tMedian percent error: {}".format(100 * df["percent_error"].median()))
    pickle.dump(model, open("backend/modeling/interceptions_model.pkl", "wb"))


def passing_yds_regression_model():
    X_train, X_test, y_train, y_test,  = load_data("H_Pass_Yds")

    # Linear regression model
    print("Passing Yards Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    df = pd.DataFrame({"predicted": predictions, "actual": y_test.tolist()})
    df = pd.DataFrame({"predicted": predictions, "actual": y_test["H_Pass_Yds"].tolist()})
    df["error"] = df["predicted"] - df["actual"]
    df["percent_error"] = (df["error"].divide(df["actual"])).abs()
    print("\tAverage percent error: {}".format(100 * df["percent_error"].mean()))
    print("\tMedian percent error: {}".format(100 * df["percent_error"].median()))
    pickle.dump(model, open("backend/modeling/passing_model.pkl", "wb"))


def rushing_yds_regression_model():
    X_train, X_test, y_train, y_test,  = load_data("H_Rush_Yds")

    # Linear regression model
    print("Rushing Yards Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    df = pd.DataFrame({"predicted": predictions, "actual": y_test.tolist()})
    df = pd.DataFrame({"predicted": predictions, "actual": y_test["H_Rush_Yds"].tolist()})
    df["error"] = df["predicted"] - df["actual"]
    df["percent_error"] = (df["error"].divide(df["actual"])).abs()
    print("\tAverage percent error: {}".format(100 * df["percent_error"].mean()))
    print("\tMedian percent error: {}".format(100 * df["percent_error"].median()))
    pickle.dump(model, open("backend/modeling/rushing_model.pkl", "wb"))


def total_ply_regression_model():
    X_train, X_test, y_train, y_test,  = load_data("H_Total_Ply")

    # Linear regression model
    print("Total Plays Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    df = pd.DataFrame({"predicted": predictions, "actual": y_test.tolist()})
    df = pd.DataFrame({"predicted": predictions, "actual": y_test["H_Total_Ply"].tolist()})
    df["error"] = df["predicted"] - df["actual"]
    df["percent_error"] = (df["error"].divide(df["actual"])).abs()
    print("\tAverage percent error: {}".format(100 * df["percent_error"].mean()))
    print("\tMedian percent error: {}".format(100 * df["percent_error"].median()))
    pickle.dump(model, open("backend/modeling/total_plays_model.pkl", "wb"))


def total_yds_regression_model():
    X_train, X_test, y_train, y_test,  = load_data("H_Total_Yds")

    # Linear regression model
    print("Total Yards Linear Regression model:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    df = pd.DataFrame({"predicted": predictions, "actual": y_test.tolist()})
    df = pd.DataFrame({"predicted": predictions, "actual": y_test["H_Total_Yds"].tolist()})
    df["error"] = df["predicted"] - df["actual"]
    df["percent_error"] = (df["error"].divide(df["actual"])).abs()
    print("\tAverage percent error: {}".format(100 * df["percent_error"].mean()))
    print("\tMedian percent error: {}".format(100 * df["percent_error"].median()))
    pickle.dump(model, open("backend/modeling/total_yards_model.pkl", "wb"))


def main_classifier():
    attempts_regression_model()
    completions_regression_model()
    fumbles_regression_model()
    interceptions_regression_model()
    passing_yds_regression_model()
    rushing_yds_regression_model()
    total_yds_regression_model()
    total_ply_regression_model()


if __name__ == '__main__':
    main_classifier()
