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


def load_data_classifier():
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
    print()
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
    df["win_lose"] = df["H_Score"] - df["A_Score"]
    df["win_lose"] = df["win_lose"] > 0
    y = df["win_lose"].astype(int)

    # X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=.2, random_state=17)

    return X_standardized, y


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
    X, y = load_data_classifier()
    kf = KFold(n_splits=5, shuffle=True, random_state=17)

    # baseline nn
    for train_index, test_index in kf.split(X)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
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

    # # Support Vector Machine
    # svm = SVC(probability=True)
    # svm.fit(X_train, y_train.values.ravel())
    # svm_prob = svm.predict_proba(X_test)
    # svm_prob = np.ndarray.tolist(svm_prob)
    # svm_predictions = [1 if x[1] > .53 else 0 for x in svm_prob]
    # print("Support Vector Machine report:")
    # print(classification_report(y_test, svm_predictions))

    # # Naive Bayes
    # gnb = GaussianNB()
    # gnb.fit(X_train, y_train.values.ravel())
    # gnb_predicitions = gnb.predict(X_test)
    # print("Gaussian Navie Bayes:")
    # print(classification_report(y_test, gnb_predicitions))
    #
    # # Logistic Regression
    # logreg = LogisticRegression()
    # logreg.fit(X_train, y_train.values.ravel())
    # logreg_predicitons = logreg.predict(X_test)
    # print("Logistic Regression report:")
    # print(classification_report(y_test, logreg_predicitons))

    # # Decision Tree
    # tree = DecisionTreeClassifier()
    # tree.fit(X_train, y_train.values.ravel())
    # tree_predictions = tree.predict(X_test)
    # print("Decision Tree report:")
    # print(classification_report(y_test, tree_predictions))
    #
    # # Random Forest
    # forest = RandomForestClassifier()
    # forest.fit(X_train, y_train.values.ravel())
    # forest_predicions = forest.predict(X_test)
    # print("Random Forest report:")
    # print(classification_report(y_test, forest_predicions))


if __name__ == '__main__':
    main_classifier()
