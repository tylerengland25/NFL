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


def load_data_classifier():
    """
    Loads data and splits into X and Y training and testing sets
    :return:
    """
    y_df = pd.read_csv("backend/data/weekly_stats.csv")
    y_df = y_df.set_index(["Home", "Away", "Week", "Year"])
    y_df["Win_Loss"] = (y_df["H_Score"] - y_df["A_Score"]) > 0
    y_df["Win_Loss"] = y_df["Win_Loss"].astype(int)

    home_X_df = pd.read_csv("backend/data/aggregated_stats.csv")
    home_X_df = home_X_df.set_index(["Home", "Away", "Week", "Year"])

    away_X_df = pd.read_csv("backend/data/aggregated_stats.csv")
    away_X_df = away_X_df.set_index(["Home", "Away", "Week", "Year"])

    X_df = home_X_df.join(away_X_df, rsuffix="_A", lsuffix="_H")
    X_df = X_df.reset_index()
    X_df = X_df[(X_df["Team_H"] == X_df["Home"]) &
                (X_df["Team_A"] == X_df["Away"])]
    X_df = X_df.set_index(["Home", "Away", "Week", "Year"])

    y_cols = ["Win_Loss"]
    X_cols = ['1st', '3rd_Att', '3rd_Cmp', '4th_Att', '4th_Cmp', 'Att',
              'Cmp', 'Fg_Att', 'Fg_Cmp', 'Fum', 'Int', 'Int_Yds', 'Kick_Ret_Yds',
              'Opp_1st', 'Opp_3rd_Att', 'Opp_3rd_Cmp', 'Opp_4th_Att', 'Opp_4th_Cmp',
              'Opp_Att', 'Opp_Cmp', 'Opp_Fg_Att', 'Opp_Fg_Cmp', 'Opp_Fum', 'Opp_Int',
              'Opp_Int_Yds', 'Opp_Kick_Ret_Yds', 'Opp_P_1st', 'Opp_Pass_Yds',
              'Opp_Pen_Yds', 'Opp_Poss', 'Opp_Punt_Ret_Yds', 'Opp_Punt_Yds',
              'Opp_Punts', 'Opp_R_1st', 'Opp_Rush_Ply', 'Opp_Rush_Yds',
              'Opp_Sack_Yds', 'Opp_Sacks', 'Opp_Score', 'Opp_Total_Ply',
              'Opp_Total_Y', 'P_1st', 'Pass_Yds', 'Pen_Yds', 'Poss',
              'Punt_Ret_Yds', 'Punt_Yds', 'Punts', 'R_1st', 'Rush_Ply', 'Rush_Yds',
              'Sack_Yds', 'Sacks', 'Score', 'Total_Ply', 'Total_Y']
    cols = []
    for col in X_cols:
        cols.append(col + "_H")
        cols.append(col + "_A")

    df = X_df.join(y_df)

    # Standardized X values
    X = df[cols]
    X.astype(float)
    scaler = StandardScaler()
    X_standardized = pd.DataFrame(scaler.fit_transform(X))

    # Y values
    y = df[y_cols]

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
    model.add(Dense(16, input_dim=112, kernel_initializer='normal', activation="relu"))
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
    model.add(Dense(16, input_dim=112, kernel_initializer='normal', activation="relu"))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
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
    model.add(Dense(32, input_dim=112, kernel_initializer='normal', activation="relu"))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
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
    model.add(Dense(64, input_dim=112, kernel_initializer='normal', activation="relu"))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
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


if __name__ == '__main__':
    main_classifier()
