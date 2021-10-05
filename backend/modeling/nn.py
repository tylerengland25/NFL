import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support


def load_data_classifier():
    """
    Loads data and splits into X and Y training and testing sets
    :return:
    """
    df = pd.read_csv("backend/data/inputs.csv")
    df["Win_Loss"] = (df["H_Score"] - df["A_Score"]) > 0
    df["Win_Loss"] = df["Win_Loss"].astype(int)
    y_cols = ["Win_Loss"]
    X_cols = {'H_pts_per_game_scored', 'H_pts_per_game_allowed', 'A_pts_per_game_allowed', 'A_pts_per_game_scored',
              'H_yds_per_game_gained', 'H_yds_per_game_allowed', 'A_yds_per_game_gained', 'A_yds_per_game_allowed',
              'H_yds_per_ply_gained', 'H_yds_per_ply_allowed', 'A_yds_per_ply_gained', 'A_yds_per_ply_allowed'}
    # 'H_yds_per_run_gained', 'A_yds_per_point_allowed', 'H_yds_per_ply_gained',
    # 'A_yds_per_ply_allowed', 'A_yds_per_game_gained', 'A_pts_per_game_allowed', 'A_sacks_per_ply',
    # 'H_int_per_ply_thrown', 'A_pass_run_ratio', 'H_yds_per_point_scored', 'A_int_per_ply_thrown',
    # 'H_int_per_ply', 'H_yds_per_run_allowed', 'A_yds_per_run_allowed', 'A_yds_per_game_allowed',
    # 'H_sacks_per_ply_allowed', 'A_yds_per_point_scored', 'H_yds_per_game_gained', 'A_yds_per_pass_allowed',
    # 'H_yds_per_pass_gained', 'H_yds_per_ply_allowed', 'A_yds_per_pass_gained', 'H_pass_run_ratio',
    # 'H_yds_per_point_allowed', 'A_yds_per_run_gained', 'H_sacks_per_ply', 'A_pts_per_game_scored',
    # 'H_pts_per_game_scored', 'H_yds_per_pass_allowed', 'H_yds_per_game_allowed', 'A_yds_per_ply_gained',
    # 'H_pts_per_game_allowed', 'A_int_per_Ply', 'A_sacks_per_ply_allowed'

    # Standardized X values
    X = df[X_cols].values
    bad_indices = np.where(np.isinf(X))
    X[bad_indices] = 10
    X_standardized = pd.DataFrame(StandardScaler().fit_transform(X))

    # Y values
    y = df[y_cols].values

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=.2, random_state=17)

    return X_train, X_test, y_train, y_test


def load_data_regression():
    """
    Loads data and splits into X and Y training and testing sets
    :return:
    """
    df = pd.read_csv("backend/data/inputs.csv")
    df = df[df["Week"] >= 2]
    y_cols = {"H_Score", "A_Score"}
    X_cols = set(df.columns).difference({"H_Score", "A_Score", "Home",
                                         "Away", "Week", "Year", "Unnamed: 0", "index"})

    # Standardized X values
    X = df[X_cols].values
    bad_indices = np.where(np.isinf(X))
    X[bad_indices] = 10
    X_standardized = pd.DataFrame(StandardScaler().fit_transform(X))

    # Y values
    y = df[y_cols].values

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=.2, random_state=17)

    return X_train, X_test, y_train, y_test


def baseline_nn():
    """
    Creates a fully connected neural network with the input layer having as many nodes as the input, and an
    output layer
    :return:
    """
    # Create model
    model = Sequential()
    model.add(Dense(34, input_dim=34, kernel_initializer='normal', activation="relu"))
    model.add(Dense(2, activation="linear"))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics='mean_squared_error')
    return model


def wider_nn():
    """
    Creates a fully connected neural network with the input layer having double as many nodes as the input, and an
    output layer
    :return:
    """
    # Create model
    model = Sequential()
    model.add(Dense(68, input_dim=34, kernel_initializer='normal', activation="relu"))
    model.add(Dense(2, activation="linear"))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics='mean_squared_error')
    return model


def deeper_nn_1():
    """
    Creates a fully connected neural network with the three hidden layers
    :return:
    """
    # Create model
    model = Sequential()
    model.add(Dense(64, input_dim=34, kernel_initializer='normal', activation="relu"))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, activation="linear"))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics='mean_squared_error')
    return model


def main_1():
    # Load datasets
    X_train, X_test, y_train, y_test = load_data_regression()

    # Hyper-parameters for NN
    epochs = 50
    batch_size = 40

    # Fit baseline model
    baseline_model = baseline_nn()
    baseline_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Fit wider model
    wider_model = wider_nn()
    wider_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Fit deeper_nn_1
    deeper_model_1 = deeper_nn_1()
    deeper_model_1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Evaluate models
    baseline_loss, baseline_metrics = baseline_model.evaluate(X_test, y_test)
    print('Baseline Loss (MSE): %.2f\tBaseline Metrics (MSE): %.2f' % (baseline_loss, baseline_metrics))
    wider_loss, wider_metrics = wider_model.evaluate(X_test, y_test)
    print('Wider Loss (MSE): %.2f\tWider Metrics (MSE): %.2f' % (wider_loss, wider_metrics))
    deeper_loss_1, deeper_metrics_1 = deeper_model_1.evaluate(X_test, y_test)
    print('Deeper Loss (MSE): %.2f\tDeeper Metrics (MSE): %.2f' % (deeper_loss_1, deeper_metrics_1))

    # Predict on test set, [(Home_score, Away_score), ...]
    baseline_predict_table = pd.DataFrame()
    wider_predict_table = pd.DataFrame()
    baseline_predictions = baseline_model.predict(X_test)
    wider_predictions = wider_model.predict(X_test)
    for baseline, wider, actual in zip(baseline_predictions, wider_predictions, y_test.tolist()):
        baseline_row = {'Home_Score_Predict': baseline[0],
                        'Home_score': actual[0],
                        'Away_Score_Predict': baseline[1],
                        'Away_Score': actual[1]}
        wider_row = {'Home_Score_Predict': wider[0],
                     'Home_score': actual[0],
                     'Away_Score_Predict': wider[1],
                     'Away_Score': actual[1]}

        baseline_predict_table = baseline_predict_table.append(baseline_row, ignore_index=True)
        wider_predict_table = wider_predict_table.append(wider_row, ignore_index=True)


def classifier_baseline_nn():
    """
    Creates a fully connected neural network with the input layer having as many nodes as the input, and an
    output layer
    :return:
    """
    # Create model
    model = Sequential()
    model.add(Dense(16, input_dim=12, kernel_initializer='normal', activation="relu"))
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
    model.add(Dense(16, input_dim=12, kernel_initializer='normal', activation="relu"))
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
    model.add(Dense(16, input_dim=12, kernel_initializer='normal', activation="relu"))
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
    model.add(Dense(16, input_dim=12, kernel_initializer='normal', activation="relu"))
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
    gnb.fit(X_train, y_train)
    gnb_predicitions = gnb.predict(X_test)
    print("Gaussian Navie Bayes:")
    print(classification_report(y_test, gnb_predicitions))

    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    logreg_predicitons = logreg.predict(X_test)
    print("Logistic Regression report:")
    print(classification_report(y_test, logreg_predicitons))

    # Support Vector Machine
    svm = SVC()
    svm.fit(X_train, y_train)
    svm_predictions = svm.predict(X_test)
    print("Support Vector Machine report:")
    print(classification_report(y_test, svm_predictions))

    # Decision Tree
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    tree_predictions = tree.predict(X_test)
    print("Decision Tree report:")
    print(classification_report(y_test, tree_predictions))

    # Random Forest
    forest = RandomForestClassifier()
    forest.fit(X_train, y_train)
    forest_predicions = forest.predict(X_test)
    print("Random Forest report:")
    print(classification_report(y_test, forest_predicions))

    # Sklearn NN
    nn = MLPClassifier(hidden_layer_sizes=(16, 16), batch_size=40, random_state=1)
    nn.fit(X_train, y_train)
    nn_predictions = nn.predict(X_test)
    print("Sklearn NN report:")
    print(classification_report(y_test, nn_predictions))



if __name__ == '__main__':
    main_classifier()
