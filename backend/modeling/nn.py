import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler


def load_data_classifier():
    """
    Loads data and splits into X and Y training and testing sets
    :return:
    """
    df = pd.read_csv("backend/data/inputs.csv")
    df["Win_Lose"] = (df["H_Score"] - df["A_Score"]) > 0
    df["Win_Lose"] = df["Win_Lose"].astype(int)
    y_cols = {"Win_Loss"}
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


def main():
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


if __name__ == '__main__':
    main()
