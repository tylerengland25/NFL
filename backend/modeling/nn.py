import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_data():
    """
    Loads data and splits into X and Y training and testing sets
    :return:
    """
    df = pd.read_csv("backend/data/inputs.csv")
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


def main():
    # Load datasets
    X_train, X_test, y_train, y_test = load_data()

    # Fit baseline model
    baseline_model = baseline_nn()
    baseline_model.fit(X_train, y_train, epochs=50, batch_size=50)

    # Evaluate model
    loss, metrics = baseline_model.evaluate(X_test, y_test)
    print('Baseline Loss (MSE): %.2f\nBaseline Metrics (MSE): %.2f' % (loss, metrics))

    # Predict on test set, [(Home_score, Away_score), ...]
    baseline_predict_table = pd.DataFrame()
    predictions = baseline_model.predict(X_test)
    for prediction, actual in zip(predictions, y_test.tolist()):
        baseline_row = {'Home_Score_Predict': prediction[0],
                        'Home_score': actual[0],
                        'Away_Score_Predict': prediction[1],
                        'Away_Score': actual[1]}

        baseline_predict_table = baseline_predict_table.append(baseline_row, ignore_index=True)


if __name__ == '__main__':
    main()
