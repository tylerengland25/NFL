import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


def load_data():
    df = pd.read_csv("backend/data/input_data.csv")
    return df


def nn():
    """
    Initializes and compiles a neural network
    :return:
    """
    model = Sequential()
    model.add(Dense(12, input_dim=28, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def main():
    load_data()


if __name__ == '__main__':
    main()
