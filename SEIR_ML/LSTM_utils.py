"""
ECE227 Final Project SP21 by Samuel Cowin and Fatemeh Asgarinejad

Data found at https://github.com/govex/COVID-19 by Johns Hopkins University
"""


import pandas as pd
import numpy as np
from numpy import array

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


def vaccination_data():
    """
    Retrieve vaccination data from John Hopkin's Github
    """

    url = "https://raw.githubusercontent.com/govex/COVID-19/master/data_tables/vaccine_data/us_data/time_series/time_series_covid19_vaccine_doses_admin_US.csv"
    vaccination = pd.read_csv(url, error_bad_lines=False)

    for c in list(vaccination.columns)[:10]:
        del vaccination[c]

    vaccination = vaccination.fillna(0)

    return vaccination


def split_sequence(sequence, n_steps):
    """
    Split input sequence into sample steps for LSTM
    """

    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    return array(X), array(y)


def LSTM_full(sequence):
    """
    Prediction with LSTM using entire input sequence
    """

    n_steps = 3
    X, y = split_sequence(sequence, n_steps)  # splitting into samples
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=200, verbose=0)

    return model


def LSTM_after_start(sequence, epochs):
    """
    LSTM prediction after vaccination rate is nonzero
    """

    sequence = list(filter((0.0).__ne__, sequence))
    n_steps = 3
    X, y = split_sequence(sequence, n_steps)  # splitting into samples
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=epochs)

    return model, np.shape(X)
