import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop


def split_sequence(sequence, n_steps):
    X, y = list(), list()

    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x = sequence[i:end_ix, : -1]
        seq_y = sequence[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def lstm(df):
    data = np.array(df[['Revenue', 'Promo']])
    n_features = 2
    n_steps = 3
    in_seq11 = np.array(df[['Revenue']])
    in_seq12 = np.array(df[['Promo']])
    data = np.hstack((in_seq11, in_seq12, in_seq11))
    tr_data = data[:data.shape[0] - 52 - n_steps]
    test_data = data[data.shape[0] - 52 - n_steps:]
    X_tr, y_tr = split_sequence(tr_data, n_steps)
    X_test, y_test = split_sequence(test_data, n_steps)
    model = Sequential()
    model.add(layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    X_tr = X_tr.reshape((X_tr.shape[0], X_tr.shape[1], n_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    model.fit(X_tr, y_tr, epochs=200, verbose=2)
    y_pred = model.predict(X_test)
    plt.plot(np.arange(y_tr.shape[0]), y_tr, color='blue')
    plt.plot(np.arange(y_tr.shape[0], y_tr.shape[0] + y_test.shape[0]), y_test, color='blue')
    plt.plot(np.arange(y_tr.shape[0], y_tr.shape[0] + y_pred.shape[0]), y_pred, color='red')
    plt.savefig(str(df.iloc[0, 2]) + "plot.png")
    plt.close()
    return y_pred

