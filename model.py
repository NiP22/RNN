import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.metrics import mean_squared_error
plt.rcParams['figure.figsize'] = (17, 5)


def mse(test, pred):
    sum = 0
    for i, test_val in enumerate(test):
        sum += (test_val - pred[i])**2
    return sum/len(test)


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


def lstm(df, plot_without_season=1):
    data = np.array(df[['Revenue', 'Promo', 'Season']])
    n_features = 3
    n_steps = 3
    in_seq1 = np.array(df[['Revenue']])
    in_seq2 = np.array(df[['Promo']])
    in_seq3 = np.array(df[['Season']])
    data = np.hstack((in_seq1, in_seq2, in_seq3, in_seq1))
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
    if plot_without_season:
        n_features = 2
        n_steps = 3
        data = np.hstack((in_seq1, in_seq2, in_seq1))
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
        y_pred_seasonal = model.predict(X_test)
        plt.plot(np.arange(y_tr.shape[0] - 1, y_tr.shape[0] + y_pred.shape[0] - 1), y_pred_seasonal, color='black',
                 label='prediction without season')
        print('mse without seasons')
        print(mse(y_test, y_pred_seasonal))
    plt.plot(np.arange(y_tr.shape[0]), y_tr, color='blue')
    plt.plot(np.arange(y_tr.shape[0], y_tr.shape[0] + y_test.shape[0]), y_test, color='blue', label='real series')
    plt.plot(np.arange(y_tr.shape[0] - 1, y_tr.shape[0] + y_pred.shape[0] - 1), y_pred, color='red', label='prediction')
    y_pred = y_pred.reshape((52, ))
    
    print('mse with seasons')
    print(mse(y_test, y_pred))

    plt.legend()
    plt.savefig(str(df.iloc[0, 0]) + "plot.png")
    plt.close()
    return y_pred

