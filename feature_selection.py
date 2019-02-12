import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

df = pd.read_csv('prep_sample.csv', sep='|')
data = np.array(df[['Revenue', 'Promo', 'Year', 'Date']])


def prep_data(dataset, lookback=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - lookback - 1):
        a = dataset[i:(i+lookback), 0]
        dataX.append(a)
        dataY.append(dataset[i + lookback, 0])
    return np.array(dataX), np.array(dataY)



train = data[:222]
test = data[222:274]

trainX, trainY = prep_data(train, 5)
print(trainX)
print(trainX.shape)
print(trainY)
print(trainY.shape)

testX, testY = prep_data(test, 5)


trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(layers.LSTM(16, input_shape=(1, 5)))
model.add(layers.Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=1000, batch_size=1)




X = np.zeros((data.shape[0], data.shape[1] - 1))
Y = np.zeros((data.shape[0], ))

for i, line in enumerate(data):
    X[i, :] = line[1:]
    Y[i] = line[0]
Y_tr = Y[:222]
Y_val = Y[222:274]
X_tr = X[:222, :]
X_val = X[222:274, :]

'''
X_tr = np.reshape(X_tr, (X_tr.shape[0], X_tr.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

model = Sequential()
model.add(layers.LSTM(32, input_shape=(3, 1), dropout=0.4))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mape')
history = model.fit(X_tr, Y_tr, epochs=600, batch_size=5, validation_data=(X_val, Y_val), shuffle=False)

loss =history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('losses')
plt.legend()

plt.show()
'''
Y_test = model.predict(testX)
print(Y_test.shape)
print(Y_test)

plt.plot(np.arange(Y_tr.shape[0]), Y_tr)
plt.plot(np.arange(Y_tr.shape[0]), Y_tr)
plt.plot(np.arange(Y_tr.shape[0], Y_tr.shape[0] + Y_val.shape[0]), Y_val)
plt.plot(np.arange(Y_tr.shape[0], Y_tr.shape[0] + Y_test.shape[0]), Y_test)
plt.show()
