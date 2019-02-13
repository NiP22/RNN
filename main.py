import pandas as pd
from RNN.preprocessing import prep_data
import numpy as np
from RNN.model import lstm
from matplotlib import pyplot as plt

data = pd.read_csv('data.csv', sep='|')
pln = 40000550039
sample = data[data['PLN'] == pln]

sample = prep_data(sample)
pred = lstm(sample)

data = np.array(sample['Revenue'])
plt.plot(np.arange(data.shape[0]), data, color='blue')
plt.plot(np.arange(data.shape[0] - 52, data.shape[0]), pred, color='red')
plt.show()
