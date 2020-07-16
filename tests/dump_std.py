import sys
import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# prefix = sys.argv[1]


def read_data(prefix):
    data = []
    for epoch in range(1, 201):
        file_name = prefix + '{}.log'.format(epoch)
        if not os.path.exists(file_name):
            continue

        with open(file_name) as f:
            for line in f:
                if line.find('conv_1_1_1, batch grad mean') != -1:
                    line = line.replace(',', '=').split('=')
                    data.append([epoch, float(line[2]), float(line[4]), float(line[6])])

    return np.array(data)

# data = np.array(data)
# quant_std_ma = []
# for i in range(200):
#     quant_std_ma.append(data[max(i-3+1,0):i+1, 2].mean())

data_6 = read_data('6bit/')
data_6_persample = read_data('6bit_persample/')
data_5 = read_data('5bit/')
data_5_persample = read_data('5bit_persample/')

fig, ax = plt.subplots()
ax.plot(data_5[:, 0], data_5[:, 1], label='batch mean')
ax.plot(data_5[:, 0], data_5[:, 2], label='sample std')
ax.plot(data_5[:, 0], data_5[:, 3], label='5 bit std')
ax.plot(data_6[:, 0], data_6[:, 3], label='6 bit std')
ax.plot(data_5_persample[:, 0], data_5_persample[:, 3], label='5 bit std persample')
ax.plot(data_6_persample[:, 0], data_6_persample[:, 3], label='6 bit std persample')
ax.legend()
ax.set_yscale('log')
fig.savefig('std.pdf')
