import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

prefix = sys.argv[1]
data = []

for epoch in range(1, 201):
    with open('{}/{}.log'.format(prefix, epoch)) as f:
        for line in f:
            if line.find('conv_1_1_1, batch grad mean') != -1:
                line = line.replace(',', '=').split('=')
                data.append([float(line[2]), float(line[4]), float(line[6]), float(line[8])])

data = np.array(data)
quant_std_ma = []
for i in range(200):
    quant_std_ma.append(data[max(i-3+1,0):i+1, 2].mean())

fig, ax = plt.subplots()
ax.plot(data[:, 0], label='batch mean')
# ax.plot(data[:, 1], label='quant bias')
ax.plot(quant_std_ma, label='quant std')
ax.plot(data[:, 3], label='sample std')
ax.legend()
ax.set_yscale('log')
fig.savefig('std.pdf')
