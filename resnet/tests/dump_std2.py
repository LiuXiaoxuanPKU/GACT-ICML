import sys
import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# prefix = sys.argv[1]


def read_data(file_name):
    titles = []
    data = []
    with open(file_name) as f:
        for line in f:
            if line.find('batch grad mean') != -1:
                line = line.replace(',', '=').split('=')
                data.append([float(line[2]), float(line[4]), float(line[6]), float(line[8]), float(line[10])])

    return titles, np.array(data)


_, d_8b = read_data('b8_pFalse_biasFalse.log')
_, d_6b = read_data('b6_pFalse_biasFalse.log')
_, d_4b = read_data('b4_pFalse_biasFalse.log')
_, d_ps_4b = read_data('b4_pTrue_biasFalse.log')
_, d_ps_b_4b = read_data('b4_pTrue_biasTrue.log')

fig, ax = plt.subplots(2, figsize=(5, 10))
ax[0].plot(d_8b[:, 0], 'k:', label='batch mean')
ax[0].plot(d_8b[:, 1], 'k',  label='sample std')
ax[0].plot(d_8b[:, 3], 'b',  label='8bit std')
ax[0].plot(d_6b[:, 3], 'g',  label='6bit std')
ax[0].plot(d_4b[:, 3], 'r',  label='4bit std')
ax[0].plot(d_ps_4b[:, 3], 'c',  label='4bit persample std')
ax[0].plot(d_ps_b_4b[:, 3], 'm',  label='4bit persample biased std')
ax[0].legend()
ax[0].set_yscale('log')

ax[1].plot(d_8b[:, 0], 'k:', label='batch mean')
ax[1].plot(d_8b[:, 2], 'k',  label='sample bias')
ax[1].plot(d_8b[:, 4], 'b',  label='8bit bias')
ax[1].plot(d_6b[:, 4], 'g',  label='6bit bias')
ax[1].plot(d_4b[:, 4], 'r',  label='4bit bias')
ax[1].plot(d_ps_4b[:, 4], 'c',  label='4bit persample bias')
ax[1].plot(d_ps_b_4b[:, 4], 'm',  label='4bit persample biased bias')
ax[1].legend()
ax[1].set_yscale('log')

fig.savefig('std.pdf')
