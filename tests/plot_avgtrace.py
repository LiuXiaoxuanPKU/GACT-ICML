import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

with open('block_avgtrace.txt') as f:
    avgtrace = [float(l) for l in f]

fig, ax = plt.subplots()
ax.plot(avgtrace)
ax.set_yscale('log')

fig.savefig('avgtrace.pdf')