import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

x = []
ma = []
pred = []
for i in range(10, 1000001):
    x.append(i)
    # a = np.random.randn(i)
    # ma.append(a.max())
    coeff = i**2 / 2 / np.pi
    pred.append(np.sqrt(np.log(coeff / np.log(coeff))) * (1 + 0.577 / np.log(i)))

fig, ax = plt.subplots()
# ax.plot(x, ma)
ax.plot(x, pred)
ax.set_xscale('log')
fig.savefig('extreme.pdf')
