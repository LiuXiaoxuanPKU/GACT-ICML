import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

fig, ax = plt.subplots(11, figsize=(15, 55))
for epoch in range(11):
    for i in range(4):
        scale = torch.load('scales/conv1_{}_{}.scale'.format(epoch, i*100)).numpy()
        # scale = np.exp(scale) / np.sum(np.exp(scale), 1, keepdims=True)
        # scale = np.max(scale, 1)
        ax[epoch].plot(scale, '.', label=str(i))
        ax[epoch].set_yscale('log')
        ax[epoch].legend()

fig.savefig('scale.pdf')