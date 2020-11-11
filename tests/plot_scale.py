import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

fig, ax = plt.subplots(3, figsize=(15, 15))
for epoch in range(3):
    for i in range(4):
        scale = torch.load('{}_{}.scale'.format(epoch, i*100)).numpy()
        ax[epoch].plot(scale, '.', label=str(i))
        ax[epoch].set_yscale('log')
        ax[epoch].legend()

fig.savefig('scale.pdf')