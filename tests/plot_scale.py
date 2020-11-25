import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# fig, ax = plt.subplots(11, figsize=(15, 55))
# for epoch in range(11):
#     for i in range(4):
#         scale = torch.load('scales/conv1_{}_{}.scale'.format(epoch, i*100)).numpy()
#         # scale = np.exp(scale) / np.sum(np.exp(scale), 1, keepdims=True)
#         # scale = np.max(scale, 1)
#         ax[epoch].plot(scale, '.', label=str(i))
#         ax[epoch].set_yscale('log')
#         ax[epoch].legend()
#
# fig.savefig('scale.pdf')

fig, ax = plt.subplots(figsize=(15, 5))
old_scale = torch.load('scales/old_scale.pt')[0]
new_scale = torch.load('scales/new_scale.pt')[0]
N = old_scale.shape[0]

ax.plot(old_scale, '.', label='old')
ax.plot(new_scale, 'x', label='new')

for i in range(N):
    c = 'r' if new_scale[i] > old_scale[i] else 'b'
    ax.plot([i, i], [old_scale[i], new_scale[i]], c)

ax.set_yscale('log')
ax.legend()

fig.savefig('scale.pdf')

# fig, ax = plt.subplots(figsize=(15, 5))
# scales = []
# for epoch in range(11):
#     for i in range(4):
#         scale = torch.load('scales/conv1_{}_{}.scale'.format(epoch, i*100)).numpy()
#         scales.append(scale)
#
# scales = np.stack(scales, 1)
# N, T = scales.shape
# color_step = 1.0 / T
#
# for i in range(N):
#     ax.plot([i, i], [1e-5, 0.1], color=[0.8, 0.8, 0.8])
#
# for i in range(T):
#     ax.plot(range(N), scales[:, i], '.', color=(1.0 - color_step * i, 1.0 - color_step * i, 1.0))
#
# ax.set_yscale('log')
# fig.savefig('scale.pdf')
