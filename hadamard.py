import numpy as np
import matplotlib
import math
import torch
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from image_classification.quantize import config, Hadamard


config.hadamard = True


def calc_range(e):
    e = np.reshape(e, [64, -1])
    e_min = np.min(e, 1, keepdims=True)
    e_max = np.max(e, 1, keepdims=True)
    return e_min, e_max - e_min


def normalize_per_sample(e):
    e = np.reshape(e, [64, -1])
    e_min, e_range = calc_range(e)
    e = (e - e_min) / e_range
    return e.ravel()


errors = np.load('errors.pkl.npz', allow_pickle=True)
n = len(errors)
errors = [torch.tensor(errors['arr_{}'.format(i)][0], dtype=torch.float32).cuda() for i in range(n)]
print('Data loaded')

e0 = errors[-1]
e1 = Hadamard.sample_channel_hadamard(e0, False)
e2 = Hadamard.width_hadamard(e1, False)
e3 = Hadamard.height_hadamard(e2, False)
e4 = Hadamard.height_hadamard(e3, True)
e5 = Hadamard.width_hadamard(e4, True)
e6 = Hadamard.sample_channel_hadamard(e5, True)
print('Error: {}'.format((e0-e6).abs().mean()))
print('Mean: {}'.format(e0.abs().mean()))
es = [e0, e1, e2, e3]

n, c, w, _ = e0.shape
vmin = e1.min()
vmax = e1.max()
fig, ax = plt.subplots(n, 4, figsize=(20, 5 * n))
for i in range(n):
    for a, e in zip(ax[i], es):
        a.imshow(np.reshape(e[i].cpu(), [c, w * w]), vmin=vmin, vmax=vmax, aspect='auto')
        a.set_title(str(i) + ' ' + str(e[i].max() - e[i].min()))

fig.savefig('fmap.pdf')
