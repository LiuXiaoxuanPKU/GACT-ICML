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
n, c, w, _ = e0.shape
print('N = {}, C = {}, W = {}'.format(n, c, w))
e1 = Hadamard.sample_channel_hadamard(e0, False)
e2 = Hadamard.width_hadamard(e1, False)
e3 = Hadamard.height_hadamard(e2, False)
H = Hadamard.get_hadamard(w * w * c, False)
e_star = torch.reshape(e3, [-1, w * w * c]) @ H
e4 = Hadamard.height_hadamard(e3, True)
e5 = Hadamard.width_hadamard(e4, True)
e6 = Hadamard.sample_channel_hadamard(e5, True)
print('Error: {}'.format((e0-e6).abs().mean()))
print('Mean: {}'.format(e0.abs().mean()))
es = [e0, e1, e2, e3, e_star]


vmin = e1.min()
vmax = e1.max()
fig, ax = plt.subplots(n, 5, figsize=(25, 5 * n))
for i in range(n):
    for a, e in zip(ax[i], es):
        a.imshow(np.reshape(e[i].cpu(), [c, w * w]), vmin=vmin, vmax=vmax, aspect='auto')
        a.set_title(str(i) + ' ' + str(e[i].max() - e[i].min()))

    best_e = torch.ones_like(es[3][0]) * torch.norm(es[3][i]) / np.sqrt(c * w * w)
    print(torch.norm(es[3][i]), torch.norm(best_e))
    print(es[0][i].max() - es[0][i].min(), es[3][i].max() - es[3][i].min(), es[4][i].max() - es[4][i].min(), best_e.max())

fig.savefig('fmap.pdf')
