import numpy as np
import matplotlib
import math
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def hadamard(order):
    if order == 0:
        return np.array([1.0])

    result = np.zeros([2**order, 2**order])
    n = 2**(order - 1)
    sub_mat = hadamard(order - 1)
    result[:n, :n] = sub_mat
    result[:n, n:] = sub_mat
    result[n:, :n] = sub_mat
    result[n:, n:] = -sub_mat
    result /= np.sqrt(2.0)
    return result


# a = hadamard(3)
# print(a)
# print(np.dot(a, a))
#
# v = np.reshape(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), [8, 1])
#
# print(np.dot(a, v))


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


H = hadamard(6)
errors = np.load('errors.pkl.npz', allow_pickle=True)
n = len(errors)
errors = [errors['arr_{}'.format(i)] for i in range(n)]
print('Data loaded')

e0 = errors[-1][0]
e = e0
print('Before: {} {}'.format(e.min(), e.max()))
_, stds = calc_range(e)
print(stds.transpose())
print(stds.mean())

e = np.reshape(e, [64, -1])
e = np.dot(H, e)
print('After: {} {}'.format(e.min(), e.max()))
_, stds = calc_range(e)
print(stds.transpose())
print(stds.mean())

if len(e0.shape) == 4:
    n, c, h, w = e0.shape
    e = np.transpose(e0, (0, 2, 3, 1))      # NHWC
    print(e.shape)
    e = np.reshape(e, [n*h*w, c])
    H = hadamard(math.floor(math.log2(c)))
    e = np.dot(e, H)
    e = np.reshape(e, [n, h, w, c])
    e = np.transpose(e, (0, 3, 1, 2))       # NCHW
    print(e.shape)
print('After channel hadamard: {} {}'.format(e.min(), e.max()))
_, stds = calc_range(e)
print(stds.transpose())
print(stds.mean())

dat = e0[17]
dat1 = e[17]
print(dat.max((1, 2))-dat.min((1, 2)))
print(dat1.max((1, 2))-dat1.min((1, 2)))

# if len(e0.shape) == 4:
#     n, c, h, w = e0.shape
#     e = np.reshape(e0, [n*c, h*w])
#     H = hadamard(math.floor(math.log2(h*w)))
#     e = np.dot(e, [H])
#     e = np.reshape(e, [n, c, h, w])
# print('After all hadamard: {} {}'.format(e.min(), e.max()))
# _, stds = calc_range(e)
# print(stds.transpose())
# print(stds.mean())

# vmin = e0.min()
# vmax = e0.max()
# bs = e0.shape[0]
# fig, ax = plt.subplots(bs, 2, figsize=(10, 5*bs))
# for i in range(bs):
#     ax[i, 0].imshow(np.reshape(e0[i], [16, 1024]), vmin=vmin, vmax=vmax, aspect='auto')
#     ax[i, 0].set_title(str(i) + ' ' + str(e0[i].max() - e0[i].min()))
#     ax[i, 1].imshow(np.reshape(e[i], [16, 1024]), vmin=vmin, vmax=vmax, aspect='auto')
#     ax[i, 1].set_title(str(e[i].max() - e[i].min()))
# fig.savefig('fmap.pdf')


# fig, ax = plt.subplots(n, 3, figsize=(15, 5*n))
# for i in range(n):
#     print(i, errors[i][0].shape)
#     e0 = errors[i][0]
#     e = e0
#     print('Before: {} {}'.format(e.min(), e.max()))
#     ax[i, 0].hist(normalize_per_sample(e), bins=256)
#
#     e = np.reshape(e, [64, -1])
#     e = np.dot(H, e)
#     print('After: {} {}'.format(e.min(), e.max()))
#     ax[i, 1].hist(normalize_per_sample(e), bins=256)
#
#     if len(e0.shape) == 4:
#         e = np.transpose(e0, (0, 2, 3, 1))      # NHWC
#         h = hadamard(math.floor(math.log2(e.shape[3])))
#         e = np.dot(e, h)
#         e = np.transpose(e, (0, 3, 1, 2))       # NCHW
#     print('After channel hadamard: {} {}'.format(e.min(), e.max()))
#     ax[i, 2].hist(normalize_per_sample(e), bins=256)
#
# fig.savefig('hadamard.pdf')

# f = np.abs(errors[1][0])
# vmin = f.min()
# vmax = f.max()
# bs = f.shape[0]
# fig, ax = plt.subplots(bs, figsize=(5, 5*bs))
# for i in range(bs):
#     ax[i].imshow(np.reshape(f[i], [64, 64]), vmin=vmin, vmax=vmax)
# fig.savefig('fmap.pdf')
