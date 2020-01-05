import numpy as np
import matplotlib
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


def normalize_per_sample(e):
    e = np.reshape(e, [64, -1])
    e_min = np.min(e, 1, keepdims=True)
    e_max = np.max(e, 1, keepdims=True)
    e = (e - e_min) / (e_max - e_min)
    return e.ravel()


H = hadamard(6)
errors = np.load('errors.pkl.npz', allow_pickle=True)
n = len(errors)
errors = [errors['arr_{}'.format(i)] for i in range(n)]
print('Data loaded')

fig, ax = plt.subplots(n, 2, figsize=(10, 5*n))
for i in range(n):
    print(i)
    e = errors[i][0]
    print('Before: {} {}'.format(e.min(), e.max()))
    ax[i, 0].hist(normalize_per_sample(e), bins=256)

    e = np.reshape(e, [64, -1])
    e = np.dot(H, e)
    print('After: {} {}'.format(e.min(), e.max()))
    ax[i, 1].hist(normalize_per_sample(e), bins=256)

fig.savefig('hadamard.pdf')
