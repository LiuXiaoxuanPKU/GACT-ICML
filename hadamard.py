import numpy as np
import matplotlib
import math
import torch
import time
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from image_classification.quantize import config
from fwht import FWHT
from image_classification.preconditioner import init, BlockwiseHouseholderPreconditioner, ScalarPreconditioner, DiagonalPreconditioner


config.hadamard = True


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


acts = np.load('errors.pkl.npz', allow_pickle=True)
n = len(acts)
acts = [torch.tensor(acts['arr_{}'.format(i)][0], dtype=torch.float32).cuda() for i in range(n)]
a0 = acts[10]


# Regular
a0 = a0.cpu().numpy()
a0 = np.reshape(a0, [128, -1])
# a0 = np.random.randn(128, 128).astype(np.float32)
x = torch.tensor(a0)

# Center it
# x_max = x.max(1, keepdim=True)[0]
# x_min = x.min(1, keepdim=True)[0]
# x -= (x_max + x_min) / 2

# x = torch.randn_like(x) * 1e-8
# x[0, 0] = 1

U_T = torch.tensor(hadamard(7), dtype=torch.float32)
weight = x.max(1)[0] - x.min(1)[0]
S_T = torch.pow(weight, -1.0 / 3)
S_T = torch.nn.Parameter(S_T)


# u = torch.nn.Parameter(torch.tensor(np.diag(1.0 / a0.max(1))) + 1.0 * torch.randn(128, 128))
u = torch.nn.Parameter(torch.qr(torch.randn(128, 128))[0])

# s = torch.nn.Parameter(torch.ones(128))
# a = torch.nn.Parameter(torch.randn(128, 2) / 10)
# b = torch.nn.Parameter(torch.randn(2, 128) / 10)

from image_classification.quantize import quantize, config


def calc_std(u, x, persample, num_bits=4):
    ux = u @ x
    u_inv_2 = u.inverse() ** 2
    if persample:
        m = ux.max(1, keepdim=True)[0] - ux.min(1, keepdim=True)[0]
        return torch.sqrt(u_inv_2 @ m**2 @ torch.ones(1, x.shape[1])) / (2**num_bits - 1) / 2
    else:
        m = ux.max() - ux.min()
        return m * torch.sqrt(u_inv_2 @ torch.ones(x.shape)) / (2**num_bits - 1) / 2


def calc_real_std(Preconditioner, x):
    xs = []
    for i in range(100):
        qx = quantize(x, Preconditioner, stochastic=True)
        xs.append(qx)
    x_std = torch.stack(xs, 0).std(0)
    return x_std


I = torch.eye(128)
P, _ = torch.qr(u)


def get_loss(u, x, persample, num_bits=4):
    std = calc_std(u, x, persample, num_bits)
    return std.norm()


loss = get_loss(u, x, True)
print(loss)

opt = torch.optim.Adam([S_T], lr=1e-2)
# opt = torch.optim.Adam([s, a, b], lr=1e-1)
for step in range(0):
    opt.zero_grad()
    # q, _ = torch.qr(u)
    # loss = get_loss(q, x, True)
    # loss = get_loss(u, x, False)
    u = U_T @ torch.diag(S_T)
    loss = get_loss(u, x, False)
    # loss = get_loss(u, torch.eye(128), True)
    # loss = get_loss(torch.diag(s) + a @ b, x, True)
    loss.backward()
    opt.step()
    # with torch.no_grad():
    #     u *= u.inverse().norm()
    print(step, loss)


init(128)
for i in range(len(acts)):
    x = acts[i].cpu().view(128, -1)

    naive = calc_real_std(lambda x: ScalarPreconditioner(x, 4), x)
    ps = calc_real_std(lambda x: DiagonalPreconditioner(x.cuda(), 4), x)

    t = time.time()
    hh = calc_real_std(lambda x: BlockwiseHouseholderPreconditioner(x.cuda(), 4), x)
    t = time.time() - t

    print('Layer {}, naive={:.6f} diagonal={:.6f} householder={:.6f}, in {:.4f} seconds'.format(i, naive.norm(), ps.norm(), hh.norm(), t))
