import numpy as np
import matplotlib
import math
import torch
import time
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from image_classification.quantize import config, Hadamard
from fwht import FWHT
from image_classification.preconditioner import init, get_transform


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

from image_classification.quantize import quantize, calculate_qparams, config


def calc_std(u, x, persample, num_bits=4):
    ux = u @ x
    u_inv_2 = u.inverse() ** 2
    if persample:
        m = ux.max(1, keepdim=True)[0] - ux.min(1, keepdim=True)[0]
        return torch.sqrt(u_inv_2 @ m**2 @ torch.ones(1, x.shape[1])) / (2**num_bits - 1) / 2
    else:
        m = ux.max() - ux.min()
        return m * torch.sqrt(u_inv_2 @ torch.ones(x.shape)) / (2**num_bits - 1) / 2


def calc_real_std(u, x, persample, num_bits=4):
    config.backward_persample = persample
    ux = u @ x
    u_inv = u.inverse()
    qparams = calculate_qparams(ux, num_bits=num_bits, reduce_type='extreme')
    xs = []
    for i in range(100):
        qux = quantize(x, qparams=qparams, num_bits=num_bits, stochastic=True)
        uiqux = u_inv @ qux
        xs.append(uiqux)
    x_std = torch.stack(xs, 0).std(0)
    return x_std


I = torch.eye(128)
P, _ = torch.qr(u)
I_std_0 = calc_std(I, x, False)
I_per_std_0 = calc_std(I, x, True)
U_std_0 = calc_std(u, x, False)
U_per_std_0 = calc_std(u, x, True)
P_std_0 = calc_std(P, x, False)
P_per_std_0 = calc_std(P, x, True)

I_std = calc_real_std(I, x, False)
I_per_std = calc_real_std(I, x, True)
U_std = calc_real_std(u, x, False)
U_per_std = calc_real_std(u, x, True)
P_std = calc_real_std(P, x, False)
P_per_std = calc_real_std(P, x, True)

# x_norm = (x**2).sum(0)
# x_rank = torch.flip(torch.argsort(x_norm), [0])
# x_rank = x_rank[:100]
# partial_x = x[:, x_rank]


def get_loss(u, x, persample, num_bits=4):
    std = calc_std(u, x, persample, num_bits)
    return std.norm()


# def get_loss(u, x, persample):
#     ux = u @ x
#     return ux.abs().max()
#     u_inv = u.inverse()
    # return (u_inv**2).sum() * (ux**2).sum()


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


def persample(x):
    mn = x.min(1, keepdim=True)[0]
    mx = x.max(1, keepdim=True)[0]
    return (x - mn) / (mx - mn)


# u, _ = torch.qr(u)
# u = torch.diag(s) + a @ b
u = U_T @ torch.diag(S_T)
ux = u @ x
fig, ax = plt.subplots()
ax.hist(persample(ux).detach().numpy().ravel(), bins=50)
fig.savefig('hist.png')

O = u
O_per_std_0 = calc_std(O, x, True)
O_per_std = calc_real_std(O, x, True)
Q = torch.diag(1.0 / ux.abs().max(1)[0]) @ O
Q_std_0 = calc_std(Q, x, False)

fig, ax = plt.subplots()
ax.hist(persample(x).detach().numpy().ravel(), bins=50)
fig.savefig('hist_0.png')

Px = P @ x
fig, ax = plt.subplots()
ax.hist(persample(Px).detach().numpy().ravel(), bins=50)
fig.savefig('hist_p.png')


def householder(src, tar):
    N = src.shape[0]
    v = tar - src
    v = v / v.norm()
    return torch.eye(N) - 2 * v.view(N, 1) @ v.view(1, N)


one_vec = torch.ones(128) / math.sqrt(128)
max_values = x.abs().max(1)[0]
sample_rank = max_values.argsort().flip(0)
mvec = torch.zeros(128)
mvec[sample_rank[0]] = 1
e1 = torch.zeros(128)
e1[0] = 1
V = householder(mvec, e1)

s = torch.ones(128) * max_values[sample_rank[0]] / max_values[sample_rank[1]]
s[0] = 1
U = householder(e1, one_vec)
# U = torch.tensor(hadamard(7), dtype=torch.float32)

T = U @ torch.diag(s) @ V
T_per_std = calc_real_std(T, x, True)

U_x, S_x, V_x = torch.svd(x)
V_T = U_x
U_T = torch.tensor(hadamard(7), dtype=torch.float32)
V_x_max = V_x.abs().max(0)[0]     # Max for each singular vector
weight = S_x * V_x_max
S_T = torch.pow(weight, -1.0 / 3)
S_T *= (1.0 / S_T).norm()
T2 = U_T @ torch.diag(S_T) @ V_T.transpose(0, 1)
T2_per_std = calc_real_std(T2, x, True)

U_T3 = torch.tensor(hadamard(7), dtype=torch.float32)
weight = x.abs().max(1)[0]
S_T3 = torch.pow(weight, -1.0 / 3)
S_T3 *= (1.0 / S_T3).norm()
T3 = U_T3 @ torch.diag(S_T3)
T3_per_std = calc_real_std(T3, x, True)

init(128)
t = time.time()
T = get_transform(x).cpu()
print('Get optimal transform in {} seconds'.format(time.time() - t))
T_per_std = calc_real_std(T, x, True)

for i in range(len(acts)):
    x = acts[i].cpu().view(128, -1)

    t = time.time()
    T = get_transform(x).cpu()
    t = time.time() - t

    naive = calc_real_std(torch.eye(128), x, False)
    ps = calc_real_std(torch.eye(128), x, True)
    hh = calc_real_std(T, x, True)
    print('Layer {}, naive={:.6f} diagonal={:.6f} householder={:.6f}, in {:.4f} seconds'.format(i, naive.norm(), ps.norm(), hh.norm(), t))
