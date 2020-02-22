import numpy as np
import matplotlib
import math
import torch
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from image_classification.quantize import config, Hadamard
from fwht import FWHT


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


def process(e0, file_name):
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
    print('Error: {}'.format((e0 - e6).abs().mean()))
    print('Mean: {}'.format(e0.abs().mean()))
    es = [e0, e1, e2, e3, e_star]

    print('FWHT...')
    e_np = e0.cpu().numpy().ravel()
    e_wh = FWHT(e_np)
    e_wh = np.reshape(e_wh, [n, c, w, w])
    print('FWHT finished')

    # vmin = e1.min()
    # vmax = e1.max()
    # fig, ax = plt.subplots(n, 7, figsize=(35, 5 * n))
    # for i in range(n):
    #     for a, e in zip(ax[i], es):
    #         a.imshow(np.reshape(e[i].cpu(), [c, w * w]), vmin=vmin, vmax=vmax, aspect='auto')
    #         a.set_title(str(i) + ' ' + str(e[i].max() - e[i].min()))
    #
    #     ax[i, -2].hist(es[0][i].cpu().numpy().ravel(), bins=50)
    #     ax[i, -1].hist(es[-1][i].cpu().numpy().ravel(), bins=50)
    #
    #     best_e = torch.ones_like(es[3][0]) * torch.norm(es[3][i]) / np.sqrt(c * w * w)
    #     print(torch.norm(es[3][i]), torch.norm(best_e))
    #     print(es[0][i].max() - es[0][i].min(), es[3][i].max() - es[3][i].min(), es[4][i].max() - es[4][i].min(),
    #           e_wh[i].max() - e_wh[i].min(), best_e.max())
    #
    # fig.savefig(file_name)


# acts = np.load('acts.pkl.npz', allow_pickle=True)
# n = len(acts)
# acts = [torch.tensor(acts['arr_{}'.format(i)], dtype=torch.float32).cuda() for i in range(n)]
# a0 = acts[-3]
# process(a0, 'amap.jpg')


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

mvec = x.abs().max(1)[0]
# mvec = x.max(1)[0] - x.min(1)[0]
rank = (-mvec).argsort()
values = mvec[rank]
x = x[rank]
N = mvec.shape[0]

Qs = [[], [torch.eye(1), 1.0, 1.0]]
for i in range(2, N+1):
    e1 = torch.zeros(i)
    e1[0] = 1
    ones = torch.ones(i) / math.sqrt(i)
    H = householder(e1, ones)
    H1 = 1 / math.sqrt(i)
    Hmax = H.abs().max()
    Qs.append([H, H1, Hmax])

G = [[i] for i in range(N)]


##################################################
# N = 16
# x_part = torch.cat([x[:1], x[113:]], 0)
# lambda_1 = x_part[0].abs().max()
# lambda_2 = x_part[1:].abs().max()
# s = torch.tensor([lambda_1**(-1/3) * N**(1/6), lambda_2**(-1/3) * N**(1/3)])
# s *= (1 / s).norm()
# U = Qs[N][0]
# S = torch.ones(N) * s[1]
# S[0] = s[0]
# S *= (1 / S).norm()
# T = U @ torch.diag(S)


def compute_obj(G):
    weight = values.clone()
    for i in range(N):
        if G[i]:
            sz = len(G[i])
            for cnt, k in enumerate(G[i]):
                if cnt == 0:
                    weight[k] *= Qs[sz][1]
                else:
                    weight[k] *= Qs[sz][2]

    s = torch.pow(weight, -1/3)
    s *= (1 / s).norm()
    return (weight * s).sum(), s


def compute_obj2(G):
    weight = values.clone()
    all_s = torch.zeros_like(weight)
    group_objs = []
    for i in range(N):
        if G[i]:
            sz = len(G[i])
            if sz == 1:
                all_s[i] = 1
                group_objs.append(weight[i])
            else:
                w = torch.tensor([weight[i] / math.sqrt(sz), weight[G[i][1]] * Qs[sz][2]])
                s = torch.tensor([w[0]**(-1/3), (w[1]/(N-1))**(-1/3)])
                s *= (1 / s).norm()
                for k in G[i]:
                    all_s[k] = s[1]
                all_s[i] = s[0]
                group_objs.append((w*s).sum())

    return torch.tensor(group_objs).norm(), all_s



# last_obj, _ = compute_obj2(G)
# for i in range(N-1, 0, -1):
#     print(i)
#     if len(G[i]) > 1:
#         break
#     G[i] = []
#     best_obj = 1e10
#     best_j = -1
#     for j in range(i):
#         G[j].append(i)
#         obj, _ = compute_obj2(G)
#         if obj < best_obj:
#             best_obj = obj
#             best_j = j
#         G[j].pop()
#
#     if best_obj < last_obj:
#         print('Adding {} to {}, new obj = {}'.format(i, best_j, best_obj))
#         last_obj = best_obj
#         G[best_j].append(i)
#         print(G)
#     else:
#         G[i] = [i]
#         break

num_zeros = 0
total_values = values.sum()
while True:
    num_zeros += 1
    total_values -= values[N - num_zeros]
    num = num_zeros * values[N - num_zeros - 1] / total_values
    if num >= 1:
        break

num_nonzeros = N - num_zeros
nums = (num_zeros * values / total_values)[:num_nonzeros]
nums = torch.floor(torch.cumsum(nums, 0) + 1e-7).int()
G = [[] for i in range(N)]
cnt = num_nonzeros
for i in range(num_nonzeros):
    G[i] = [i]
    for j in range(cnt, num_nonzeros + nums[i]):
        G[i].append(j)
    cnt = num_nonzeros + nums[i]

T = torch.zeros(N, N)
_, s = compute_obj2(G)
for g in G:
    if g:
        sz = len(g)
        q = Qs[sz][0]
        for i in range(sz):
            for j in range(sz):
                T[g[i], g[j]] = q[i, j]

T = T @ torch.diag(s)
T_per_std = calc_real_std(T, x, True)


# mvec = x.abs().max(1)[0]
# rank = (-mvec).argsort()
# values = mvec[rank]
# x = x[rank]
# N = mvec.shape[0]
# num_max = 15
# num_zero = N - num_max
# zeros = values[num_max:]
# prop = values[:num_max]
# prop = torch.cumsum(prop, 0)
# prop *= num_zero / prop[-1]
# prop = torch.floor(prop).int()
# T = torch.zeros(N, N)
# zero_start = 0
# for i_max in range(num_max):
#     i_zero = zero_start
#     next_zero_start = prop[i_max]
#     m = values[i_max] / zeros[i_zero]
#     n = next_zero_start - zero_start + 1
#
#     e1 = torch.zeros(n)
#     e1[0] = 1
#     one_vec = torch.ones(n) / math.sqrt(n)
#     u = householder(e1, one_vec)
#     s = torch.ones(n)
#     s[0] = m ** (-1.0 / 3)
#     t = u @ torch.diag(s)
#     indices = [i_max]
#     for i in range(zero_start, next_zero_start):
#         indices.append(num_max + i)
#
#     for r in range(n):
#         for c in range(n):
#             T[indices[r], indices[c]] = t[r, c]
#
#     zero_start = next_zero_start

#
# N = 16
# x = torch.rand(N, 1000) * 2 - 1
# x[0] *= 1000
# mvec = torch.ones(N) * 1
# mvec[0] = 1000
#
# e1 = torch.zeros(N)
# e1[0] = 1
# one_vec = torch.ones(N) / math.sqrt(N)
# U_T3 = householder(e1, one_vec)
#
# S_T3 = torch.pow(mvec, -1.0 / 3)
# T3 = U_T3 @ torch.diag(S_T3)
#
# T3_per_std = calc_real_std(T3, x, True)
#
#
# def mymat(N):
#     T = torch.zeros(N, N)
#     for i in range(N):
#         T[i, 0] = 1.0 / N
#     for i in range(N-1):
#         T[i+1, i+1] = 10
#         T[i, i+1] = -10
#
#     return T

# for i in range(len(acts)):
#     a0 = acts[i]
#     a0 = a0.cpu().numpy()
#     a0 = np.reshape(a0, [128, -1])
#     x = torch.tensor(a0) #[perm]
#     I_per_std = calc_real_std(I, x, True)
#     P_per_std = calc_real_std(P, x, True)
#     O_per_std = calc_real_std(O, x, True)
#     print(i, I_per_std.norm(), P_per_std.norm(), O_per_std.norm())


# def sqrt_mat(m):
#     q, s, _ = torch.svd(m)
#     return q @ torch.diag(torch.sqrt(s)) @ q.transpose(0, 1)
#
# xxt = x @ x.transpose(0, 1)
# q, s, _ = torch.svd(xxt)
# sqrt_xxt = q @ torch.diag(torch.sqrt(s)) @ q.transpose(0, 1)
# utu = q @ torch.diag(1.0 / torch.sqrt(s)) @ q.transpose(0, 1)
#
# u0 = torch.cholesky(utu, upper=True)

# p = (1.0 / torch.diag(torch.sqrt(s))) @ q.transpose(0, 1)
# u0 = torch.cholesky(p, upper=True)

