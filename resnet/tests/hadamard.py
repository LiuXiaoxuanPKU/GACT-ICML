import numpy as np
import matplotlib
import math
import torch
import time
matplotlib.use('Agg')
from quantize.quantize import config
from quantize.preconditioner import init, BlockwiseHouseholderPreconditioner, ScalarPreconditioner, DiagonalPreconditioner

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
x = torch.tensor(a0)

U_T = torch.tensor(hadamard(7), dtype=torch.float32)
weight = x.max(1)[0] - x.min(1)[0]
S_T = torch.pow(weight, -1.0 / 3)
S_T = torch.nn.Parameter(S_T)
u = torch.nn.Parameter(torch.qr(torch.randn(128, 128))[0])

from quantize.quantize import quantize


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
for step in range(0):
    opt.zero_grad()
    u = U_T @ torch.diag(S_T)
    loss = get_loss(u, x, False)
    loss.backward()
    opt.step()
    print(step, loss)


init(128)


x = acts[10].cpu().view(128, -1)
mvec = x.abs().max(1)[0]
T1 = torch.diag(1.0 / mvec)
print((T1@x).abs().max())
print(T1.inverse().norm()**2)
N = 128

s = math.sqrt(N) * mvec**(-1./3) / (2 * (mvec**(2./3)).sum())
H = torch.tensor(hadamard(7), dtype=torch.float32)
T = H @ torch.diag(s)
print((T@x).abs().max(), (s*mvec).sum()*2/math.sqrt(N))
print(T.inverse().norm()**2, (s**-2).sum(), 4/N*(mvec**(2/3)).sum()**3)

exit(0)

for i in range(len(acts)):
    x = acts[i].cpu().view(128, -1)

    naive = calc_real_std(lambda x: ScalarPreconditioner(x, 4), x)

    t = time.time()
    ps = calc_real_std(lambda x: DiagonalPreconditioner(x.cuda(), 4), x)
    t0 = time.time() - t

    t = time.time()
    hh = calc_real_std(lambda x: BlockwiseHouseholderPreconditioner(x.cuda(), 4), x)
    t = time.time() - t

    print('Layer {}, naive={:.6f} diagonal={:.6f} householder={:.6f}, in {:.4f} seconds (baseline {:.4f} seconds)'
          .format(i, naive.norm(), ps.norm(), hh.norm(), t, t0))
