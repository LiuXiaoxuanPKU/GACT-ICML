import torch
from quantize.quantize import quantize, config, QParams


def calc_real_std(x, w, qparams, num_bits=4):
    config.quantize_gradient = True
    config.backward_persample = False
    xs = []
    for i in range(100):
        qx = quantize(x, qparams=qparams, num_bits=num_bits, stochastic=True)
        xs.append(qx @ w)
    x_std = torch.stack(xs, 0).std(0)
    return x_std


torch.manual_seed(0)

# Standard deviation of Y = XW
N = 128
D = 500
W = torch.randn(D, D)
# bin: -7, -6, -5, 0, ..., +7
# num: -6.5, ..., +6.5
X = torch.randint(-7, 7, size=(N, D)) + 0.5
qparams = QParams(range=15.0, zero_point=-7.0, num_bits=4)
bin_size = 1.0

VarF_X = (bin_size**2 / 4) * N * D
Std_X = calc_real_std(X, torch.eye(D), qparams)
VarF_X_0 = Std_X.norm() ** 2

VarF_XW = (bin_size**2 / 4) * N * W.norm()**2
Std_XW = calc_real_std(X, W, qparams)
VarF_XW_0 = Std_XW.norm() ** 2

