import torch
import numpy as np
import actnn.cpp_extension.calc_precision as ext_calc_precision
from sklearn.linear_model import Ridge


# Automatically compute the precision for each tensor
class AutoPrecision:
    """
    Usage diagram:

    In each iteration:
    1. during forward and back propagation, use self.bits to quantize activations
    2. update optimizer parameters
    3. sample the gradient
    4. call iterate(gradient)
    5. call end_epoch
    # TODO make delta a moving average...
    """
    def __init__(self, bits, groups, dims,
                 momentum=0.999, warmup_iters=100, update_interval=10, sample_size=1000,
                 max_bits=8, adaptive=True, reg=1.0, delta_momentum=0.9):
        """
        :param bits: average number of bits (Python int)
        :param groups: group id of each tensor (Python list)
        :param dims: dimensionality of each tensor (torch.long)
        :param warmup_epochs: use adaptive sensitivity after certain epochs
        :param max_bits: maximum number of bits per each dim
        :param adaptive: use adaptive sensitivity or not.
                         If False, no gradient is required in iterate()
        :param reg: weight decay for ridge regression
        """
        self.L = len(groups)
        self.num_groups = np.max(groups) + 1
        self.groups = groups
        self.dims = dims
        # Sensitivity for each tensor, tied within each group
        self.C = torch.ones(self.L)
        self.iter = 0
        self.epoch = 0
        self.adaptive = adaptive

        self.batch_grad_ema = 0
        self.beta1 = 0

        self.abits = bits
        self.bits = torch.ones(self.L, dtype=torch.int32) * bits
        self.total_bits = bits * dims.sum()
        self.max_bits = max_bits
        self.X = []     # The linear system, epoch_size * num_groups matrix
        self.y = []

        self.momentum = momentum
        self.warmup_iters = warmup_iters
        self.update_interval = update_interval
        self.sample_size = sample_size
        self.reg = reg
        self.reward = 0
        self.deltas = torch.zeros(self.L, 8)
        self.delta_beta = torch.zeros(self.L, 8)
        self.delta_normal = torch.ones(self.L, 1) * 1e9
        self.delta_momentum = delta_momentum
        # self.refresh_bits()

    def get_delta(self):
        return self.deltas / (self.delta_beta + 1e-9)

    def refresh_bits(self):
        total_bits = self.total_bits

        self.bits = torch.ones(self.L, dtype=torch.int32) * self.max_bits
        # print((self.deltas / self.delta_normal)[0])
        # print((self.deltas / self.delta_normal)[-1])
        # torch.save([self.deltas, self.delta_normal], 'delta_normal.pkl')
        self.bits = ext_calc_precision.calc_precision_table(self.bits, self.get_delta() / self.delta_normal,
                                                      self.C, self.dims, total_bits)

        # Collect some data
        # prob_dist = torch.tensor([0.1, 0.4, 0.25, 0.15, 0.0, 0.0, 0.0, 0.1])
        # self.bits = torch.multinomial(prob_dist, self.L, replacement=True).to(torch.int32) + 1

        # if self.adaptive:
        #     # Decide an average number of bits
        #     avg_bits = torch.rand([]) * 7
        #     bits = torch.rand([self.L])
        #     bits = bits / bits.mean() * avg_bits     # [0, 7]
        #     self.bits = torch.round(bits).to(torch.int32) + 1

        if self.adaptive:       # TODO control the overall bits
            mask = (torch.rand(self.L) < 0.05).int()
            new_bits = torch.randint_like(self.bits, 2) * 7 + 1
            self.bits = self.bits * (1 - mask) + mask * new_bits

        # torch.save([self.C, self.deltas / self.delta_normal, self.bits], 'used_b.pkl')

    def generate_ls(self, grad):
        X_row = [0 for i in range(self.L)]
        delta = self.get_delta()
        for l in range(self.L):
            X_row[l] += delta[l, self.bits[l] - 1]

        y_row = ((grad - self.batch_grad_ema / self.beta1)**2).sum()
        return X_row, y_row

    def iterate(self, grad, deltas=None, gsizes=None):
        # print(grad)
        """
        Given the sampled gradient vector (gather selected dimensions from the full gradient)
        This procedure will calculate the bits allocation for next iteration, which
        is available in self.bits.

        If grad is not available, simply pass torch.tensor(1.0)
        - Deltas is a vector of quantization error:
          deltas[l] = ||Q(a_l) - a_l||^2, if not available, use 2^(-2b)
        - Gsizes[l] = (gradient**2).mean()
          If not available, use 1
        """
        # Collect deltas
        if deltas is not None:
            for l in range(self.L):
                self.deltas[l, self.bits[l] - 1] = self.deltas[l, self.bits[l] - 1] * self.delta_momentum + \
                                                   (1 - self.delta_momentum) * deltas[l]
                self.delta_beta[l, self.bits[l] - 1] = self.delta_beta[l, self.bits[l] - 1] * self.delta_momentum + \
                                                       (1 - self.delta_momentum)

        delta = self.get_delta()
        self.delta_normal = delta[:, self.abits-1:self.abits] * 2 ** (2 * (self.abits - 1)) + 1e-9
        # self.delta_normal = self.deltas.max(1, keepdims=True)[0] + 1e-9

        if gsizes is not None:
            self.gsizes = gsizes

        self.iter += 1
        grad = grad.detach().cpu()

        # Update the underlying linear system
        if self.iter >= self.warmup_iters:
            X_row, y_row = self.generate_ls(grad)
            self.reward += y_row
            if y_row < 1e6:
                self.X.append(X_row)
                self.y.append(y_row)
            if len(self.X) > self.sample_size:
                self.X.pop(0)
                self.y.pop(0)

        self.refresh_bits()

        # Update batch gradient
        # beta1 will converge to 1
        self.batch_grad_ema = self.momentum * self.batch_grad_ema + (1 - self.momentum) * grad
        self.beta1 = self.momentum * self.beta1 + (1 - self.momentum)

        if self.iter >= 2 * self.warmup_iters and self.iter % self.update_interval == 0:
            self.update_coef()

    def update_coef(self):
        """
        Update the per-tensor sensitivity by solving the linear system
        """
        # Normalize X
        gsizes_normal = self.gsizes.abs().mean()
        X = torch.tensor(self.X) / self.delta_normal.view(1, -1)
        P = torch.zeros(self.L, self.num_groups * 2)
        for l in range(self.L):
            P[l, self.groups[l]] = 1
            P[l, self.groups[l] + self.num_groups] = self.gsizes[l] / gsizes_normal * 10

        # X = np.array(X @ P)
        # y = np.array(self.y)
        # torch.save([self.X, self.y], 'linear_system.pkl')
        # print(X)
        # print(y)
        X = X @ P
        y = torch.tensor(self.y, dtype=torch.float32)

        # Ridge Regression
        N = X.shape[0]
        X = torch.cat([X, torch.ones([N, 1])], 1)
        F = X.shape[1]
        V = torch.eye(F) * self.reg + X.t() @ X
        Xy = (X * y.view(-1, 1)).sum(0)
        coefs = V.inverse() @ Xy
        intercept = coefs[-1]
        coefs = coefs[:-1]

        # print('Intercept: ', intercept)
        # C = P @ coefs
        # C0 = P[:, :self.num_groups] @ coefs[:self.num_groups]
        # for l in range(self.L):
        #     print(C[l], C0[l])

        # torch.save([X, y, self.deltas, self.delta_normal, coefs, intercept, P, self.gsizes], 'linear_system.pkl')

        self.C = P @ coefs
        min_coef = self.C.min()
        # print('Coefficients: ', coefs)
        if min_coef < 0:
            print('ActNN Warning: negative coefficient detected ', min_coef)