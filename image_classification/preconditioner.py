import torch
import math
import time
import pytorch_minimax


def householder(src, tar):
    N = src.shape[0]
    v = tar - src
    v = v / v.norm()
    return torch.eye(N) - 2 * v.view(N, 1) @ v.view(1, N)


Qs = [[], [torch.ones(1), 1.0]]


def init(max_bs):
    for i in range(2, max_bs+1):
        e1 = torch.zeros(i)
        e1[0] = 1
        ones = torch.ones(i) / math.sqrt(i)
        H = householder(e1, ones)
        Hmax = H.abs().max()
        Qs.append([H, Hmax])


# TODO make this CUDA
def get_transform(x):
    N = x.shape[0]
    x = x.view(N, -1)

    # mvec = x.abs().max(1)[0]
    mvec = x.max(1)[0] - x.min(1)[0]
    rank = (-mvec).argsort()
    values = mvec[rank]

    # Get block configurations
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
    nums = torch.round(torch.cumsum(nums, 0)).int()

    # Construct the matrix
    T = torch.zeros(N, N)
    all_s = torch.zeros(N)

    cnt = num_nonzeros
    indices = []
    index_cnt = 0

    for i in range(num_nonzeros):
        # [i] + [cnt ~ num_nonzeros + nums[i]]
        indices.append(i)
        lambda_1 = values[i]
        lambda_2 = values[cnt]
        sz = num_nonzeros + nums[i] - cnt + 1
        Q, Qmax = Qs[sz]
        w = torch.tensor([lambda_1 / math.sqrt(sz), lambda_2 * Qmax])
        s = torch.tensor([w[0] ** (-1 / 3), (w[1] / (sz - 1)) ** (-1 / 3)])
        s *= (1 / s).norm()
        all_s[index_cnt] = s[0]
        all_s[index_cnt+1 : index_cnt+sz] = s[1]
        T[index_cnt:index_cnt+sz, index_cnt:index_cnt+sz] = Q
        index_cnt += sz
        for j in range(cnt, num_nonzeros + nums[i]):
            indices.append(j)
        cnt = num_nonzeros + nums[i]

    # print(nums)
    # print(indices)
    # print(num_nonzeros)
    assert len(indices) == N

    T = T @ torch.diag(all_s)
    indices = rank[indices]
    inv_indices = torch.zeros(N, dtype=torch.int64)
    inv_indices[indices] = torch.arange(N)

    T = T[inv_indices]
    T = T[:, inv_indices]
    return T.cuda()


class Preconditioner:
    def __init__(self, x, num_bits, left=True):
        self.left = left
        self.x_shape = x.shape
        self.num_bins = 2 ** num_bits - 1

        self.x = self.flatten(x)
        self.Tx = self.transform(self.x)

    def flatten(self, x):
        if left:
            self.x_shape2 = x_shape
            return x.view(x.shape[0], -1)
        else:
            # NCHW -> CNHW
            x = x.transpose(0, 1)
            self.x_shape2 = x.shape
            return x.reshape(x.shape[0], -1)

    def deflatten(self, Tx):
        x = x.view(**self.x_shape2)
        if left:
            return x
        else:
            return x.transpose(0, 1)

    def forward(self):
        return self.Tx

    def inverse(self, Tx):
        x = self.inverse_transform(Tx)
        return self.deflatten(x)


class ScalarPreconditioner(Preconditioner):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, x, num_bits, left=True):
        super(ScalarPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x):
        mn = x.min() - 1e-8
        mx = x.max() + 1e-8

        self.zero_point = mn
        self.scale = self.num_bins * (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse(self, x):
        return x / self.scale + self.zero_point


class ForwardPreconditioner(Preconditioner):
    # Y = D (Y - z 1^\top)
    # X = D^-1 Y + z 1^\top
    def __init__(self, x, left=True):
        super(ForwardPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x):
        mn = pytorch_minimax.min(x).mean() - 1e-8
        mx = pytorch_minimax.max(x).mean() + 1e-8

        self.zero_point = mn
        self.scale = self.num_bins * (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse(self, x):
        return x / self.scale + self.zero_point


class DiagonalPreconditioner(Preconditioner):
    # Y = D (Y - z 1^\top)
    # X = D^-1 Y + z 1^\top
    def __init__(self, x, left=True):
        super(DiagonalPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x):
        mn = pytorch_minimax.min(x).unsqueeze(1) - 1e-8
        mx = pytorch_minimax.max(x).unsqueeze(1) + 1e-8

        self.zero_point = mn
        self.scale = self.num_bins * (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse(self, x):
        return x / self.scale + self.zero_point

