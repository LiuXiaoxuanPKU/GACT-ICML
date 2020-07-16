import torch
import numpy as np
from quantize.quantize import config, QF, calc_precision_dp
from tqdm import tqdm

data = np.load('55.pkl.npz')
input = torch.tensor(data['input']).cuda()
weight = torch.tensor(data['weight']).cuda()
grad_H = torch.tensor(np.load('55_grad.npy')).cuda()
weight.requires_grad_()
N = input.shape[0]


def get_gradient(grad_H, input):
    N, D_in, R, C = input.shape
    D_out = grad_H.shape[1]
    print(N, D_in, D_out, R, C)
    grad_W = torch.zeros([D_out, D_in, 3, 3]).cuda()
    for kr in range(3):
        for kc in range(3):
            for r in range(R):
                for c in range(C):
                    dr = kr - 1
                    dc = kc - 1
                    if 0 <= r + dr < R and 0 <= c + dc < C:
                        for n in range(N):
                            grad_W[:, :, kr, kc] += \
                                grad_H[n, :, r, c].view(D_out, 1) * input[n, :, r+dr, c+dc].view(1, D_in)
    return grad_W


# def grad_var_line_3(grad_H, input):
#     \sum_{n, i1, i2} Cov[grad_H[n,i1]X[n,Di1]mn, grad_H[n,i2]X[n,Di2]mn]

grad_sum = (grad_H.view(N, -1)**2).sum(1)
Range = (input.view(N, -1)).max(1)[0] - (input.view(N, -1)).min(1)[0]
input_sum = (input.view(N, -1)**2).sum(1)
D_in = input.shape[1]
D_out = weight.shape[0]
# input_mat = input.view(N, D_in, -1)        # [N, D, L]
# ||input_mat^\top||_infty = maximum L1 norm of a column
# Linf_norm = input_mat.abs().sum(1).max(1)[0]
# L1_norm = input_mat.abs().sum(2).max(1)[0]
# input_sum = Linf_norm * L1_norm

C = (9 * D_in * grad_sum * Range**2 / 4).cpu()
A = (9 * grad_sum * input_sum).cpu()
# print(grad_sum)
# print(A, C)

config.compress_activation = True
config.activation_compression_bits = 32

# Initialize QF
QF.init(128)
QF.set_current_batch(list(range(128)))
QF.update_scale = True

output = QF.conv2d(input, weight, None, (1, 1), (1, 1), (1, 1), 1, 'conv0')
loss = (output * grad_H).sum()
loss.backward()
exact_grad = weight.grad.detach().clone()
mygrad = get_gradient(grad_H, input)
print('Gradient checking...', mygrad.norm(), (exact_grad - mygrad).norm())

config.activation_compression_bits = 2
QF.update_scale = False
b, keep_prob = calc_precision_dp(A, C, 8, config.activation_compression_bits, 4)
print(b)
print(keep_prob)
# b = torch.ones(128) * 8
# keep_prob = torch.ones(128)
# keep_prob[0] = 0.5

B = 2**b - 1
var_bound = (C / (keep_prob * B * B) + (1 - keep_prob) / keep_prob * A).sum()
print('Variance bound ', var_bound)

grads = []
for iter in tqdm(range(10)):
    weight.grad = None
    output = QF.conv2d(input, weight, None, (1, 1), (1, 1), (1, 1), 1, 'conv0')
    loss = (output * grad_H).sum()
    loss.backward()
    grads.append(weight.grad.detach().clone())

grads = torch.stack(grads, 0)
mean_grad = grads.mean(0)
var_grad = grads.var(0)
for i in range(10):
    print((grads[i]**2).sum())
print('Bias = {}, Var = {}'.format(((exact_grad - mean_grad)**2).sum(), var_grad.sum()))

# Y_mat = grad_H[0].view(D_out, -1)       # D_out, loc
# X_mat = input[0].view(D_in, -1)         # D_in, loc
# myvar = ((Y_mat @ X_mat.transpose(0, 1)) ** 2).sum()
# print((Y_mat**2).sum() * (X_mat**2).sum())
#
# Linf_norm = X_mat.abs().sum(0).max()
# L1_norm = X_mat.abs().sum(1).max()
# # input_sum = Linf_norm * L1_norm
# input_sum = torch.svd(X_mat)[1][0] ** 2
# print((Y_mat**2).sum() * input_sum)

print(var_grad.sum([0, 1]))
# print(myvar)
