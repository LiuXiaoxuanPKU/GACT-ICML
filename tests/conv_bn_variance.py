import torch
from tqdm import tqdm
from quantize import config, QScheme, QBNScheme, QConv2d, QBatchNorm2d, quantize_mixed_precision, dequantize_mixed_precision
import pytorch_minimax
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

config.activation_compression_bits = 2
config.initial_bits = 2
config.compress_activation = True
num_bins = 3
num_locations = 7 * 7
QScheme.num_samples = 50
QScheme.update_scale = True
QScheme.batch = torch.arange(0, 50)
QBNScheme.num_samples = 50
QBNScheme.batch = torch.arange(0, 50)
QBNScheme.update_scale = True


input, weight, grad_output, grad_weight = torch.load('../resnet/layer_0.pt')
N, C_in, H, W = input.shape
_, C_out, _, _ = grad_output.shape
conv = torch.nn.Conv2d(C_in, C_out, 7, stride=2, padding=3, bias=False).cuda()
with torch.no_grad():
    conv.weight.copy_(weight)

bn_inp, bn_weight, bn_bias, bn_grad_output, bn_grad_input = torch.load('../resnet/bn_layer_0.pt')
bn_weight = bn_weight.view(-1)
bn = QBatchNorm2d(C_out).cuda()
with torch.no_grad():
    bn.weight.copy_(bn_weight)
    bn.bias.copy_(bn_bias)

config.compress_activation = False
conv_output = conv(input)
output = bn(conv_output)
grad = torch.autograd.grad(output, conv.weight, bn_grad_output)[0]
exact_grad = grad.detach().clone()
print(exact_grad.norm(), (exact_grad - grad_weight).norm())

QScheme.update_scale = False
QBNScheme.update_scale = False


config.compress_activation = True
grads = []
batch_mean = bn_inp.mean((0, 2, 3), keepdim=True)
batch_std = torch.sqrt(bn_inp.var((0, 2, 3), keepdim=True) + bn.eps)
normalized = (bn_inp - batch_mean) / batch_std
q_bits, q_min, mx = bn.scheme.compute_quantization_bits(normalized)
print(q_bits)
normalized_0 = normalized.clone()

fig, ax = plt.subplots(2, figsize=(5, 10))
ax[0].hist(normalized_0.cpu().detach().numpy().ravel(), bins=50)
ax[0].set_yscale('log')
ax[1].hist(grad_output.cpu().detach().numpy().ravel(), bins=50)
ax[1].set_yscale('log')
fig.savefig('bn.pdf')

for iter in range(10):
    # compute grad_bn_inp
    q_input, q_input_shape, q_bits, q_scale = \
        quantize_mixed_precision(normalized_0, q_bits, q_min, mx, True)
    normalized = dequantize_mixed_precision(q_input, q_input_shape, q_bits, q_scale, q_min)
    # normalized = normalized_0
    grad_normalized = bn_grad_output * bn_weight.view(1, -1, 1, 1)

    # print(normalized_0.norm(), (normalized - normalized_0).norm())

    mean_grad_normalized = grad_normalized.mean((0, 2, 3), keepdim=True)
    mean_grad = (normalized * grad_normalized).mean((0, 2, 3), keepdim=True)
    # print(mean_grad.view(-1))
    grad_input = grad_normalized - mean_grad_normalized - normalized * mean_grad
    # grad_input = grad_normalized - mean_grad_normalized
    # grad_input = - normalized * mean_grad
    grad_input = grad_input / batch_std
    # print(grad_input.norm(), (grad_input - bn_grad_input).norm())

    output = conv(input)
    grad = torch.autograd.grad(output, conv.weight, grad_input)[0]
    grads.append(grad.detach().clone())

grads = torch.stack(grads, 0)
grad_std = grads.std(0)
grad_var = (grad_std ** 2).sum()
print((exact_grad**2).sum())
print(grad_var)
grad_mean = grads.mean(0)
print(grad_mean.norm(), (grad_mean - exact_grad).norm())


#
# grad_sum = (grad_output.view(N, -1) ** 2).sum(1)
# input_flatten = input.view(N, -1)
# mn = pytorch_minimax.min(input_flatten)
# mx = pytorch_minimax.max(input_flatten)
# Range = mx - mn
# var = num_locations * C_in / 4 * (Range ** 2 * grad_sum).sum() / num_bins**2
# # var = (C / 9).sum()
# print(var)
#
