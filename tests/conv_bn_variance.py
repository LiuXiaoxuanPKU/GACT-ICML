import torch
from tqdm import tqdm
from quantize import config, QScheme, QBNScheme, QConv2d, QBatchNorm2d, quantize_mixed_precision, dequantize_mixed_precision
import pytorch_minimax
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

config.activation_compression_bits = 2
config.initial_bits = 8
config.compress_activation = True
num_bins = 3
num_locations = 7 * 7
QScheme.num_samples = 50
QScheme.update_scale = True
QScheme.batch = torch.arange(0, 50)
QBNScheme.num_samples = 50
QBNScheme.batch = torch.arange(0, 50)
QBNScheme.update_scale = True


input, weight, grad_output, grad_weight = torch.load('pts/layer_0.pt')
N, C_in, H, W = input.shape
_, C_out, _, _ = grad_output.shape
conv = torch.nn.Conv2d(C_in, C_out, 7, stride=2, padding=3, bias=False).cuda()
with torch.no_grad():
    conv.weight.copy_(weight)

bn_inp, bn_weight, bn_bias, bn_grad_output, bn_grad_input = torch.load('pts/bn_layer_0.pt')
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
bn.scheme.conv_input_norm = (input**2).sum((1,2,3)) * 7 * 7


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

zmeans = []
for iter in range(10):
    # compute grad_bn_inp
    q_input, q_input_shape, q_bits, q_scale = \
        quantize_mixed_precision(normalized_0, q_bits, q_min, mx, True)
    normalized = dequantize_mixed_precision(q_input, q_input_shape, q_bits, q_scale, q_min)
    grad_normalized = bn_grad_output * bn_weight.view(1, -1, 1, 1)

    mean_grad_normalized = grad_normalized.mean((0, 2, 3), keepdim=True)
    mean_grad = (normalized * grad_normalized).mean((0, 2, 3), keepdim=True)
    zmeans.append(mean_grad.detach().clone())
    grad_input = grad_normalized - mean_grad_normalized - normalized * mean_grad
    grad_input = grad_input / batch_std

    output = conv(input)
    grad = torch.autograd.grad(output, conv.weight, grad_input)[0]
    grads.append(grad.detach().clone())

grads = torch.stack(grads, 0)
grad_std = grads.std(0)
grad_var = (grad_std ** 2).sum()
print('Grad norm', (exact_grad**2).sum())
print('Grad var ', grad_var)
grad_mean = grads.mean(0)
print(grad_mean.norm(), (grad_mean - exact_grad).norm())

zmeans = torch.stack(zmeans, 0)
zstd = torch.std(zmeans, 0).view(-1)
zmean = torch.mean(zmeans, 0).view(-1)
D = normalized_0 / batch_std
Range = ((mx - q_min)**2 / 4 / 9).view(-1)
coeff_0 = (grad_normalized**2).sum((1,2,3))
myz_var = (Range * coeff_0).sum() / (N * H * W)**2 / C_out

coeff = 0
for cin in range(C_in):
    Xmap = input[:, cin:cin+1, :, :]
    Xmap = Xmap[:, :, torch.arange(0, H, 2), :]
    Xmap = Xmap[:, :, :, torch.arange(0, W, 2)]
    prod = (D * Xmap).sum((0, 2, 3))
    prod = prod ** 2
    coeff = coeff + prod

# coeff = coeff * 7 * 7
# myvar = (zstd**2 * coeff).sum()
coeff = (D**2).sum() * (input**2).sum() * 7 * 7
myvar = (zstd**2).mean() * coeff
print('Myvar ', myvar)

coeff_2 = 7 * 7 * (zmean**2).sum() * (input**2).sum((1,2,3))
myvar2 = (Range * coeff_2).sum()
print(myvar2)

print(myvar + myvar2)
