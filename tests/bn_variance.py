import torch
from tqdm import tqdm
from quantize import config, QScheme, QBatchNorm2d, quantize_mixed_precision, dequantize_mixed_precision
import pytorch_minimax

inp, weight, bias, grad_output, grad_input = torch.load('../resnet/bn_layer_0.pt')
weight = weight.view(-1)
inp.requires_grad_()
N, C, H, W = inp.shape

config.activation_compression_bits = 2
config.initial_bits = 8
config.compress_activation = True
# num_bins = 3
QScheme.num_samples = N
QScheme.batch = torch.arange(0, N)

model = QBatchNorm2d(C).cuda()
with torch.no_grad():
    model.weight.copy_(weight)
    model.bias.copy_(bias)


config.compress_activation = False
output = model(inp)
grad = torch.autograd.grad(output, inp, grad_output)[0]
exact_grad = grad.detach().clone()
print('Grad error ', ((exact_grad - grad_input)**2).sum())

QScheme.update_scale = False
config.compress_activation = True
grads = []
for iter in range(10):
    output = model(inp)
    grad = torch.autograd.grad(output, inp, grad_output)[0]
    grads.append(grad.detach().clone())

grads = torch.stack(grads, 0)
grad_std = grads.std(0)
grad_var = (grad_std ** 2).sum()
print('Grad norm ', (exact_grad**2).sum())
print('Grad var ', grad_var.cpu().numpy())
grad_mean = grads.mean(0)
print('Grad bias ', ((exact_grad - grad_mean)**2).sum())

print(model.scheme.var)

# grad_sum = (grad_output.view(N, -1) ** 2).sum(1)
# input_flatten = inp.view(N, -1)
# mn = pytorch_minimax.min(input_flatten)
# mx = pytorch_minimax.max(input_flatten)
# Range = mx - mn
# var = num_locations * C_in / 4 * (Range ** 2 * grad_sum).sum() / num_bins**2
# # var = (C / 9).sum()
# print(var)

