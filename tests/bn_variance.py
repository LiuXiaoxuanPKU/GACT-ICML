import torch
from tqdm import tqdm
from quantize import config, QScheme, QBNScheme, QBatchNorm2d, quantize_mixed_precision, dequantize_mixed_precision
import pytorch_minimax

inp, weight, bias, grad_output, grad_input = torch.load('pts/bn_layer_0.pt')
weight = weight.view(-1)
# inp.requires_grad_()
N, C, H, W = inp.shape

config.activation_compression_bits = 2
config.initial_bits = 8
config.compress_activation = True
# num_bins = 3
QBNScheme.num_samples = N
QBNScheme.batch = torch.arange(0, N)
eps = 1e-5

model = QBatchNorm2d(C).cuda()
with torch.no_grad():
    model.weight.copy_(weight)
    model.bias.copy_(bias)


config.compress_activation = False
output = model(inp)
grad = torch.autograd.grad(output, inp, grad_output)[0]
exact_grad = grad.detach().clone()
print('Grad error ', ((exact_grad - grad_input)**2).sum())

QBNScheme.update_scale = False
config.compress_activation = True
grads = []
quantized = []
batch_mean = inp.mean((0, 2, 3), keepdim=True)
batch_std = torch.sqrt(inp.var((0, 2, 3), keepdim=True) + eps)
normalized = (inp - batch_mean) / batch_std
q_bits, q_min, mx = model.scheme.compute_quantization_bits(normalized)

for iter in range(10):
    output = model(inp)
    grad = torch.autograd.grad(output, inp, grad_output)[0]
    grads.append(grad.detach().clone())

    q_input, q_input_shape, q_bits, q_scale = \
        quantize_mixed_precision(normalized, q_bits, q_min, mx, True)
    normalized_1 = dequantize_mixed_precision(q_input, q_input_shape, q_bits, q_scale, q_min)
    quantized.append(normalized_1)

weight = weight.view(1, -1, 1, 1)
grad_normalized = grad_output * weight

grads = torch.stack(grads, 0)
grad_std = grads.std(0)
grad_var = (grad_std ** 2).sum()
print('Grad norm ', (exact_grad**2).sum())
print('Grad var ', grad_var.cpu().numpy())
grad_mean = grads.mean(0)
print('Grad bias ', ((exact_grad - grad_mean)**2).sum())
print(model.scheme.var)

quantized = torch.stack(quantized, 0)
quantized_std = quantized.std(0)

myvar = 0
for d in range(C):
    myvar += (normalized[:, d, :, :]**2).sum() * \
             (quantized_std[:, d, :, :]**2 * grad_normalized[:, d, :, :]**2).sum()

print(myvar / (N*H*W)**2)

