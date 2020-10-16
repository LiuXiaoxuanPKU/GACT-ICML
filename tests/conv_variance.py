import torch
from tqdm import tqdm
from quantize import config, QScheme, QConv2d, quantize_mixed_precision, dequantize_mixed_precision
import pytorch_minimax

input, weight, grad_output, grad_weight = torch.load('../resnet/layer_0.pt')
N, C_in, H, W = input.shape
_, C_out, _, _ = grad_output.shape

config.activation_compression_bits = 2
config.initial_bits = 2
config.compress_activation = True
num_bins = 3
num_locations = 7 * 7
QScheme.num_samples = N
QScheme.update_scale = False
QScheme.batch = torch.arange(0, 32)

model = QConv2d(C_in, C_out, 7, stride=2, padding=3, bias=False).cuda()
with torch.no_grad():
    model.weight.copy_(weight)

config.compress_activation = False
output = model(input)
grad = torch.autograd.grad(output, model.weight, grad_output)[0]
exact_grad = grad.detach().clone()

config.compress_activation = True
grads = []
for iter in range(100):
    output = model(input)
    grad = torch.autograd.grad(output, model.weight, grad_output)[0]
    grads.append(grad.detach().clone())

grads = torch.stack(grads, 0)
grad_std = grads.std(0)
grad_var = (grad_std ** 2).sum()
print((exact_grad**2).sum())
print(grad_var)

grad_sum = (grad_output.view(N, -1) ** 2).sum(1)
input_flatten = input.view(N, -1)
mn = pytorch_minimax.min(input_flatten)
mx = pytorch_minimax.max(input_flatten)
Range = mx - mn
var = num_locations * C_in / 4 * (Range ** 2 * grad_sum).sum() / num_bins**2
# var = (C / 9).sum()
print(var)


# def get_grad(grad_output, input):
#     grad = torch.zeros_like(weight)
#     for kh in range(3):
#         for kw in range(3):
#             dh = kh - 1
#             dw = kw - 1
#             for h in range(H):
#                 for w in range(W):
#                     h0 = h + dh
#                     w0 = w + dw
#                     if not (h0 in range(H)) or not (w0 in range(W)):
#                         continue
#
#                     grad[:, :, kh, kw] += grad_output[:, :, h, w].t() @ input[:, :, h0, w0]
#
#     return grad
#
#
# grads = []
# scheme = model.scheme
# for iter in tqdm(range(10)):
#     q_bits, q_min, mx = scheme.compute_quantization_bits(input)
#     q_input, q_input_shape, q_bits, q_scale = \
#         quantize_mixed_precision(input, q_bits, q_min, mx, True)
#     input2 = dequantize_mixed_precision(q_input, q_input_shape, q_bits, q_scale, q_min)
#     grads.append(get_grad(grad_output, input2))
#     # grads.append(input2)
#
# grads = torch.stack(grads, 0)
# grad_std = grads.std(0)
# grad_var = (grad_std ** 2).sum()
# print(grad_var)

# total_var = 0
# for kh in range(3):
#     for kw in range(3):
#         dh = kh - 1
#         dw = kw - 1
#         print(kh, kw)
#         for h in range(H):
#             for w in range(W):
#                 h0 = h + dh
#                 w0 = w + dw
#                 if not (h0 in range(H)) or not (w0 in range(W)):
#                     continue
#
#                 grads = []
#                 for iter in range(10):
#                     q_bits, q_min, mx = scheme.compute_quantization_bits(input)
#                     q_input, q_input_shape, q_bits, q_scale = \
#                         quantize_mixed_precision(input, q_bits, q_min, mx, True)
#                     input2 = dequantize_mixed_precision(q_input, q_input_shape, q_bits, q_scale, q_min)[:, :, h0, w0]
#                     grads.append(grad_output[:, :, h, w].t() @ input2)
#
#                 grads = torch.stack(grads, 0)
#                 grad_std = grads.std(0)
#                 total_var += (grad_std ** 2).sum()
#
# print(total_var)


# total_var = 0
# for kh in range(3):
#     for kw in range(3):
#         dh = kh - 1
#         dw = kw - 1
#         print(kh, kw)
#         for h in range(H):
#             for w in range(W):
#                 h0 = h + dh
#                 w0 = w + dw
#                 if not (h0 in range(H)) or not (w0 in range(W)):
#                     continue
#
#                 _, q_min, mx = scheme.compute_quantization_bits(input)
#                 Range = (mx - q_min).view(-1)
#                 grad_sum = (grad_output[:,:,h,w]**2).sum(1)
#                 total_var += C_in * (Range**2 * grad_sum).sum() / 4 / 9
#
# print(total_var)


# grads = []
# h = 16
# w = 16
# h0 = 0
# w0 = 0
# for iter in range(10):
#     q_bits, q_min, mx = scheme.compute_quantization_bits(input)
#     # q_min = q_min.min(keepdim=True)
#     # mx = mx.max(keepdim=True)
#     q_input, q_input_shape, q_bits, q_scale = \
#         quantize_mixed_precision(input, q_bits, q_min, mx, True)
#     input2 = dequantize_mixed_precision(q_input, q_input_shape, q_bits, q_scale, q_min)[:, :, h0, w0]
#     grads.append(grad_output[:, :, h, w].t() @ input2)
#
# grads = torch.stack(grads, 0)
# grad_std = grads.std(0)
# total_var = (grad_std ** 2).sum()
# print(total_var)
#
# _, q_min, mx = scheme.compute_quantization_bits(input.clone())
# Range = (mx - q_min).view(-1)
# grad_sum = (grad_output[:,:,h,w]**2).sum(1)
# total_var = C_in * (Range**2 * grad_sum).sum() / 4 / num_bins**2
# print(total_var)


# grads = []
# h = 16
# w = 16
# h0 = 0
# w0 = 0
# i = 0
# j = 0
# for iter in range(10):
#     q_bits, q_min, mx = scheme.compute_quantization_bits(input)
#     q_input, q_input_shape, q_bits, q_scale = \
#         quantize_mixed_precision(input, q_bits, q_min, mx, True)
#     input2 = dequantize_mixed_precision(q_input, q_input_shape, q_bits, q_scale, q_min)[:, :, h0, w0]
#     A = grad_output[:, :, h, w].t()
#     B = input2
#     grads.append((A[i, :] * B[:, j]).sum())
#
# grads = torch.stack(grads, 0)
# grad_std = grads.std(0)
# total_var = (grad_std ** 2).sum()
# print(total_var)
#
# A = grad_output[:, :, h, w].t()[i, :]
# B = input[:,:,h0,w0].clone()[:, j]
# Range = B.max() - B.min()
# grad_sum = (A**2).sum()
# total_var = (Range**2 * grad_sum) / 4 / 9
# print(total_var)

# total_var = 0
# _, q_min, mx = scheme.compute_quantization_bits(input.clone())
# Range = (mx - q_min).view(-1)
# for kh in range(3):
#     for kw in range(3):
#         dh = kh - 1
#         dw = kw - 1
#         print(kh, kw)
#         for h in range(H):
#             for w in range(W):
#                 h0 = h + dh
#                 w0 = w + dw
#                 if not (h0 in range(H)) or not (w0 in range(W)):
#                     continue
#
#                 grad_sum = (grad_output[:,:,h,w]**2).sum(1)
#                 total_var += (Range**2 * grad_sum).sum() / 4 / 9
#
# print(total_var)

# _, q_min, mx = scheme.compute_quantization_bits(input.clone())
# Range = mx - q_min
# grad_sum = (grad_output**2).sum([1,2,3])
#
# total_var = (Range**2 * grad_sum).sum() / 4