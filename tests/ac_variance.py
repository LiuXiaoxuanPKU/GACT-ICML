import torch
from image_classification.quantize import MixedPrecisionQuantize
from quantizers import calc_precision, calc_precision_dp
torch.set_printoptions(linewidth=250, sci_mode=False, precision=3)
torch.manual_seed(0)

N = 8
D = 20
target = 2

H = torch.randn(N, D)
grad = torch.tensor([10.0, 2.0, 0.5, 0.2, 0.1, 0.05, 0.03, 0.01])
grad = grad.view(N, 1) * torch.randn(N, D)
keep_prob = torch.ones(8)
b = torch.ones(N, dtype=torch.int32) * 8

H_sum = (H**2).sum(1)
grad_sum = (grad**2).sum(1)
Range = H.max(1)[0] - H.min(1)[0]
C = D / 4 * Range**2 * grad_sum
A = H_sum * grad_sum

# bits = calc_precision(b, C, target * N).float()
#    torch.tensor([10.0, 10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0])
bits, keep_prob = calc_precision_dp(A, C, 8, target, 2)

print('Keep prob = ', keep_prob)
print('bits = ', bits)

def quantize(H, grad, keep_prob, bits):
    m = torch.distributions.Bernoulli(probs=keep_prob).sample() \
        / keep_prob
    qH = MixedPrecisionQuantize().apply(H.cuda(), bits.cuda(), True).cpu()
    H = qH * m.view(N, 1)
    return H.transpose(0, 1) @ grad


def variance(H, grad, keep_prob, bits):
    H_sum = (H**2).sum(1)
    grad_sum = (grad**2).sum(1)
    dropout_var = ((1 - keep_prob) / keep_prob * H_sum * grad_sum).sum()
    R = H.max(1)[0] - H.min(1)[0]
    B = 2**bits - 1
    quantization_var = (D / 8 * (R**2 / keep_prob / B**2) * grad_sum).sum()
    return dropout_var + quantization_var


print(grad)
batch_mean = quantize(H, grad, keep_prob=torch.ones(N),
                      bits=torch.ones(N) * 16)
var = variance(H, grad, keep_prob, bits)
print('Variance = {}'.format(var))

results = []
for i in range(1000):
    results.append(quantize(H, grad, keep_prob, bits))
results = torch.stack(results, 0)
sample_mean = results.mean(0)
std = results.std(0)
print('Sample Bias = {}, Sample Var = {}'.format(((sample_mean-batch_mean)**2).sum(),
                                   (std**2).sum()))
