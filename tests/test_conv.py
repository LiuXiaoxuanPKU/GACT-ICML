import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from actnn import config, QConv2d, QConvTranspose2d

torch.manual_seed(0)

config.activation_compression_bits = [2]
# config.perlayer = False
# config.initial_bits = 2
# config.pergroup = False
in_channels = 100
out_channels = 4
kernel_size = 3
stride = 2
groups = 2
layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, groups=groups).cuda()
qlayer = QConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, groups=groups).cuda()
with torch.no_grad():
    qlayer.weight.copy_(layer.weight)
    qlayer.bias.copy_(layer.bias)

print(qlayer.weight.shape)
x = torch.rand([10, in_channels, 160, 200]).cuda()
y = target = torch.empty(10, dtype=torch.long).random_(4).cuda()
print(x.shape, y)
ce = nn.CrossEntropyLoss().cuda()

def get_grad(model):
    pred = model(x)
    pred = F.relu(pred)
    pred = pred.mean([2, 3])
    loss = ce(pred, y)
    model.weight.grad = None
    model.bias.grad = None
    loss.backward()
    return model.weight.grad.cpu().numpy()

true_grad = get_grad(layer)
grads = []
for i in range(10):
    grads.append(get_grad(qlayer))

grads = np.stack(grads, 0)
grad_mean = grads.mean(0)
grad_std = grads.std(0)

bias = np.linalg.norm(grad_mean - true_grad)
print('Grad = {}, Bias = {}, Std = {}'.format(np.linalg.norm(true_grad), bias, np.linalg.norm(grad_std)))
