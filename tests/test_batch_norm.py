import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize import config, QF, batch_norm


x = torch.randn([128, 3, 6, 6]) * 0.01
x.requires_grad_()
w = torch.randn([128, 3, 6, 6])

bn = nn.BatchNorm2d(3)
bn.train()
with torch.no_grad():
    bn.weight.copy_(torch.rand(3))
    bn.bias.copy_(torch.randn(3))

y = bn(x)
loss = (y * w).sum()

loss.backward()
print(loss, x.grad.norm(), bn.weight.grad, bn.bias.grad)

x.grad.zero_()
bn.weight.grad.zero_()
bn.bias.grad.zero_()

mybn = batch_norm()
y = mybn.apply(x, bn.running_mean, bn.running_var, bn.weight, bn.bias,
               True, bn.momentum, bn.eps)
loss = (y * w).sum()
loss.backward()
print(loss, x.grad.norm(), bn.weight.grad, bn.bias.grad)

# Inside
x.grad.zero_()
bn.weight.grad.zero_()
bn.bias.grad.zero_()

batch_mean = x.mean((0, 2, 3), keepdim=True)
batch_std = torch.sqrt(x.var((0, 2, 3), keepdim=True) + bn.eps)#.detach()

normalized = (x - batch_mean) / batch_std

weight = bn.weight.view(1, -1, 1, 1)
bias = bn.bias.view(1, -1, 1, 1)
my_output = normalized * weight + bias

loss = (my_output * w).sum()
loss.backward(retain_graph=True)
grad_output = torch.autograd.grad(loss, my_output)[0]

grad_weight = (grad_output * normalized).sum((0, 2, 3))
grad_bias = grad_output.sum((0, 2, 3))
grad_normalized = grad_output * weight

mean_grad_normalized = grad_normalized.mean((0, 2, 3), keepdim=True)
t1 = grad_normalized
t2 = mean_grad_normalized
t3 = normalized * (normalized * grad_normalized).mean((0, 2, 3), keepdim=True)
grad_input = grad_normalized - mean_grad_normalized - normalized * \
             (normalized * grad_normalized).mean((0, 2, 3), keepdim=True)
grad_input = grad_input / batch_std

print(grad_input.norm(), x.grad.norm())
