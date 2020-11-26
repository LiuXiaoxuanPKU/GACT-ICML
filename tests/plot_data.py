import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


old_data, old_weights, old_output, _ = torch.load('new_scales/old_data.pt')
new_data, new_weights, new_output, targets = torch.load('new_scales/new_data.pt')
targets = targets[0].cuda()

old_output_softm = old_output.exp()
new_output_softm = new_output.exp()
old_output_softm = old_output_softm / old_output_softm.sum(1, keepdims=True)
new_output_softm = new_output_softm / new_output_softm.sum(1, keepdims=True)


def rel_dist(a, b):
    # N = a.shape[0]
    # a = a.view([N, -1])
    # b = b.view([N, -1])
    return (a - b).norm() / a.norm()

print('Output dist ', rel_dist(new_output, old_output))
print('Output softmax dist ', rel_dist(new_output_softm, old_output_softm))

for old_layer, new_layer, old_weight, new_weight in zip(old_data, new_data, old_weights, new_weights):
    _, old_input, _, old_grad_input = old_layer
    _, new_input, _, new_grad_input = new_layer
    N = old_input.shape[0]

    old_grad_scale = (old_grad_input.view(N, -1) ** 2).sum(1)
    new_grad_scale = (new_grad_input.view(N, -1) ** 2).sum(1)
    print('Input dist {}, weight dist {}, grad dist {}, grad scale dist {}'.format(
          rel_dist(old_input, new_input), rel_dist(old_weight, new_weight),
          rel_dist(old_grad_input, new_grad_input), rel_dist(old_grad_scale, new_grad_scale)))


def accuracy(pred, targets):
    pred = pred.max(1)[1]
    return torch.eq(pred, targets).float().mean()

print('Old accuracy ', accuracy(old_input, targets))
print('New accuracy ', accuracy(new_input, targets))

y = torch.eye(10)[targets].cuda()
my_old_grad = y - old_output_softm
my_new_grad = y - new_output_softm
my_old_grad = (my_old_grad ** 2).sum(1)
my_new_grad = (my_new_grad ** 2).sum(1)
my_old_grad /= my_old_grad.sum()
my_new_grad /= my_new_grad.sum()

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(my_old_grad.detach().cpu(), '.', label='old')
ax.plot(my_new_grad.detach().cpu(), 'x', label='new')
ax.set_yscale('log')
ax.legend()

fig.savefig('scale.pdf')