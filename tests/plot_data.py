import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


old_data, old_output = torch.load('scales/old_data.pt')
new_data, new_output = torch.load('scales/new_data.pt')

# old_output = old_output.exp()
# new_output = new_output.exp()
# old_output = old_output / old_output.sum(1, keepdims=True)
# new_output = new_output / new_output.sum(1, keepdims=True)


def rel_dist(a, b):
    # N = a.shape[0]
    # a = a.view([N, -1])
    # b = b.view([N, -1])
    return (a - b).norm() / a.norm()

print('Output dist ', )
