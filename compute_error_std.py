import torch
import sys
from image_classification.utils import *

num_samples = 10


errors = []
for iter in range(num_samples):
    data = np.load('errors_{}.pkl.npz'.format(iter), allow_pickle=True)
    n = len(data)
    if not errors:
        errors = [[] for i in range(n)]

    print(iter)
    for i in range(n):
        elem = data['arr_{}'.format(i)][0]
        errors[i].append(elem)

for i in range(n):
    errors[i] = np.stack(errors[i], 0)
    print(np.std(errors[i], 0).mean())
