import numpy as np
import pickle
import torch
import torch.nn.functional as F
from quantize.preconditioner import ScalarPreconditioner

iacts = np.load('iacts.pkl.npz', allow_pickle=True)
acts = np.load('acts.pkl.npz', allow_pickle=True)
exact_iacts = np.load('exact_iacts.pkl.npz', allow_pickle=True)
exact_acts = np.load('exact_acts.pkl.npz', allow_pickle=True)
weights = np.load('weights.pkl.npz', allow_pickle=True)
exact_weights = np.load('exact_weights.pkl.npz', allow_pickle=True)
with open('layer_names.pkl', 'rb') as f:
    names = pickle.load(f)

for i in range(2):
    print('-----')
    ia = 'arr_{}'.format(i)
    print(np.linalg.norm(weights[ia] - exact_weights[ia]) / np.linalg.norm(exact_weights[ia]))
    print(np.linalg.norm(iacts[ia] - exact_iacts[ia]) / np.linalg.norm(exact_iacts[ia]))
    print(np.linalg.norm(acts[ia] - exact_acts[ia]) / np.linalg.norm(exact_acts[ia]))

print(np.squeeze(weights['arr_0']))
print(np.squeeze(exact_weights['arr_0']))

ia = 'arr_0'
qinput = torch.tensor(iacts[ia])
input = torch.tensor(exact_iacts[ia])
qoutput = torch.tensor(acts[ia])
output = torch.tensor(exact_acts[ia])

print(qoutput[0, 10])
print(output[0, 10])

my_qoutput = F.conv2d(qinput, torch.tensor(weights[ia]))
my_output = F.conv2d(input, torch.tensor(exact_weights[ia]))

a = torch.randn(100)
pred = ScalarPreconditioner(a, 8)
a[0] = 0.0
print(pred.transform(a))
