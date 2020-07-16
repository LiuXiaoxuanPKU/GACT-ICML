import torch
from quantize import qsoftmax

A = torch.randn(10, 3)
A.requires_grad_()
y = torch.nn.functional.softmax(A)
w = torch.randn(10, 3)
loss = (y * w).sum()
loss.backward()
print(A.grad)

A.grad.zero_()
y = qsoftmax().apply(A, 1, 'a')
loss = (y * w).sum()
loss.backward()
print(A.grad)
