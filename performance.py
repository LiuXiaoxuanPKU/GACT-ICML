import torch
import time

a = torch.randn(32, 3, 224, 224).cuda()
a = a.view(32, -1)

t = time.time()
s = 0
for i in range(1000):
    s += a.max()

print(s, time.time() - t)

s = 0
t = time.time()
for i in range(1000):
    s += a.max(1)[0].sum()
print(s, time.time() - t)

t = time.time()
s = 0
for i in range(1000):
    s += a.sum()

print(s, time.time() - t)

s = 0
t = time.time()
for i in range(1000):
    s += a.sum(1).sum()
print(s, time.time() - t)
