import numpy as np
import torch
from C import calc_precision

b = torch.from_numpy(np.array([5, 5, 5], dtype=np.int32))
C = torch.from_numpy(np.array([0.001, 1.0, 0.1], dtype=np.float32))
w = torch.from_numpy(np.array([1, 10, 1], dtype=np.int32))

target = np.array(2 * w.sum(), dtype=np.int32)

b = calc_precision(b, C, w, target)
