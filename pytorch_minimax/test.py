from IPython import get_ipython

import torch as th
import pytorch_minimax

ipython = get_ipython()
N, D = 32, 3 * 224 * 224

x = th.randn(N, D, device=th.device('cuda:0'))

assert th.allclose(x.min(1)[0], pytorch_minimax.min(x))
ipython.magic('timeit x.min(0)')
ipython.magic('timeit x.min(1)')
ipython.magic('timeit pytorch_minimax.min(x)')  # TODO Can we ignore this?
ipython.magic('timeit pytorch_minimax.min(x)')

assert th.allclose(x.max(1)[0], pytorch_minimax.max(x))
ipython.magic('timeit x.max(0)')
ipython.magic('timeit x.max(1)')
ipython.magic('timeit pytorch_minimax.max(x)')  # TODO Can we ignore this?
ipython.magic('timeit pytorch_minimax.max(x)')
