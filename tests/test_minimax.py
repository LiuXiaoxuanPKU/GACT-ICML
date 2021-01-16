import math

import numpy as np
import torch
from torch import nn, autograd
from torch.nn import init, functional as F
from torch.autograd.function import Function

from timeit_v2 import py_benchmark

from quantize import QScheme, QBNScheme, config, get_memory_usage, compute_tensor_bytes
from quantize.ops import ext_backward_func, ext_quantization, ext_minimax
from quantize.ops import conv2d as quantized_conv2d, batch_norm as quantized_batch_norm, \
        adaptive_avg_pool2d as quantized_adaptive_avg_pool2d


def test_minimax_correctness():
    data_np = np.random.randn(1024, 256).astype('float32')

    def test_implementation(func):
        data = torch.tensor(data_np).to("cuda")

        if func == torch:
            mn, mx = torch.min(data, 1)[0], torch.max(data, 1)[0]
        else:
            mn, mx = ext_minimax.minimax(data)[:2]

        return [x.detach().cpu().numpy() for x in [mn, mx]]

    mn_ref, mx_ref =  test_implementation(torch)
    mn_us, mx_us = test_implementation(ext_minimax)

    print("========== Minimax Correctness Test ==========")
    np.testing.assert_allclose(mn_ref, mn_us)
    np.testing.assert_allclose(mx_ref, mx_us)


def test_minimax_speed():
    data_np = np.random.randn(128 * 256, 256).astype('float32')

    def test_implementation(func):
        data = torch.tensor(data_np).to("cuda")

        if func == torch:
            stmt = "torch.min(data, 1)[0], torch.max(data, 1)[0]"
        else:
            stmt = "ext_minimax.minimax(data)"

        cost = py_benchmark(stmt, {**globals(), **locals()},
                            setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
        return cost

    cost_ref =  test_implementation(torch)
    cost_us = test_implementation(ext_minimax)

    print("========== Minimax Speed Test ==========")
    print("PyTorch.  Cost: %.3f ms" % (cost_ref * 1e3))
    print("Ous.      Cost: %.3f ms" % (cost_us * 1e3))


if __name__ == "__main__":
    test_minimax_correctness()
    test_minimax_speed()

