"""Test the activation quantized ops"""

import math

import numpy as np
import torch
from torch import nn, autograd
from torch.nn import init, functional as F
from torch.autograd.function import Function

from timeit_v2 import py_benchmark

from actnn import get_memory_usage, compute_tensor_bytes
from actnn.ops import ext_quantization, op_quantize, op_dequantize


def error_rate(q_input, input):
    print(((q_input - input)**2).sum() / (input**2).sum())


def test_quantize_error():
    input_shape = (64, 3, 224, 224)
    input = torch.rand(input_shape).to("cuda")
    print("==========  Quantization Error Rate Test ==========")
    print("1 bit error rate: ")
    error_rate(op_dequantize(op_quantize(input, 1), input_shape), input)
    print("2 bit error rate: ")
    error_rate(op_dequantize(op_quantize(input, 2), input_shape), input)
    print("4 bit error rate: ")
    error_rate(op_dequantize(op_quantize(input, 4), input_shape), input)
    print("8 bit error rate: ")
    error_rate(op_dequantize(op_quantize(input, 8), input_shape), input)


def test_relu_correctness():
    print("========== ReLU Correctness Test ==========")

    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")
        data_np = np.random.randn(128, 56, 56, 31).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            output = func(data)
            output.backward(torch.ones_like(output))

            return [x.detach().cpu().numpy() for x in [output, data.grad]]

        output_ref, grad_data_ref = test_implementation(F.relu)
        output_us, grad_data_us = test_implementation(
            ext_quantization.act_quantized_relu)

        np.testing.assert_allclose(output_ref, output_us)
        np.testing.assert_allclose(grad_data_ref, grad_data_us)


def test_relu_memory():
    print("========== ReLU Memory Test ==========")

    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")
        data_np = np.random.randn(128, 56, 56, 32).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            before = get_memory_usage()

            for i in range(10):
                data = func(data)

            after = get_memory_usage()

            return after - before

        usage_ref = test_implementation(F.relu)
        usage_us = test_implementation(ext_quantization.act_quantized_relu)

        print("Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
        print("Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))


def test_relu_speed():
    print("========== ReLU Speed Test ==========")

    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")

        data_np = np.random.randn(256, 56, 56, 32).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            stmt = "func(data)"
            t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            output = func(data)
            head = torch.ones_like(output)
            stmt = "output.backward(head, retain_graph=True)"
            t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            return t_forward, t_backward

        forward_ref, backward_ref = test_implementation(F.relu)
        forward_us, backward_us = test_implementation(
            ext_quantization.act_quantized_relu)

        print("Exact.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
        print("Quantized. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))


if __name__ == "__main__":
    test_quantize_error()
    # test_relu_correctness()
    # test_relu_memory()
    # test_relu_speed()
