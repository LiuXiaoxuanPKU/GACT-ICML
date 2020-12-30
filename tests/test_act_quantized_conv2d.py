"""Test the activation quantized convolution layer"""

import math

import numpy as np
import torch
from torch import nn, autograd
from torch.nn import init, functional as F

import pytorch_minimax

from timeit_v2 import py_benchmark

from quantize.conf import config
from quantize.qscheme import QScheme
from quantize.ops import ext_backward_func, ext_quantization, get_memory_usage
from quantize.ops import conv2d as quantized_conv2d

def compute_tensor_bytes(x):
    assert x.dtype in [torch.float32, torch.int]
    return np.prod(x.size()) * 4


def test_relu_correctness():
    data_np = np.random.randn(128, 56, 56, 31).astype('float32')

    def test_implementation(func):
        data = torch.tensor(data_np).to("cuda").requires_grad_()

        output = func(data)
        output.backward(torch.ones_like(output))

        return [x.detach().cpu().numpy() for x in [output, data.grad]]

    output_ref, grad_data_ref =  test_implementation(F.relu)
    output_us, grad_data_us = test_implementation(ext_quantization.act_quantized_relu)

    print("========== ReLU Correctness Test ==========")
    np.testing.assert_allclose(output_ref, output_us)
    np.testing.assert_allclose(grad_data_ref, grad_data_us)


def test_relu_speed():
    data_np = np.random.randn(256, 56, 56, 32).astype('float32')

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
    forward_us, backward_us = test_implementation(ext_quantization.act_quantized_relu)

    print("========== ReLU Speed Test ==========")
    print("Reference. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
            (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
    print("Ours.      forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
            (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))


def test_relu_memory():
    data_np = np.random.randn(128, 56, 56, 32).astype('float32')

    def test_implementation(func):
        data = torch.tensor(data_np).to("cuda").requires_grad_()

        before = get_memory_usage()

        for i in range(10):
            data = func(data)

        after = get_memory_usage()
        
        return after - before

    usage_ref = test_implementation(F.relu)
    usage_us = test_implementation(ext_quantization.act_quantized_relu)

    print("========== ReLU Memory Test ==========")
    print("Reference. Usage: %.2f MB" % (usage_ref / 2 ** 20))
    print("Ours.      Usage: %.2f MB" % (usage_us / 2 ** 20))


def test_conv2d_correctness():
    """Test the correctness of computation results"""

    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 4, 28, 28, 256, 256, 3, 1, 1, 1, 1
    #N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 1, 14, 14, 256, 1, 3, 1, 1, 1, 1
    data_np = np.random.randn(N, CI, H, W).astype('float32')
    weight_np = np.random.randn(CO, CI // groups, kernel_size, kernel_size).astype('float32')
    bias_np = np.random.randn(CO).astype('float32')

    def test_implementation(func):
        torch.manual_seed(0)
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        weight = torch.tensor(weight_np).to("cuda").requires_grad_()
        bias = torch.tensor(bias_np).to("cuda").requires_grad_()

        scheme = QScheme(num_locations=kernel_size**2)
        output = func(data, weight, bias, stride, padding, dilation, groups, scheme)

        output.backward(torch.ones_like(output))

        return [x.detach().cpu().numpy() for x in [output, data.grad, weight.grad, bias.grad]]

    config.simulate = True
    output_ref, grad_data_ref, grad_weight_ref, grad_bias_ref = test_implementation(quantized_conv2d.apply)
    config.simulate = False
    output_us, grad_data_us, grad_weight_us, grad_bias_us = test_implementation(quantized_conv2d.apply)

    atol = 8e-2
    rtol = 8e-2
    print("========== Conv2d Correctness Test ==========")
    np.testing.assert_allclose(output_ref, output_us, atol=atol, rtol=rtol)
    np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol, rtol=rtol)
    np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol, rtol=rtol)
    np.testing.assert_allclose(grad_bias_ref, grad_bias_us, atol=atol, rtol=rtol)


def test_conv2d_speed():
    """Test the speed of convolution layer"""

    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 128, 28, 28, 256, 256, 3, 1, 1, 1, 1
    #N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 128, 28, 28, 256, 256, 1, 1, 0, 1, 1
    data_np = np.random.randn(N, CI, H, W).astype('float32')
    weight_np = np.random.randn(CO, CI // groups, kernel_size, kernel_size).astype('float32')
    bias_np = np.random.randn(CO).astype('float32')

    scheme = QScheme(num_locations=kernel_size**2)

    def test_implementation(func, stride, padding, dilation, groups):
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        weight = torch.tensor(weight_np).to("cuda").requires_grad_()
        bias = torch.tensor(bias_np).to("cuda").requires_grad_()

        if func == quantized_conv2d.apply:
            output = func(data, weight, bias, stride, padding, dilation, groups, scheme)
            stmt = "func(data, weight, bias, stride, padding, dilation, groups, scheme)"
        else:
            output = func(data, weight, bias, stride, padding, dilation, groups)
            stmt = "func(data, weight, bias, stride, padding, dilation, groups)"

        t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                 setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

        head = torch.ones_like(output)
        stmt = "output.backward(head, retain_graph=True)"
        t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                  setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

        return t_forward, t_backward

    forward_ref, backward_ref = test_implementation(F.conv2d, stride, padding, dilation, groups)
    config.simulate = True
    forward_sim, backward_sim = test_implementation(quantized_conv2d.apply, stride, padding, dilation, groups)
    config.simulate = False
    forward_us, backward_us = test_implementation(quantized_conv2d.apply, stride, padding, dilation, groups)

    print("========== Conv2d Speed Test ==========")
    print("Reference.  forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
            (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
    print("Simulation. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
            (forward_sim * 1e3, backward_sim * 1e3, (forward_sim + backward_sim) * 1e3))
    print("Ours.       forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
            (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))


def test_conv2d_memory_analytical():
    """Compute the memory of activation analytically"""

    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 256, 28, 28, 256, 256, 3, 1, 1, 1, 1
    data_np = np.random.randn(N, CI, H, W).astype('float32')
    weight_np = np.random.randn(CO, CI // groups, kernel_size, kernel_size).astype('float32')
    bias_np = np.random.randn(CO).astype('float32')

    scheme = QScheme(num_locations=kernel_size**2)

    def test_implementation(conv_func, relu_func, n_layers=10):
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        weight = torch.tensor(weight_np).to("cuda").requires_grad_()
        bias = torch.tensor(bias_np).to("cuda").requires_grad_()

        # allocate input and weights
        data = torch.tensor(data_np).to("cuda").requires_grad_(False)
        weights = []
        for i in range(n_layers):
            weight = torch.tensor(weight_np).to("cuda").requires_grad_()
            weights.append(weight)

        before_size = get_memory_usage(False)

        # forward n convolution layers
        output = data
        for i in range(n_layers):
            if conv_func == quantized_conv2d.apply:
                output = conv_func(output, weights[i], None, stride, padding, dilation, groups, scheme)
            else:
                output = conv_func(output, weights[i], None, stride, padding, dilation, groups)
            output = relu_func(output)
        output = output.sum()

        after_size = get_memory_usage(False)
        output_size = compute_tensor_bytes(output)

        return after_size / 1024**2, (after_size - before_size - output_size) / 1024**2

    total_size_ref, act_size_ref = test_implementation(F.conv2d, lambda x: F.relu(x, inplace=True))
    config.simulate = True
    total_size_sim, act_size_sim = test_implementation(quantized_conv2d.apply, ext_quantization.act_quantized_relu)
    config.simulate = False
    total_size_us, act_size_us = test_implementation(quantized_conv2d.apply, ext_quantization.act_quantized_relu)

    print("========== Conv2d Activation Memory Test (bits = %d) ==========" % (config.activation_compression_bits))
    print("Reference.  Total: %7.2f MB\tAct: %7.2f MB" % (total_size_ref, act_size_ref))
    print("Simulation. Total: %7.2f MB\tAct: %7.2f MB" % (total_size_sim, act_size_sim))
    print("Ours.       Total: %7.2f MB\tAct: %7.2f MB" % (total_size_us, act_size_us))


def test_conv2d_memory_max_batch_size():
    """Find the maximum batch size by gradually increasing the batch size until hitting Out-of-memory error"""

    for device in ["cuda"]:
        def test_implementation(func, n_layers, batch_sizes):
            def run_batch_size(batch_size):
                N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = batch_size, 28, 28, 256, 256, 3, 1, 1, 1, 1
                data_np = np.random.uniform(size=(N, CI, H, W)).astype('float32')
                weight_np = np.random.uniform(size=(CO, CI // groups, kernel_size, kernel_size)).astype('float32')
                bias_np = np.random.uniform(size=(CO,)).astype('float32')

                # allocate input and weights
                data = torch.tensor(data_np).to("cuda").requires_grad_(False)
                weights = []
                for i in range(n_layers):
                    weight = torch.tensor(weight_np).to("cuda").requires_grad_()
                    weights.append(weight)

                before_size = get_memory_usage(False)
    
                # forward n convolution layers
                output = data
                for i in range(n_layers):
                    output = func(output, weights[i], None, stride, padding, dilation, groups)
                output = output.sum()

                after_size = get_memory_usage(False)
                output_size = compute_tensor_bytes(output)
    
                return after_size / 1024**2, (after_size - before_size - output_size) / 1024**2

            # try gradually increased batch sizes
            try:
                for i, batch_size in enumerate(batch_sizes):
                    total_size_ref, act_size_ref = run_batch_size(batch_size)
                    print("batch_size: %4d\t" % batch_size, end="")
                    print("total_memory: %7.2f MB\tact_memory: %7.2f MB" % (total_size_ref, act_size_ref))
            except RuntimeError:
                pass
            finally:
                print("Maximum batch size: %d" % (batch_sizes[i-1]))
       
        print("========== Conv2d Batch Size Test ==========")
        print("---> Reference")
        test_implementation(F.conv2d, n_layers=50, batch_sizes=[100, 200, 250, 300, 350, 400, 450, 500, 1000])
        print("---> Ours")
        test_implementation(act_quantized_conv2d.apply, n_layers=50, batch_sizes=[100, 200, 250, 500, 1000, 2200, 2300, 2400, 3000, 4000])


if __name__ == "__main__":
    #test_relu_correctness()
    #test_relu_speed()
    #test_relu_memory()

    #test_conv2d_correctness()

    #config.activation_compression_bits = 2
    #test_conv2d_speed()

    config.activation_compression_bits = 2
    test_conv2d_memory_analytical()

    #config.activation_compression_bits = 2
    #test_conv2d_memory_max_batch_size()

