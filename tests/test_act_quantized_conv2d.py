"""Examples of Python implemtations that call c++ backward functions"""

import math
import time

import numpy as np
import torch
from torch import nn, autograd
from torch.nn import init, functional as F
from torch.nn.modules.utils import _pair
from torch.utils.cpp_extension import load

import pytorch_minimax
from quantize.conf import config
from quantize.C import calc_precision_dp, calc_precision, calc_avg_bits

from timeit_v2 import py_benchmark

ext_backward_func = load(name="ext_backward_func", sources=["ext_backward_func.cc"], verbose=True)
ext_quantization = load(name="ext_quantization",
        sources=["ext_quantization.cc", "ext_quantization_cuda_kernel.cu"], verbose=True)

SIMULATE = False

def compute_tensor_bytes(x):
    assert x.dtype in [torch.float32, torch.int]
    return np.prod(x.size()) * 4

def get_memory_usage(print_info=False):
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if print_info:
        print("allocated: %.2f MB" % allocated)
        print("reserved:  %.2f MB" % reserved)
    return allocated

def compute_quantization_bits(input, name):
    N = input.shape[0]
    D = input.shape[1]
    input_flatten = input.view(N, -1)

    # greedy
    grad_sum = torch.tensor(5e-5 * np.random.uniform(size=N).astype('float32')).cuda()  # QF.get_scale(name).cpu()
    mn = pytorch_minimax.min(input_flatten)
    mx = pytorch_minimax.max(input_flatten)
    Range = mx - mn
    C = D / 4 * Range ** 2 * grad_sum
    b = torch.ones(N, dtype=torch.int32) * config.initial_bits
    b = calc_precision(b, C.cpu(), config.activation_compression_bits * N)

    if SIMULATE:
        mn = mn.unsqueeze(1)
        mx = mx.unsqueeze(1)

    return b, mn, mx

def quantize_mixed_precision(data, bits, mn, mx, stochastic=False):
    assert not stochastic
    raw_shape = data.shape
    output = data.view(raw_shape[0], -1)

    if SIMULATE:
        bits = bits.cuda()
        B = (2 ** bits - 1).unsqueeze(1)
        mn = mn - 1e-8
        mx = mx - 1e-8
        scale = B / (mx - mn)
        output = (output - mn) * scale

        output = F.relu(output)
        output = torch.min(output, B.float()).round_().int()
    else:
        bits = bits.cuda()
        output, scale = ext_quantization.pack_mixed_precision(output, mn, mx, bits)

    return output, raw_shape, bits, scale

def dequantize_mixed_precision(data, shape, bits, scale, mn):
    if SIMULATE:
        data = data.float() / scale + mn
    else:
        data = ext_quantization.unpack_mixed_precision(data, bits,
                scale, mn, shape[0], np.prod(shape) // shape[0])
    return data.view(*shape)

class act_quantized_conv2d(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, name=None):
        with torch.no_grad():
            q_bits, q_min, mx = compute_quantization_bits(input, name)
            q_input, q_input_shape, q_bits, q_scale =\
                quantize_mixed_precision(input, q_bits, q_min, mx, False)
            output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        ctx.name = name
        ctx.save_for_backward(q_input, q_bits, q_scale, q_min, weight, bias)
        ctx.other_args = (q_input_shape, stride, padding, dilation, groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q_input, q_bits, q_scale, q_min, weight, bias = ctx.saved_tensors
        q_input_shape, stride, padding, dilation, groups = ctx.other_args
        padding = _pair(padding)
        stride = _pair(stride)
        dilation = _pair(dilation)

        input = dequantize_mixed_precision(q_input, q_input_shape, q_bits, q_scale, q_min)

        grad_input, grad_weight = ext_backward_func.cudnn_convolution_backward(
                input, grad_output, weight, padding, stride, dilation, groups,
                False, False, [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3])
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


def test_conv2d_correctness():
    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 4, 28, 28, 256, 256, 3, 1, 1, 1, 1
    data_np = np.random.uniform(size=(N, CI, H, W)).astype('float32')
    weight_np = np.random.uniform(size=(CO, CI // groups, kernel_size, kernel_size)).astype('float32')
    bias_np = np.random.uniform(size=(CO,)).astype('float32')

    for device in ['cuda']:
        def test_implementation(func):
            data = torch.tensor(data_np).to(torch.device(device)).requires_grad_()
            weight = torch.tensor(weight_np).to(torch.device(device)).requires_grad_()
            bias = torch.tensor(bias_np).to(torch.device(device)).requires_grad_()

            output = func(data, weight, bias, stride, padding, dilation, groups)
            output.backward(torch.ones_like(output))

            return [x.detach().cpu().numpy() for x in [output, data.grad, weight.grad, bias.grad]]

        output_ref, grad_data_ref, grad_weight_ref, grad_bias_ref = test_implementation(F.conv2d)
        output_us, grad_data_us, grad_weight_us, grad_bias_us = test_implementation(act_quantized_conv2d.apply)

        atol = 2
        rtol = 0.20
        print("========== Conv2d Correctness Test ==========")
        np.testing.assert_allclose(output_ref, output_us, atol=atol, rtol=rtol)
        np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol, rtol=rtol)
        np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol, rtol=rtol)
        np.testing.assert_allclose(grad_bias_ref, grad_bias_us, atol=atol, rtol=rtol)


def test_conv2d_speed():
    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 128, 28, 28, 256, 256, 3, 1, 1, 1, 1
    #N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 128, 28, 28, 256, 256, 1, 1, 0, 1, 1
    data_np = np.random.uniform(size=(N, CI, H, W)).astype('float32')
    weight_np = np.random.uniform(size=(CO, CI // groups, kernel_size, kernel_size)).astype('float32')
    bias_np = np.random.uniform(size=(CO,)).astype('float32')

    for device in ['cuda']:
        def test_implementation(func, stride, padding, dilation, groups):
            data = torch.tensor(data_np).to(torch.device(device)).requires_grad_()
            weight = torch.tensor(weight_np).to(torch.device(device)).requires_grad_()
            bias = torch.tensor(bias_np).to(torch.device(device)).requires_grad_()

            stmt = "func(data, weight, bias, stride, padding, dilation, groups)"
            t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()" if device == "cuda" else "pass",
                                     finish="torch.cuda.synchronize()" if device == "cuda" else "pass")

            output = func(data, weight, bias, stride, padding, dilation, groups)
            stmt = "output.backward(torch.ones_like(output), retain_graph=True)"
            t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                      setup="torch.cuda.synchronize()" if device == "cuda" else "pass",
                                      finish="torch.cuda.synchronize()" if device == "cuda" else "pass")

            return t_forward, t_backward

        forward_ref, backward_ref = test_implementation(F.conv2d, stride, padding, dilation, groups)
        forward_us, backward_us = test_implementation(act_quantized_conv2d.apply, stride, padding, dilation, groups)

        print("========== Conv2d Speed Test ==========")
        print("Reference. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
                (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
        print("Ours.      forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
                (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))


def test_conv2d_memory_analytical():
    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 256, 28, 28, 256, 256, 3, 1, 1, 1, 1
    data_np = np.random.uniform(size=(N, CI, H, W)).astype('float32')
    weight_np = np.random.uniform(size=(CO, CI // groups, kernel_size, kernel_size)).astype('float32')
    bias_np = np.random.uniform(size=(CO,)).astype('float32')

    for device in ['cuda']:
        def test_implementation(func):
            data = torch.tensor(data_np).to(torch.device(device)).requires_grad_()
            weight = torch.tensor(weight_np).to(torch.device(device)).requires_grad_()
            bias = torch.tensor(bias_np).to(torch.device(device)).requires_grad_()

            before_size = get_memory_usage(False)

            output = func(data, weight, bias, stride, padding, dilation, groups)
            output = func(output, weight, bias, stride, padding, dilation, groups)
            output = func(output, weight, bias, stride, padding, dilation, groups)
            output = output.sum()

            after_size = get_memory_usage(False)
            output_size = compute_tensor_bytes(output)

            return after_size / 1024**2, (after_size - before_size - output_size) / 1024**2

        
        total_size_ref, act_size_ref = test_implementation(F.conv2d)
        total_size_us, act_size_us = test_implementation(act_quantized_conv2d.apply)

        print("========== Conv2d Activation Memory Test (bits = %d) ==========" % (config.activation_compression_bits))
        print("Reference. Total: %7.2f MB\tAct: %7.2f MB" % (total_size_ref, act_size_ref))
        print("Ours.      Total: %7.2f MB\tAct: %7.2f MB" % (total_size_us, act_size_us))


def test_conv2d_memory_max_batch_size():
    for device in ['cuda']:
        def test_implementation(func, n_layers, batch_sizes):
            def run_batch_size(batch_size):
                N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = batch_size, 28, 28, 256, 256, 3, 1, 1, 1, 1
                data_np = np.random.uniform(size=(N, CI, H, W)).astype('float32')
                weight_np = np.random.uniform(size=(CO, CI // groups, kernel_size, kernel_size)).astype('float32')
                bias_np = np.random.uniform(size=(CO,)).astype('float32')
    
                # allocate input and weights
                data = torch.tensor(data_np).to(torch.device(device)).requires_grad_(False)
                weights = []
                for i in range(n_layers):
                    weight = torch.tensor(weight_np).to(torch.device(device)).requires_grad_()
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
        test_implementation(act_quantized_conv2d.apply, n_layers=50, batch_sizes=[100, 200, 500, 1000, 2200, 2300, 2400, 3000, 4000])

if __name__ == "__main__":
    config.activation_compression_bits = 8
    test_conv2d_correctness()

    config.activation_compression_bits = 2
    test_conv2d_speed()

    config.activation_compression_bits = 2
    test_conv2d_memory_analytical()

    config.activation_compression_bits = 2
    test_conv2d_memory_max_batch_size()

