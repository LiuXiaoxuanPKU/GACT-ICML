"""Examples of Python implemtations that call c++ backward functions"""

import math

import numpy as np
import torch
from torch import nn, autograd
from torch.nn import init, functional as F
from torch.nn.modules.utils import _pair
from torch.utils.cpp_extension import load

import pytorch_minimax
from quantize.conf import config
from quantize.ops import QF
from C import calc_precision_dp, calc_precision, calc_avg_bits

# from timeit_v2 import py_benchmark

ext_backward_func = load(name="ext_backward_func", sources=["ext_backward_func.cc"], verbose=True)
ext_quantization = load(name="ext_quantization",
        sources=["ext_quantization.cc", "ext_quantization_cuda_kernel.cu"], verbose=True)


def compute_quantization_bits(input, name):
    N = input.shape[0]
    D = input.shape[1]
    return torch.ones(N, dtype=torch.int32) * config.initial_bits

    # input_flatten = input.view(N, -1)
    #
    # # greedy
    # grad_sum = torch.tensor(5e-5 * np.random.randn(N).astype('float32'))  # QF.get_scale(name).cpu()
    # mn = pytorch_minimax.min(input_flatten).cpu()
    # mx = pytorch_minimax.max(input_flatten).cpu()
    # Range = mx - mn
    # C = D / 4 * Range ** 2 * grad_sum
    # b = torch.ones(N, dtype=torch.int32) * config.initial_bits
    # b = calc_precision(b, C, config.activation_compression_bits * N)

    # return b

SIMULATE = True
def quantize_mixed_precision(data, bits, stochastic=True):
    assert stochastic
    raw_shape = data.shape
    output = data.view(raw_shape[0], -1)

    if SIMULATE:
        bits = bits.cuda()
        B = (2 ** bits - 1).unsqueeze(1)
        mn = pytorch_minimax.min(output).unsqueeze(1) - 1e-8
        mx = pytorch_minimax.max(output).unsqueeze(1) + 1e-8
        scale = B / (mx - mn)
        output = (output - mn) * scale

        if stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)

        output = F.relu(output)
        output = torch.min(output, B.float()).round_().int()
    else:
        bits = bits.cuda()
        mn = pytorch_minimax.min(output)
        mx = pytorch_minimax.max(output)

        output, scale = ext_quantization.pack_mixed_precision(output, mn, mx, bits)

    return output, raw_shape, bits, scale, mn

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
            q_bits = compute_quantization_bits(input, name)
            if not QF.training or config.activation_compression_bits >= 32:
                q_input, q_input_shape, q_bits, q_scale, q_min = input, None, None, None, None
            else:
                q_input, q_input_shape, q_bits, q_scale, q_min = quantize_mixed_precision(input, q_bits, True)

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

        if not QF.training or config.activation_compression_bits >= 32:
            input = q_input
        else:
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
    data_np = np.random.randn(N, CI, H, W).astype('float32')
    weight_np = np.random.randn(CO, CI // groups, kernel_size, kernel_size).astype('float32')
    bias_np = np.random.rand(CO).astype('float32')

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
    data_np = np.random.randn(N, CI, H, W).astype('float32')
    weight_np = np.random.randn(CO, CI // groups, kernel_size, kernel_size).astype('float32')
    bias_np = np.random.rand(CO).astype('float32')

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

        forward_us, backward_us = test_implementation(act_quantized_conv2d.apply, stride, padding, dilation, groups)
        forward_ref, backward_ref = test_implementation(F.conv2d, stride, padding, dilation, groups)

        print("========== Conv2d Speed Test ==========")
        print("Reference. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
                (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
        print("Ours.      forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
                (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))


if __name__ == "__main__":
    config.activation_compression_bits = 8
    test_conv2d_correctness()

    config.activation_compression_bits = 2
    test_conv2d_speed()

