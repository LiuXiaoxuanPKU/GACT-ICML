"""Examples of Python implemtations that call c++ backward functions"""

import math

import numpy as np
import torch
from torch import nn, autograd
from torch.nn import init, functional as F
from torch.nn.modules.utils import _pair
from torch.utils.cpp_extension import load

from timeit_v2 import py_benchmark

ext_backward_func = load(name="ext_backward_func", sources=["ext_backward_func.cc"], verbose=True)

class conv2d_explicit_backward(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, bias)
        ctx.other_args = (stride, padding, dilation, groups)
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.other_args
        padding = _pair(padding)
        stride = _pair(stride)
        dilation = _pair(dilation)

        grad_input, grad_weight = ext_backward_func.cudnn_convolution_backward(
                input, grad_output, weight, padding, stride, dilation, groups,
                False, False, [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3])
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None


class layer_norm_explicit_backward(autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight=None, bias=None, eps=1e-5):
        X, gamma, beta, M, N = ext_backward_func.prepare_layer_norm_inputs(input, normalized_shape, weight, bias)
        Y, mean, rstd = ext_backward_func.layer_norm_cuda(X, gamma, beta, M, N, eps)
        ctx.save_for_backward(X, gamma, mean, rstd)
        ctx.other_args = (M, N, eps)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, gamma, mean, rstd = ctx.saved_tensors
        M, N, eps = ctx.other_args

        grad_input, grad_weight, grad_bias = ext_backward_func.layer_norm_backward_cuda(
                grad_output, X, mean, rstd, gamma, M, N,
                [ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]])
        return grad_input, None, grad_weight, grad_bias, None


def test_conv2d_correctness():
    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 4, 28, 28, 128, 128, 3, 1, 1, 1, 1
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
        output_us, grad_data_us, grad_weight_us, grad_bias_us = test_implementation(conv2d_explicit_backward.apply)

        atol = 1e-5
        print("========== Conv2d Correctness Test ==========")
        np.testing.assert_allclose(output_ref, output_us, atol=atol)
        np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol)
        np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol)
        np.testing.assert_allclose(grad_bias_ref, grad_bias_us, atol=atol)


def test_conv2d_speed():
    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 128, 28, 28, 256, 256, 3, 1, 1, 1, 1
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

        forward_ref, backward_ref = test_implementation(F.conv2d, stride, padding, dilation, groups)
        forward_us, backward_us = test_implementation(conv2d_explicit_backward.apply, stride, padding, dilation, groups)
        print("========== Conv2d Speed Test ==========")
        print("Reference. Forward: %.6f s\tbackward: %.6f s" % (forward_ref, backward_ref))
        print("Ours.      Forward: %.6f s\tbackward: %.6f s" % (forward_us, backward_us))

def test_layer_norm_correctness():
    input_shape = (64, 1024)
    normalized_shape = (1024,)
    data_np = np.random.randn(*input_shape)
    weight_np = np.random.randn(*normalized_shape)
    bias_np = np.random.randn(*normalized_shape)

    for device in ['cuda']:
        def test_implementation(func):
            data = torch.tensor(data_np).to(torch.device(device)).requires_grad_()
            weight = torch.tensor(weight_np).to(torch.device(device)).requires_grad_()
            bias = torch.tensor(bias_np).to(torch.device(device)).requires_grad_()

            output = func(data, normalized_shape, weight, bias)
            output.backward(torch.ones_like(output))

            return [x.detach().cpu().numpy() for x in [output, data.grad, weight.grad, bias.grad]]

        output_ref, grad_data_ref, grad_weight_ref, grad_bias_ref = test_implementation(F.layer_norm)
        output_us, grad_data_us, grad_weight_us, grad_bias_us = test_implementation(layer_norm_explicit_backward.apply)
    
        atol = 1e-5
        print("========== LayerNorm Correctness Test ==========")
        np.testing.assert_allclose(output_ref, output_us, atol=atol)
        np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol)
        np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol)
        np.testing.assert_allclose(grad_bias_ref, grad_bias_us, atol=atol)


def test_layer_norm_speed():
    input_shape = (64, 1024)
    normalized_shape = (1024,)
    data_np = np.random.randn(*input_shape)
    weight_np = np.random.randn(*normalized_shape)
    bias_np = np.random.randn(*normalized_shape)

    for device in ['cuda']:
        def test_implementation(func, normalized_shape):
            data = torch.tensor(data_np).to(torch.device(device)).requires_grad_()
            weight = torch.tensor(weight_np).to(torch.device(device)).requires_grad_()
            bias = torch.tensor(bias_np).to(torch.device(device)).requires_grad_()


            stmt = "func(data, normalized_shape, weight, bias)"
            t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()" if device == "cuda" else "pass",
                                     finish="torch.cuda.synchronize()" if device == "cuda" else "pass")

            output = func(data, normalized_shape, weight, bias)
            stmt = "output.backward(torch.ones_like(output), retain_graph=True)"
            t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                      setup="torch.cuda.synchronize()" if device == "cuda" else "pass",
                                      finish="torch.cuda.synchronize()" if device == "cuda" else "pass")

            return t_forward, t_backward

        forward_ref, backward_ref = test_implementation(F.layer_norm, normalized_shape)
        forward_us, backward_us = test_implementation(layer_norm_explicit_backward.apply, normalized_shape)
        print("========== LayerNorm Speed Test ==========")
        print("Reference. Forward: %.6f s\tbackward: %.6f s" % (forward_ref, backward_ref))
        print("Ours.      Forward: %.6f s\tbackward: %.6f s" % (forward_us, backward_us))


if __name__ == "__main__":
    test_conv2d_correctness()
    test_layer_norm_correctness()

    test_conv2d_speed()
    test_layer_norm_speed()

