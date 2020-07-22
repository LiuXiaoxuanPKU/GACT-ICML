import math

import numpy as np
import torch
from torch import nn, autograd
from torch.nn import init, functional as F
from torch.nn.modules.utils import _pair

from timeit_v2 import py_benchmark

class conv2d_explicit_gradient(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, bias)
        ctx.other_args = (stride, padding, dilation, groups)
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_input = nn.grad.conv2d_input(input.shape, weight, grad_output, *ctx.other_args)
    
        if ctx.needs_input_grad[1]:
            grad_weight = nn.grad.conv2d_weight(input, weight.shape, grad_output, *ctx.other_args)
    
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None

def test_correctness():
    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 4, 28, 28, 128, 128, 3, 1, 1, 1, 1
    data_np = np.random.randn(N, CI, H, W).astype('float32')
    weight_np = np.random.randn(CO, CI // groups, kernel_size, kernel_size).astype('float32')

    for device in ['cpu', 'cuda']:
        # test forward
        def test_conv2d_implementation(func):
            data = torch.tensor(data_np).to(torch.device(device)).requires_grad_()
            weight = torch.tensor(weight_np).to(torch.device(device)).requires_grad_()

            output = func(data, weight, None, stride, padding, dilation, groups)
            output.backward(torch.ones_like(output))

            return output.detach().cpu().numpy(), data.grad.detach().cpu().numpy(), weight.grad.detach().cpu().numpy()
        output_ref, grad_data_ref, grad_weight_ref = test_conv2d_implementation(F.conv2d)
        output_us, grad_data_us, grad_weight_us = test_conv2d_implementation(conv2d_explicit_gradient.apply)
    
        atol = 3e-3
        np.testing.assert_allclose(output_ref, output_us, atol=atol)
        np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=atol)
        np.testing.assert_allclose(grad_weight_ref, grad_weight_us, atol=atol)


def test_speed():
    # arguments and test data
    N, H, W, CI, CO, kernel_size, stride, padding, dilation, groups = 128, 28, 28, 128, 128, 3, 1, 1, 1, 1
    data_np = np.random.randn(N, CI, H, W).astype('float32')
    weight_np = np.random.randn(CO, CI // groups, kernel_size, kernel_size).astype('float32')

    for device in ['cuda']:
        # test forward
        def test_conv2d_implementation(func, stride, padding, dilation, groups):
            data = torch.tensor(data_np).to(torch.device(device)).requires_grad_()
            weight = torch.tensor(weight_np).to(torch.device(device)).requires_grad_()

            stmt = "func(data, weight, None, stride, padding, dilation, groups)"
            t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()" if device == "cuda" else "pass",
                                     finish="torch.cuda.synchronize()" if device == "cuda" else "pass")
            output = func(data, weight, None, stride, padding, dilation, groups)
            stmt = "func(data, weight, None, stride, padding, dilation, groups)"
            t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                      setup="torch.cuda.synchronize()" if device == "cuda" else "pass",
                                      finish="torch.cuda.synchronize()" if device == "cuda" else "pass")

            return t_forward, t_backward

        forward_ref, backward_ref = test_conv2d_implementation(F.conv2d, stride, padding, dilation, groups)
        forward_us, backward_us = test_conv2d_implementation(conv2d_explicit_gradient.apply, stride, padding, dilation, groups)
        print("Reference. Forward: %.6f s\tbackward: %.6f s" % (forward_ref, backward_ref))
        print("Ours.      Forward: %.6f s\tbackward: %.6f s" % (forward_us, backward_us))

if __name__ == "__main__":
    test_correctness()
    test_speed()

