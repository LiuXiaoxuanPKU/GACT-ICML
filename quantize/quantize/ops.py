from collections import namedtuple
import torch
import pytorch_minimax
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.utils.cpp_extension import load
from torch.autograd.function import Function
from quantize.conf import config
from functools import reduce


QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])


ext_backward_func = load(name="ext_backward_func", sources=["ext_backward_func.cc"], verbose=True)
ext_quantization = load(name="ext_quantization",
        sources=["ext_quantization.cc", "ext_quantization_cuda_kernel.cu"], verbose=True)


def quantize_mixed_precision(data, bits, mn, mx, stochastic=True):
    assert stochastic
    if not config.compress_activation:
        return data, None, None

    N = data.shape[0]
    output = data   # N, groups, group_dim

    if config.simulate:
        bits = bits.cuda()
        B = (2 ** bits - 1).view(N, 1, 1)
        mn = mn - 1e-6
        mx = mx + 1e-6
        scale = B / (mx - mn)     # N, groups, 1
        # print('scale ', scale, B)
        output = (output - mn) * scale

        if stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)

        output = F.relu(output)
        output = torch.min(output, B.float()).round_().int()
    else:
        # TODO
        bits = bits.cuda()
        output, scale = ext_quantization.pack_mixed_precision(output, mn, mx, bits)

    return output, bits, scale


def dequantize_mixed_precision(data, shape, bits, scale, mn):
    if not config.compress_activation:
        return data

    if config.simulate:
        data = data / scale + mn
    else:
        assert config.quantize_activation
        data = ext_quantization.unpack_mixed_precision(data, bits,
                scale, mn, shape[0], np.prod(shape) // shape[0])
    return data


def quantize_activation(input, scheme):
    N = input.shape[0]
    input_groups, q_bits, q_min, mx = scheme.compute_quantization_bits(input)
    q_input, q_bits, q_scale = \
        quantize_mixed_precision(input_groups, q_bits, q_min, mx, True)

    if q_scale is None:
        return q_input, q_bits

    return q_input, q_bits, q_scale.to(torch.bfloat16), q_min.to(torch.bfloat16)  # TODO convert q_bits to int8


def dequantize_activation(quantized, q_input_shape):
    N = q_input_shape[0]
    if len(quantized) == 2:
        q_input, q_bits = quantized
        q_scale, q_min = None, None
    else:
        q_input, q_bits, q_scale, q_min = quantized
        q_scale = q_scale.to(torch.float32)
        q_min = q_min.to(torch.float32)

    input = dequantize_mixed_precision(q_input, q_input_shape, q_bits, q_scale, q_min)

    # Remove padding
    N = q_input_shape[0]
    num_features = reduce(lambda x, y: x*y, q_input_shape[1:])
    input = input.view(N, -1)[:, :num_features]
    input = input.view(*q_input_shape)
    return input


class conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, scheme=None):
        with torch.no_grad():
            quantized = quantize_activation(input, scheme)
            output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        ctx.scheme = scheme
        # ctx.save_for_backward(*quantized, weight, bias)
        ctx.saved = quantized, weight, bias # TODO hack
        ctx.other_args = (input.shape, stride, padding, dilation, groups)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        ctx.scheme.set_scale(grad_output)

        quantized, weight, bias = ctx.saved
        q_input_shape, stride, padding, dilation, groups = ctx.other_args

        padding = _pair(padding)
        stride = _pair(stride)
        dilation = _pair(dilation)

        input = dequantize_activation(quantized, q_input_shape)

        grad_input, grad_weight = ext_backward_func.cudnn_convolution_backward(
                input, grad_output, weight, padding, stride, dilation, groups,
                False, False, [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3])
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, scheme=None):
        with torch.no_grad():
            quantized = quantize_activation(input, scheme)

            # TODO: the following implementation might not be optimal
            output = input.mm(weight.t())
            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)

        ctx.scheme = scheme
        # ctx.save_for_backward(*quantized, weight, bias)
        ctx.saved = quantized, weight, bias  # TODO hack
        ctx.other_args = input.shape

        return output

    @staticmethod
    def backward(ctx, grad_output):
        ctx.scheme.set_scale(grad_output)

        quantized, weight, bias = ctx.saved
        q_input_shape = ctx.other_args

        input = dequantize_activation(quantized, q_input_shape)

        # TODO: the following implementation might not be optimal
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        if bias is not None:
            grad_bias = grad_output.sum(0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None


class batch_norm(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias,
                training, exponential_average_factor, eps, scheme):
        with torch.no_grad():
            # TODO: fused batch_norm
            output = F.batch_norm(
                input, running_mean, running_var, weight, bias,
                training, exponential_average_factor, eps)

            batch_mean = input.mean((0, 2, 3), keepdim=True)
            batch_std = torch.sqrt(input.var((0, 2, 3), keepdim=True) + eps)
            normalized = (input - batch_mean) / batch_std
            weight = weight.view(1, -1, 1, 1)

            quantized = quantize_activation(normalized, scheme)

        ctx.scheme = scheme
        # TODO save_for_backward is not working, get RuntimeError: No grad accumulator for a saved leaf!
        ctx.other_args = normalized.shape
        ctx.saved = (quantized, weight, batch_std, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # TODO save_for_backward is not working, get RuntimeError: No grad accumulator for a saved leaf!
        quantized, weight, batch_std, bias = ctx.saved
        q_input_shape = ctx.other_args
        normalized = dequantize_activation(quantized, q_input_shape)

        # TODO: fused batch_norm
        grad_weight = (grad_output * normalized).sum((0, 2, 3))
        grad_bias = grad_output.sum((0, 2, 3))
        grad_normalized = grad_output * weight

        mean_grad_normalized = grad_normalized.mean((0, 2, 3), keepdim=True)
        mean_grad = (normalized * grad_normalized).mean((0, 2, 3), keepdim=True)
        grad_input = grad_normalized - mean_grad_normalized - normalized * mean_grad
        grad_input = grad_input / batch_std

        ctx.scheme.set_scale(grad_normalized, batch_std, (mean_grad**2).sum())

        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None


if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
