from collections import namedtuple
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.utils.cpp_extension import load
from torch.autograd.function import Function
from quantize.conf import config


QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])


ext_backward_func = load(name="ext_backward_func", sources=["ext_backward_func.cc"], verbose=True)
ext_quantization = load(name="ext_quantization",
        sources=["ext_quantization.cc", "ext_quantization_cuda_kernel.cu"], verbose=True)


def quantize_mixed_precision(data, bits, mn, mx, stochastic=True):
    assert stochastic
    if not config.compress_activation:
        return data, None, None, None

    raw_shape = data.shape
    output = data.view(raw_shape[0], -1)

    if config.simulate:
        bits = bits.cuda()
        B = (2 ** bits - 1).unsqueeze(1)
        mn = mn - 1e-8
        mx = mx + 1e-8
        scale = B / (mx - mn)
        output = (output - mn) * scale

        if stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)

        output = F.relu(output)
        output = torch.min(output, B.float()).round_().int()
    else:
        bits = bits.cuda()
        output, scale = ext_quantization.pack_mixed_precision(output, mn, mx, bits)

    return output, raw_shape, bits, scale


def dequantize_mixed_precision(data, shape, bits, scale, mn):
    if not config.compress_activation:
        return data

    if config.simulate:
        data = data.float() / scale + mn
    else:
        assert config.quantize_activation
        data = ext_quantization.unpack_mixed_precision(data, bits,
                scale, mn, shape[0], np.prod(shape) // shape[0])
    return data.view(*shape)


class conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, scheme=None):
        if not config.training:
            return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        with torch.no_grad():
            q_bits, q_min, mx = scheme.compute_quantization_bits(input)
            q_input, q_input_shape, q_bits, q_scale =\
                quantize_mixed_precision(input, q_bits, q_min, mx, True)
            output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        ctx.scheme = scheme
        if config.swap:
            ctx.save_for_backward(*[x.cpu() if x is not None else None for x in [q_input, q_bits, q_scale, q_min, weight, bias]])
        else:
            ctx.save_for_backward(q_input, q_bits, q_scale, q_min, weight, bias)
        ctx.other_args = (q_input_shape, stride, padding, dilation, groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ctx.scheme.set_scale(grad_output)

        if config.swap:
            q_input, q_bits, q_scale, q_min, weight, bias = [x.cuda() if x is not None else x for x in ctx.saved_tensors]
        else:
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

        # TODO debug
        # print('Saving')
        # torch.save([input, weight, grad_output, grad_weight], ctx.scheme.name + '.pt')

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, scheme=None):
        if not config.training:
            output = input.mm(weight.t())
            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)
            return output

        with torch.no_grad():
            q_bits, q_min, mx = scheme.compute_quantization_bits(input)
            q_input, q_input_shape, q_bits, q_scale = \
                quantize_mixed_precision(input, q_bits, q_min, mx, True)
            # q_bits, q_min, mx = None, None, None
            # q_input, q_input_shape, q_bits, q_scale = \
            #     input, None, None, None

            # TODO: the following implementation might not be optimal
            output = input.mm(weight.t())
            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)

        ctx.scheme = scheme
        if config.swap:
            ctx.save_for_backward(*[x.cpu() if x is not None else None for x in [q_input, q_bits, q_scale, q_min, weight, bias]])
        else:
            ctx.save_for_backward(q_input, q_bits, q_scale, q_min, weight, bias)
        ctx.other_args = q_input_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ctx.scheme.set_scale(grad_output)

        if config.swap:
            q_input, q_bits, q_scale, q_min, weight, bias = [x.cuda() if x is not None else x for x in ctx.saved_tensors]
        else:
            q_input, q_bits, q_scale, q_min, weight, bias = ctx.saved_tensors
        q_input_shape = ctx.other_args
        input = dequantize_mixed_precision(q_input, q_input_shape, q_bits, q_scale, q_min)

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
        if not config.training:
            return F.batch_norm(
                input, running_mean, running_var, weight, bias,
                training, exponential_average_factor, eps)

        with torch.no_grad():
            # TODO: fused batch_norm
            output = F.batch_norm(
                input, running_mean, running_var, weight, bias,
                training, exponential_average_factor, eps)

            batch_mean = input.mean((0, 2, 3), keepdim=True)
            batch_std = torch.sqrt(input.var((0, 2, 3), keepdim=True) + eps)
            normalized = (input - batch_mean) / batch_std
            weight = weight.view(1, -1, 1, 1)

            q_bits, q_min, mx = scheme.compute_quantization_bits(normalized)
            q_input, q_input_shape, q_bits, q_scale = \
                quantize_mixed_precision(normalized, q_bits, q_min, mx, True)

        ctx.scheme = scheme
        # TODO save_for_backward is not working, get RuntimeError: No grad accumulator for a saved leaf!
        # if config.swap:
        #     ctx.save_for_backward(
        #         *[x.cpu() if x is not None else None for x in [q_input, q_bits, q_scale, q_min, weight, batch_std]])
        # else:
        #     ctx.save_for_backward(q_input, q_bits, q_scale, q_min, weight, batch_std)
        ctx.other_args = q_input_shape
        # ctx.saved = (q_input, q_bits, q_scale, q_min, weight, bias, batch_std, input, normalized)
        ctx.saved = (q_input, q_bits, q_scale, q_min, weight, batch_std, normalized, bias, input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # TODO save_for_backward is not working, get RuntimeError: No grad accumulator for a saved leaf!
        # if config.swap:
        #     q_input, q_bits, q_scale, q_min, weight, batch_std = [x.cuda() if x is not None else x for x in ctx.saved_tensors]
        # else:
        #     q_input, q_bits, q_scale, q_min, weight, batch_std = ctx.saved_tensors
        # q_input, q_bits, q_scale, q_min, weight, bias, batch_std, input, normalized_0 = ctx.saved
        q_input, q_bits, q_scale, q_min, weight, batch_std, normalized_0, bias, input = ctx.saved
        q_input_shape = ctx.other_args
        normalized = dequantize_mixed_precision(q_input, q_input_shape, q_bits, q_scale, q_min)
        # if ctx.scheme.name == 'bn_layer_0':
        #     normalized = dequantize_mixed_precision(q_input, q_input_shape, q_bits, q_scale, q_min)
        #     grad_weight = (grad_output * normalized).sum((0, 2, 3))
        #     grad_bias = grad_output.sum((0, 2, 3))
        #     grad_normalized = grad_output * weight
        #
        #     mean_grad_normalized = grad_normalized.mean((0, 2, 3), keepdim=True)
        #     mean_grad = (normalized * grad_normalized).mean((0, 2, 3), keepdim=True)
        #     grad_input = grad_normalized - mean_grad_normalized - normalized * mean_grad
        #     grad_input_1 = (grad_input / batch_std).clone()
        #     print((grad_normalized - mean_grad_normalized).norm(),
        #           (normalized * mean_grad).norm(), grad_input_1.norm())
        #     print(grad_input_1[0,0,0])
        #     normalized_1 = normalized.clone()

        # normalized = ctx.saved[-1]

        # TODO: fused batch_norm
        grad_weight = (grad_output * normalized).sum((0, 2, 3))
        grad_bias = grad_output.sum((0, 2, 3))
        grad_normalized = grad_output * weight

        mean_grad_normalized = grad_normalized.mean((0, 2, 3), keepdim=True)
        mean_grad = (normalized * grad_normalized).mean((0, 2, 3), keepdim=True)
        grad_input = grad_normalized - mean_grad_normalized - normalized * mean_grad
        grad_input = grad_input / batch_std

        # if ctx.scheme.name == 'bn_layer_0':
        #     print(grad_input.norm(), (grad_input - grad_input_1).norm())
        #     print(normalized.norm(), (normalized - normalized_1).norm())

        ctx.scheme.set_scale(grad_normalized, batch_std, (mean_grad**2).sum())

        # print('Saving')
        # torch.save([input, weight, bias, grad_output, grad_input], ctx.scheme.name + '.pt')

        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None


if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
