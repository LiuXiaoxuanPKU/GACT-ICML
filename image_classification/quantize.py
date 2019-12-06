from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
import time

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


quan_grad = True
record_error = False
error_signals = []
stdevs = []
running_time_stats = 0.0


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,  reduce_type='mean', keepdim=False, true_zero=False):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)

        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                # min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                # max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]
                pass

        # TODO: re-add true zero computation
        range_values = max_values - min_values
        end.record()
        torch.cuda.synchronize()

        global running_time_stats
        running_time_stats += start.elapsed_time(end) / 1000

        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        zero_point = qparams.zero_point
        num_bits = qparams.num_bits
        qmin = -(2.**(num_bits - 1)) if signed else 0.
        qmax = qmin + 2.**num_bits - 1.
        scale = qparams.range / (qmax - qmin)
        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(qmin, qmax).round_()

            if dequantize:
                output.mul_(scale).add_(
                    zero_point - qmin * scale)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None


class UniformQuantizeGrad(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD,
                reduce_dim=0, dequantize=True, signed=False, stochastic=True):
        ctx.num_bits = num_bits
        ctx.qparams = qparams
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.signed = signed
        ctx.dequantize = dequantize
        ctx.reduce_dim = reduce_dim
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):
        qparams = ctx.qparams
        with torch.no_grad():
            if qparams is None:
                assert ctx.num_bits is not None, "either provide qparams of num_bits to quantize"
                qparams = calculate_qparams(
                    grad_output, num_bits=ctx.num_bits, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim, reduce_type='extreme')

            grad_input = quantize(grad_output, num_bits=None,
                                  qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                                  dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
            if record_error:
                error_signals.append(grad_input.detach().cpu().numpy())

                # grad_input_2 = quantize(grad_output, num_bits=None,
                #                       qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                #                       dequantize=False, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
                # error_signals.append(grad_input_2)
                #
                # grads = []
                # with torch.no_grad():
                #     for iter in range(100):
                #         grads.append(
                #             quantize(grad_output, num_bits=None,
                #                      qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                #                      dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False))
                #     grads = torch.stack(grads, 0)
                #     stdevs.append(grads.std(0).detach().cpu().numpy())

        return grad_input, None, None, None, None, None, None, None


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    if quan_grad:
        out1 = F.conv2d(input.detach(), weight, bias,
                        stride, padding, dilation, groups)
        out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None,
                        stride, padding, dilation, groups)
        out2 = quantize_grad(out2, num_bits=num_bits_grad, flatten_dims=(1, -1))
        return out1 + out2 - out1.detach()
    else:
        out = F.conv2d(input, weight, bias,
                        stride, padding, dilation, groups)
        return out


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    if quan_grad:
        out1 = F.linear(input.detach(), weight, bias)
        out2 = F.linear(input, weight.detach(), bias.detach()
                        if bias is not None else None)
        out2 = quantize_grad(out2, num_bits=num_bits_grad)
        return out1 + out2 - out1.detach()
    else:
        out = F.linear(input, weight, bias)
        return out


def quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace)


def quantize_grad(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True, signed=False, stochastic=True):
    return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic)


# class QuantMeasure(nn.Module):
#     """docstring for QuantMeasure."""
#
#     def __init__(self, num_bits=8, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
#                  inplace=False, dequantize=True, stochastic=False, momentum=0.1, measure=False):
#         super(QuantMeasure, self).__init__()
#         self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
#         self.register_buffer('running_range', torch.zeros(*shape_measure))
#         self.measure = measure
#         if self.measure:
#             self.register_buffer('num_measured', torch.zeros(1))
#         self.flatten_dims = flatten_dims
#         self.momentum = momentum
#         self.dequantize = dequantize
#         self.stochastic = stochastic
#         self.inplace = inplace
#         self.num_bits = num_bits
#
#     def forward(self, input, qparams=None):
#
#         if self.training or self.measure:
#             if qparams is None:
#                 qparams = calculate_qparams(
#                     input, num_bits=self.num_bits, flatten_dims=self.flatten_dims, reduce_dim=0)
#             with torch.no_grad():
#                 if self.measure:
#                     momentum = self.num_measured / (self.num_measured + 1)
#                     self.num_measured += 1
#                 else:
#                     momentum = self.momentum
#                 self.running_zero_point.mul_(momentum).add_(
#                     qparams.zero_point * (1 - momentum))
#                 self.running_range.mul_(momentum).add_(
#                     qparams.range * (1 - momentum))
#         else:
#             qparams = QParams(range=self.running_range,
#                               zero_point=self.running_zero_point, num_bits=self.num_bits)
#         if self.measure:
#             return input
#         else:
#             q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
#                                stochastic=self.stochastic, inplace=self.inplace)
#             return q_input


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 inplace=False, dequantize=True, stochastic=False, momentum=0.1, measure=False):
        super(QuantMeasure, self).__init__()
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace
        self.num_bits = num_bits

    def forward(self, input, qparams=None):
        if qparams is None:
            qparams = calculate_qparams(
                input, num_bits=self.num_bits, flatten_dims=self.flatten_dims, reduce_dim=0)

        q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
                           stochastic=self.stochastic, inplace=self.inplace)
        return q_input

class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=8, biprecision=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(
            self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
        self.biprecision = biprecision

    def forward(self, input):
        qinput = self.quantize_input(input)
        weight_qparams = calculate_qparams(
            self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams)
        self.qweight = qweight

        if self.bias is not None:
            qbias = quantize(
                self.bias, num_bits=self.num_bits_weight + self.num_bits,
                flatten_dims=(0, -1))
        else:
            qbias = None
        if not self.biprecision or self.num_bits_grad is None:
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                              self.padding, self.dilation, self.groups)
            if self.num_bits_grad is not None:
                output = quantize_grad(
                    output, num_bits=self.num_bits_grad, flatten_dims=(1, -1))
        else:
            output = conv2d_biprec(qinput, qweight, qbias, self.stride,
                                   self.padding, self.dilation, self.groups, num_bits_grad=self.num_bits_grad)
        return output


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=8, biprecision=True):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.quantize_input = QuantMeasure(self.num_bits)

    def forward(self, input):
        qinput = self.quantize_input(input)
        weight_qparams = calculate_qparams(
            self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams)
        if self.bias is not None:
            qbias = quantize(
                self.bias, num_bits=self.num_bits_weight + self.num_bits,
                flatten_dims=(0, -1))
        else:
            qbias = None

        if not self.biprecision or self.num_bits_grad is None:
            output = F.linear(qinput, qweight, qbias)
            if self.num_bits_grad is not None:
                output = quantize_grad(
                    output, num_bits=self.num_bits_grad)
        else:
            output = linear_biprec(qinput, qweight, qbias, self.num_bits_grad)
        return output


if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
