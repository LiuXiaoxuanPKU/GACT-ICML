from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
import time
import math
import numpy as np
from image_classification.preconditioner import get_transform

import pytorch_minimax

def hadamard(order):
    if order == 0:
        return np.array([1.0])

    result = np.zeros([2**order, 2**order])
    n = 2**(order - 1)
    sub_mat = hadamard(order - 1)
    result[:n, :n] = sub_mat
    result[:n, n:] = sub_mat
    result[n:, :n] = sub_mat
    result[n:, n:] = -sub_mat
    result /= np.sqrt(2.0)
    return result


class QuantizationConfig:
    def __init__(self):
        self.quantize_activation = True
        self.quantize_weights = True
        self.quantize_gradient = True
        self.forward_num_bits = 8
        self.backward_num_bits = 8
        self.backward_persample = False
        self.biased = False
        self.grads = None
        self.acts = None
        self.hadamard = False
        self.biprecision = True


config = QuantizationConfig()

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,  reduce_type='mean', keepdim=False, true_zero=False):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        if reduce_type == 'extreme':
            if config.backward_persample:
                if len(x.shape) == 2:
                    min_values = x.min(-1, keepdim=True)[0]
                    max_values = x.max(-1, keepdim=True)[0]
                else:
                    x_flat = x.flatten(start_dim=1)
                    '''
                    min_values = _deflatten_as(x_flat.min(-1, keepdim=True)[0], x) - 1e-8
                    max_values = _deflatten_as(x_flat.max(-1, keepdim=True)[0], x) + 1e-8
                    '''
                    min_values = _deflatten_as(pytorch_minimax.min(x_flat).unsqueeze(1), x) - 1e-8
                    max_values = _deflatten_as(pytorch_minimax.max(x_flat).unsqueeze(1), x) + 1e-8
            else:
                min_values = x.min()
                max_values = x.max()
        else:
            x_flat = x.flatten(*flatten_dims)
            if x_flat.dim() == 1:
                min_values = _deflatten_as(x_flat.min(), x)
                max_values = _deflatten_as(x_flat.max(), x)
            else:
                # min_values = _deflatten_as(x_flat.min(-1)[0], x)
                # max_values = _deflatten_as(x_flat.max(-1)[0], x)
                min_values = _deflatten_as(pytorch_minimax.min(x_flat).unsqueeze(1), x) - 1e-8
                max_values = _deflatten_as(pytorch_minimax.max(x_flat).unsqueeze(1), x) + 1e-8

            if reduce_dim is not None:
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)

        # if reduce_dim is not None:
        #     if reduce_type == 'mean':
        #         min_values = min_values.mean(reduce_dim, keepdim=keepdim)
        #         max_values = max_values.mean(reduce_dim, keepdim=keepdim)
        #     else:
        #         if config.backward_persample:
        #             pass
        #         else:
        #             min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
        #             max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]

        # TODO: re-add true zero computation
        range_values = max_values - min_values
        end.record()
        torch.cuda.synchronize()

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

        cmin = -(2. ** (num_bits - 1)) if signed else 0.
        cmax = cmin + 2. ** num_bits - 1.

        # For biased quantization
        scale_factor = 0.2
        delta = round(2. ** num_bits * scale_factor)
        qmin = cmin
        qmax = cmax
        if config.biased:
            qmin -= delta
            qmax += delta

        scale = qparams.range / (qmax - qmin)
        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(cmin, cmax).round_()

            if dequantize:
                output.mul_(scale).add_(
                    zero_point - qmin * scale)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None


class Hadamard:
    Hs = {}

    @staticmethod
    def get_hadamard(n, inverse):
        # if not (n in Hadamard.Hs):
        #     order = math.floor(math.log2(n))
        #     if 2 ** order != n:
        #         raise RuntimeError("Batch size is not power of two " + str(n))
        #
        #     H = hadamard(order)
        #     Hadamard.Hs[n] = torch.tensor(H, dtype=torch.float32).cuda()
        #
        # return Hadamard.Hs[n]
        if not (n in Hadamard.Hs):
            print('Generating transformation', n)
            w = torch.randn(n, n).cuda()
            q, _ = torch.qr(w)
            Hadamard.Hs[n] = q

        if inverse:
            return Hadamard.Hs[n].transpose(0, 1)
        else:
            return Hadamard.Hs[n]

    @staticmethod
    def sample_hadamard(x, inverse):
        n = x.shape[0]
        x_shape = x.shape
        H = Hadamard.get_hadamard(n, inverse)
        x = H @ x.reshape(n, -1)
        return x.reshape(*x_shape)

    @staticmethod
    def sample_channel_hadamard(x, inverse):
        n, c, w, _ = x.shape
        x = x.view(n * c, w * w)
        H = Hadamard.get_hadamard(n * c, inverse)
        x = H @ x
        x = x.view(n, c, w, w)
        return x

    @staticmethod
    def channel_hadamard(x, inverse):
        n, c, w, _ = x.shape
        x = x.transpose(1, 3)
        x = x.reshape(n * w * w, c)
        H = Hadamard.get_hadamard(c, inverse)
        x = x @ H
        x = x.reshape(n, w, w, c)
        x = x.transpose(1, 3)
        return x

    @staticmethod
    def width_hadamard(x, inverse):
        n, c, w, _ = x.shape
        x = x.reshape(n * c * w, w)
        H = Hadamard.get_hadamard(w, inverse)
        x = x @ H
        x = x.reshape(n, c, w, w)
        return x

    @staticmethod
    def height_hadamard(x, inverse):
        n, c, w, _ = x.shape
        x = x.transpose(2, 3)
        x = x.reshape(n * c * w, w)
        H = Hadamard.get_hadamard(w, inverse)
        x = x @ H
        x = x.reshape(n, c, w, w)
        x = x.transpose(2, 3)
        return x

    @staticmethod
    def apply_hadamard(x):
        if not config.hadamard:
            return x
        if len(x.shape) == 2:
            return Hadamard.sample_hadamard(x, False)

        x = Hadamard.height_hadamard(x, False)
        x = Hadamard.width_hadamard(x, False)
        x = Hadamard.channel_hadamard(x, False)
        # x = Hadamard.sample_hadamard(x, False)
        return x

    @staticmethod
    def apply_inverse_hadamard(x):
        if not config.hadamard:
            return x
        if len(x.shape) == 2:
            return Hadamard.sample_hadamard(x, True)

        # x = Hadamard.sample_hadamard(x, True)
        x = Hadamard.channel_hadamard(x, True)
        x = Hadamard.width_hadamard(x, True)
        x = Hadamard.height_hadamard(x, True)
        return x


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
            # grad_output = Hadamard.apply_hadamard(grad_output)
            if config.hadamard:
                grad_shape = grad_output.shape
                N = grad_shape[0]
                T = get_transform(grad_output.view(N, -1))
                grad_output = (T @ grad_output.view(N, -1)).view(*grad_shape)

            if qparams is None:
                assert ctx.num_bits is not None, "either provide qparams of num_bits to quantize"
                qparams = calculate_qparams(
                    grad_output, num_bits=ctx.num_bits, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim, reduce_type='extreme')

            grad_input = quantize(grad_output, num_bits=None,
                                  qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                                  dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)

            if config.grads is not None:
                # g = quantize(grad_output, num_bits=None,
                #               qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                #               dequantize=False, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
                # config.grads.append([g.detach().cpu().numpy(),
                #                      grad_output.min(),
                #                      grad_output.max()])
                config.grads.append([grad_output.detach().cpu().numpy(),
                                     grad_output.min(),
                                     grad_output.max()])
                # grad_output = apply_hadamard(grad_output)
                # qparams = calculate_qparams(
                #     grad_output, num_bits=ctx.num_bits, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                #     reduce_type='extreme')
                # g = quantize(grad_output, num_bits=None,
                #              qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                #              dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
                # config.grads.append([g.detach().cpu().numpy(),
                #                      grad_output.min(),
                #                      grad_output.max()])

            # grad_input = Hadamard.apply_inverse_hadamard(grad_input)
            if config.hadamard:
                grad_input = (T.inverse() @ grad_input.view(N, -1)).view(*grad_shape)

        return grad_input, None, None, None, None, None, None, None


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    if config.quantize_gradient:
        out1 = F.conv2d(input.detach(), weight, bias,
                        stride, padding, dilation, groups)
        out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None,
                        stride, padding, dilation, groups)
        out2 = quantize_grad(out2, num_bits=config.backward_num_bits, flatten_dims=(1, -1))
        return out1 + out2 - out1.detach()
    else:
        out = F.conv2d(input, weight, bias,
                        stride, padding, dilation, groups)
        return out


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    if config.quantize_gradient:
        out1 = F.linear(input.detach(), weight, bias)
        out2 = F.linear(input, weight.detach(), bias.detach()
                        if bias is not None else None)
        out2 = quantize_grad(out2, num_bits=config.backward_num_bits)
        return out1 + out2 - out1.detach()
    else:
        out = F.linear(input, weight, bias)
        return out


def quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace)


def quantize_grad(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True, signed=False, stochastic=True):
    return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 inplace=False, dequantize=True, stochastic=False, momentum=0.1, measure=False):
        super(QuantMeasure, self).__init__()
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace

    def forward(self, input, qparams=None):
        if qparams is None:
            qparams = calculate_qparams(
                input, num_bits=config.forward_num_bits, flatten_dims=self.flatten_dims, reduce_dim=0)

        q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
                           stochastic=self.stochastic, inplace=self.inplace, num_bits=config.forward_num_bits)
        return q_input


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=8):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))

    def forward(self, input):
        if config.acts is not None:
            config.acts.append(input.detach().cpu().numpy())

        if config.quantize_activation:
            qinput = self.quantize_input(input)
        else:
            qinput = input

        if config.quantize_weights:
            weight_qparams = calculate_qparams(
                self.weight, num_bits=config.forward_num_bits, flatten_dims=(1, -1), reduce_dim=None)
            qweight = quantize(self.weight, qparams=weight_qparams)
        else:
            qweight = self.weight

        if self.bias is not None:
            if config.quantize_weights:
                qbias = quantize(
                    self.bias, num_bits=config.forward_num_bits + self.num_bits,
                    flatten_dims=(0, -1))
            else:
                qbias = self.bias
        else:
            qbias = None
        if not config.biprecision or self.num_bits_grad is None:
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                              self.padding, self.dilation, self.groups)
            if config.quantize_gradient and self.num_bits_grad is not None:
                output = quantize_grad(
                    output, num_bits=config.backward_num_bits, flatten_dims=(1, -1))
        else:
            output = conv2d_biprec(qinput, qweight, qbias, self.stride,
                                   self.padding, self.dilation, self.groups, num_bits_grad=self.num_bits_grad)
        return output


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=8):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure()

    def forward(self, input):
        if config.quantize_activation:
            qinput = self.quantize_input(input)
        else:
            qinput = input

        if config.quantize_weights:
            weight_qparams = calculate_qparams(
                self.weight, num_bits=config.forward_num_bits, flatten_dims=(1, -1), reduce_dim=None)
            qweight = quantize(self.weight, qparams=weight_qparams)
        else:
            qweight = self.weight

        if self.bias is not None:
            if config.quantize_weights:
                qbias = quantize(
                    self.bias, num_bits=config.forward_num_bits + self.num_bits,
                    flatten_dims=(0, -1))
            else:
                qbias = self.bias
        else:
            qbias = None

        if not config.biprecision or self.num_bits_grad is None:
            output = F.linear(qinput, qweight, qbias)
            if config.quantize_gradient and self.num_bits_grad is not None:
                output = quantize_grad(
                    output, num_bits=config.backward_num_bits)
        else:
            output = linear_biprec(qinput, qweight, qbias, self.num_bits_grad)
        return output


class QBatchNorm2D(nn.BatchNorm2d):         # TODO buggy simulation of nonlinear operations
    def __init__(self, num_features):
        super(QBatchNorm2D, self).__init__(num_features)
        self.quantize_input = QuantMeasure()

    def forward(self, input):
        if config.quantize_activation:
            qinput = self.quantize_input(input)
        else:
            qinput = input

        output = super(QBatchNorm2D, self).forward(qinput)

        if config.quantize_gradient:
            output = quantize_grad(output, num_bits=config.backward_num_bits)

        return output


if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
