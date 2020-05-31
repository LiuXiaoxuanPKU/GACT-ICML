from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction
from image_classification.preconditioner import ScalarPreconditioner, ForwardPreconditioner, DiagonalPreconditioner, BlockwiseHouseholderPreconditioner, ScalarPreconditionerAct


class QuantizationConfig:
    def __init__(self):
        self.quantize_activation = True
        self.quantize_weights = True
        self.quantize_gradient = True
        self.activation_num_bits = 8
        self.weight_num_bits = 8
        self.bias_num_bits = 16
        self.backward_num_bits = 8
        self.bweight_num_bits = 8
        self.backward_persample = False
        self.biased = False
        self.grads = None
        self.acts = None
        self.hadamard = False
        self.biprecision = True

    def activation_preconditioner(self):
        # return lambda x: ForwardPreconditioner(x, self.activation_num_bits)
        return lambda x: ScalarPreconditionerAct(x, self.activation_num_bits)
        # return lambda x: ScalarPreconditioner(x, 16)

    def weight_preconditioner(self):
        return lambda x: ScalarPreconditioner(x, self.weight_num_bits)
        # return lambda x: ForwardPreconditioner(x, self.weight_num_bits)
        # return lambda x: DiagonalPreconditioner(x, self.weight_num_bits)

    def bias_preconditioner(self):
        return lambda x: ScalarPreconditioner(x, self.bias_num_bits)

    def activation_gradient_preconditioner(self):
        if self.hadamard:
            return lambda x: BlockwiseHouseholderPreconditioner(x, self.backward_num_bits)
        if self.backward_persample:
            return lambda x: DiagonalPreconditioner(x, self.backward_num_bits)
        else:
            return lambda x: ScalarPreconditioner(x, self.backward_num_bits)

    def weight_gradient_preconditioner(self):
        if self.backward_persample:
            return lambda x: DiagonalPreconditioner(x, self.bweight_num_bits, left=False)
        else:
            return lambda x: ScalarPreconditioner(x, self.bweight_num_bits)


config = QuantizationConfig()

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, Preconditioner, stochastic=False, inplace=False):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print('---')
        #     print(input.view(-1)[:10], input.min(), input.max())
        with torch.no_grad():
            preconditioner = Preconditioner(output)
            output = preconditioner.forward()

            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(0.0, preconditioner.num_bins).round_()

            output = preconditioner.inverse(output)

        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print(output.view(-1)[:10])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None


class UniformQuantizeGrad(InplaceFunction):
    @staticmethod
    def forward(ctx, input, Preconditioner, stochastic=True):
        ctx.stochastic = stochastic
        ctx.inplace = False
        ctx.Preconditioner = Preconditioner
        return input

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            if config.grads is not None:
                config.grads.append(grad_output.detach())

            grad_input = quantize(grad_output, ctx.Preconditioner, stochastic=ctx.stochastic, inplace=False)

        return grad_input, None, None


def quantize(x, Preconditioner, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, Preconditioner, stochastic, inplace)


def quantize_grad(x, Preconditoner, stochastic=True):
    return UniformQuantizeGrad().apply(x, Preconditoner, stochastic)


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if config.quantize_gradient:
        out1 = F.conv2d(input.detach(), weight, bias,
                        stride, padding, dilation, groups)
        out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None,
                        stride, padding, dilation, groups)
        out1 = quantize_grad(out1, config.weight_gradient_preconditioner())
        out2 = quantize_grad(out2, config.activation_gradient_preconditioner())
        return out1 + out2 - out1.detach()
    else:
        out = F.conv2d(input, weight, bias,
                        stride, padding, dilation, groups)
        return out


def linear_biprec(input, weight, bias=None):
    if config.quantize_gradient:
        out1 = F.linear(input.detach(), weight, bias)
        out2 = F.linear(input, weight.detach(), bias.detach()
                        if bias is not None else None)
        out1 = quantize_grad(out1, config.weight_gradient_preconditioner())
        out2 = quantize_grad(out2, config.activation_gradient_preconditioner())
        return out1 + out2 - out1.detach()
    else:
        out = F.linear(input, weight, bias)
        return out


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, inplace=False, stochastic=False):
        super(QuantMeasure, self).__init__()
        self.stochastic = stochastic
        self.inplace = inplace

    def forward(self, input):
        q_input = quantize(input, config.activation_preconditioner(),
                           stochastic=self.stochastic, inplace=self.inplace)
        return q_input


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.quantize_input = QuantMeasure()

    def forward(self, input):
        if config.acts is not None:
            config.acts.append(input.detach().cpu().numpy())

        if config.quantize_activation:
            qinput = self.quantize_input(input)
        else:
            qinput = input

        if config.quantize_weights:     # TODO weight quantization scheme...
            qweight = quantize(self.weight, config.weight_preconditioner())
            if self.bias is not None:
                qbias = quantize(self.bias, config.bias_preconditioner())
            else:
                qbias = None
            qbias = self.bias
        else:
            qweight = self.weight
            qbias = self.bias

        self.qweight = qweight

        self.iact = qinput

        if hasattr(self, 'exact'):
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                              self.padding, self.dilation, self.groups)
        else:
            output = conv2d_biprec(qinput, qweight, qbias, self.stride,
                                   self.padding, self.dilation, self.groups)
        self.act = output

        return output


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True,):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.quantize_input = QuantMeasure()

    def forward(self, input):
        if config.quantize_activation:
            qinput = self.quantize_input(input)
        else:
            qinput = input

        if config.quantize_weights:
            qweight = quantize(self.weight, config.weight_preconditioner())
            if self.bias is not None:
                qbias = quantize(self.bias, config.bias_preconditioner())
            else:
                qbias = None
        else:
            qweight = self.weight
            qbias = self.bias

        if hasattr(self, 'exact'):
            output = F.linear(qinput, qweight, qbias)
        else:
            output = linear_biprec(qinput, qweight, qbias)

        return output


class QBatchNorm2D(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(QBatchNorm2D, self).__init__(num_features)
        self.quantize_input = QuantMeasure()

    def forward(self, input):       # TODO: weight is not quantized
        self._check_input_dim(input)
        if config.quantize_activation:
            qinput = self.quantize_input(input)
        else:
            qinput = input

        # if config.quantize_weights:
        #     qweight = quantize(self.weight, config.bias_preconditioner())
        #     qbias = quantize(self.bias, config.bias_preconditioner())
        # else:
        #     qweight = self.weight
        #     qbias = self.bias

        qweight = self.weight
        qbias = self.bias

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, qweight, qbias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
