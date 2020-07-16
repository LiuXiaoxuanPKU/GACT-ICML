import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize import config
from quantize.ops import *


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

    num_layers = 0

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.quantize_input = QuantMeasure()
        self.name = 'conv_{}'.format(QConv2d.num_layers)
        QConv2d.num_layers += 1

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

        # self.qweight = qweight

        # self.iact = qinput

        if hasattr(self, 'exact') or not config.biprecision:
            if config.compress_activation:
                output = QF.conv2d(qinput, qweight, qbias, self.stride,
                              self.padding, self.dilation, self.groups, self.name)
            else:
                output = F.conv2d(qinput, qweight, qbias, self.stride,
                              self.padding, self.dilation, self.groups)
        else:
            output = conv2d_biprec(qinput, qweight, qbias, self.stride,
                                   self.padding, self.dilation, self.groups)
        # self.act = output

        return output


# class QLinear(nn.Linear): # The linear layer for quantized training
#     """docstring for QConv2d."""
#
#     def __init__(self, in_features, out_features, bias=True,):
#         super(QLinear, self).__init__(in_features, out_features, bias)
#         self.quantize_input = QuantMeasure()
#
#     def forward(self, input):
#         if config.quantize_activation:
#             qinput = self.quantize_input(input)
#         else:
#             qinput = input
#
#         if config.quantize_weights:
#             qweight = quantize(self.weight, config.weight_preconditioner())
#             if self.bias is not None:
#                 qbias = quantize(self.bias, config.bias_preconditioner())
#             else:
#                 qbias = None
#         else:
#             qweight = self.weight
#             qbias = self.bias
#
#         if hasattr(self, 'exact'):
#             output = F.linear(qinput, qweight, qbias)
#         else:
#             output = linear_biprec(qinput, qweight, qbias)
#
#         return output


class QLinear(nn.Module):
    num_layers = 0

    def __init__(self, input_features, output_features, bias=True):
        super(QLinear, self).__init__()
        self.name = 'linear_{}'.format(QLinear.num_layers)
        QLinear.num_layers += 1

        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return MyLinear_apply.apply(input, self.weight, self.bias, self.name)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'name={}, input_features={}, output_features={}, bias={}'.format(
            self.name, self.input_features, self.output_features, self.bias is not None
        )


class QBatchNorm2D(nn.BatchNorm2d):
    num_layers = 0

    def __init__(self, num_features):
        super(QBatchNorm2D, self).__init__(num_features)
        self.quantize_input = QuantMeasure()
        self.name = 'bn_{}'.format(QBatchNorm2D.num_layers)
        QBatchNorm2D.num_layers += 1

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

        return batch_norm().apply(
            input, self.running_mean, self.running_var, qweight, qbias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps, self.name)


