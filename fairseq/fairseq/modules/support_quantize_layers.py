import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
import numpy as np
import numbers


class QF:
    def __init__(self):
        super().__init__()

    def set_bit(bit):
        global _quantize_bit
        _quantize_bit = bit

def clamp(input, min, max, inplace=False):
    """
    Clamp tensor input to (min, max).
    input: input tensor to be clamped
    """

    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """


    # stochasitc quantization
    
    # input.mul_(scale).sub_(zero_point)
    # noise = input.new(input.shape).uniform_(-0.5, 0.5)
    # input.add_(noise)

    if inplace:
        input.mul_(scale).sub_(zero_point)
        noise = input.new(input.shape).uniform_(-0.5, 0.5)
        input.add_(noise)
        input.round_()
        # input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point +  input.new(input.shape).uniform_(-0.5, 0.5))


def linear_dequantize(input, scale, zero_point, inplace=False):
    """
    Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
    input: integer input tensor to be mapped
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    if inplace:
        return input.add_(zero_point).div_(scale)

    return (input + zero_point) / scale



def asymmetric_linear_quantization_params(num_bits,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point=True,
                                          signed=True):
    """
    Compute the scaling factor and zeropoint with the given quantization range.
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    n = 2 ** num_bits - 1
    scale = n / torch.clamp((saturation_max - saturation_min), min=1e-8)
    zero_point = scale * saturation_min

    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2**(num_bits - 1)
    return scale, zero_point


class AsymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """
    @staticmethod
    def forward(ctx, x, k):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """

        # my guess is that for NLP, the input size is always in Length * Batch Size * tokens, and we want each token has its own quantization range.
        # make random quantization
        

        x_min, x_max = x.min(dim=-1, keepdim=True)[0], x.max(dim=-1, keepdim=True)[0]
        scale, zero_point = asymmetric_linear_quantization_params(
            k, x_min, x_max)
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        n = 2 ** (k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)

        return new_quant_x.detach(), scale, zero_point

        # quant_x = linear_dequantize(new_quant_x,
        #                             scale,
        #                             zero_point,
        #                             inplace=False)
        
        # return quant_x.detach()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# _quantize_bit = 5
# _quantize_bit = QF.get_bit()


class qlinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # print(input.size(), weight.size(), '*'*100)
        with torch.no_grad():

            output = torch.matmul(input, weight.t())
            if bias is not None:
                if len(output.size()) == 2:
                    output += bias.unsqueeze(0).expand_as(output)
                elif len(output.size()) == 3:
                    output += bias.unsqueeze(0).unsqueeze(0).expand_as(output)
                else:
                    raise Exception("Error happens in linear bias term")
            
            _input, _scale, _zero = AsymmetricQuantFunction.apply(input, _quantize_bit)

            ctx.scale = _scale 
            ctx.zero = _zero
            ctx.save_for_backward(_input.type(torch.int8), weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            input, weight, bias = ctx.saved_tensors
            input = linear_dequantize(input.type(torch.float), ctx.scale, ctx.zero)
            # Record gradient
            scale = (grad_output ** 2).sum(1)

            grad_input = grad_weight = grad_bias = None

            # grad_input = grad_output.bmm(weight)
            grad_input = torch.matmul(grad_output, weight)

            last_dim_grad = grad_output.size(-1)
            last_dim_input = input.size(-1)

            grad_output = grad_output.reshape(-1, last_dim_grad).contiguous()
            input = input.reshape(-1, last_dim_input).contiguous()

            grad_weight = grad_output.t().mm(input)
            if bias is not None:
                # if len(grad_output.size()) == 2:
                grad_bias = grad_output.sum(0)
            #     # else:
            #         # grad_bias = grad_output.sum([0, 1])
            # print(grad_bias.size(), '************************'*100)
        return grad_input, grad_weight, grad_bias# The name does not need gradient.

class QLinear(nn.Module):

    def __init__(self, input_features, output_features, bias=True):
        super(QLinear, self).__init__()

        self.name = QLinear
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
        return qlinear.apply(input, self.weight, self.bias)
        # return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'name={}, input_features={}, output_features={}, bias={}'.format(
            'QLinear', self.input_features, self.output_features, self.bias is not None
        )


#################
# TODO: This have not been implemented yet!
#################

class qlayernorm(Function):

    @staticmethod
    def forward(ctx, input, narmalized_shape, weight, bias, eps=1e-5):
        with torch.no_grad():
            mean = input.mean(-1, keepdim=True)
            var = input.var(-1, keepdim=True)
            std = var.sqrt().add_(eps)

            normalized = (input - mean) / std 

            weight = weight.view(1, 1, -1)
            bias = bias.view(1, 1, -1)

            output = weight * normalized + bias 

            ctx.std = std 
            ctx.weight = weight 
            _normalized, ctx.scale, ctx.zero = AsymmetricQuantFunction.apply(normalized, _quantize_bit)
            ctx.normalized = _normalized.type(torch.int8)

        return output 

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            std = ctx.std 
            weight = ctx.weight 
            normalized = linear_dequantize(ctx.normalized.type(torch.float), ctx.scale, ctx.zero)

            grad_weight = (grad_output * normalized).sum((0, 1))
            grad_bias = grad_output.sum((0, 1))
            grad_normalized = grad_output * weight

            mean_grad_normalized = grad_normalized.mean((-1), keepdim=True)
            grad_input = grad_normalized - mean_grad_normalized - normalized * \
                     (normalized * grad_normalized).mean((-1), keepdim=True)

            grad_input = grad_input / std 

        return grad_input, None, grad_weight, grad_bias, None


class QLayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps = 1e-5, elementwise_affine = True):
        super(QLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)


        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        return qlayernorm().apply(
            input, self.normalized_shape, self.weight, self.bias, self.eps)
        # return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class qsoftmax(Function):
    @staticmethod
    def forward(ctx, input, dim=-1):
        with torch.no_grad():
            
            ori_output = F.softmax(input, dim=dim)
            ctx.dim = dim
            
            output1, ctx.scale, ctx.zero = AsymmetricQuantFunction.apply(ori_output, _quantize_bit)
            ctx.output1 = output1.type(torch.int8)

            # output2, _, _ = AsymmetricQuantFunction.apply(ori_output, _quantize_bit)
            # ctx.output2 = output2.type(torch.int8)

        return ori_output

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            # Record gradient
            scale = (grad_output.view(grad_output.shape[0], -1) ** 2).sum(1)

            output1 = linear_dequantize(ctx.output1.type(torch.float), ctx.scale, ctx.zero)
            # output2 = linear_dequantize(ctx.output2.type(torch.float), ctx.scale, ctx.zero)

            grad = output1 * (grad_output - (output1 * grad_output).sum(ctx.dim, keepdim=True))

        return grad, None


class QSoftmax(nn.Module):

    def __init__(self):
        super(QSoftmax, self).__init__()
        self.name = QSoftmax

    def forward(self, input, dim=-1):
        return qsoftmax().apply(input, dim)
        # return F.softmax(input, dim)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'name={}'.format(
            'QSoftmax'
        )


class qbmm(Function):
    @staticmethod
    def forward(ctx, inputA, inputB, transpose=True):
        with torch.no_grad():

            if transpose:
                output = torch.bmm( inputA, inputB.transpose(1, 2) )
            else:
                output = torch.bmm( inputA, inputB )

            _inputA, ctx.scaleA, ctx.zeroA = AsymmetricQuantFunction.apply(inputA, _quantize_bit)
            ctx.inputA = _inputA.type(torch.int8)

            _inputB, ctx.scaleB, ctx.zeroB = AsymmetricQuantFunction.apply(inputB, _quantize_bit)
            ctx.inputB = _inputB.type(torch.int8)



            ctx.transpose = transpose

        return output

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():


            transpose = ctx.transpose 

            inputA = linear_dequantize(ctx.inputA.type(torch.float), ctx.scaleA, ctx.zeroA)
            inputB = linear_dequantize(ctx.inputB.type(torch.float), ctx.scaleB, ctx.zeroB)
            
            if transpose:
                grad_inputA = torch.bmm(grad_output, inputB)
                grad_inputB = torch.bmm(grad_output.transpose(1, 2), inputA)
            else:
                grad_inputA = torch.bmm(grad_output, inputB.transpose(1, 2))
                grad_inputB = torch.bmm(inputA.transpose(1, 2), grad_output)

        return grad_inputA, grad_inputB, None
