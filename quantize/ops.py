from collections import namedtuple
import torch
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
import pytorch_minimax

from quantize.conf import config
from quantize.C import calc_precision_dp, calc_precision


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


class GradRecorder(InplaceFunction):
    @staticmethod
    def forward(ctx, input, name):
        ctx.name = name
        return input

    @staticmethod
    def backward(ctx, grad):
        scale = (grad.view(grad.shape[0], -1) ** 2).sum(1)
        QF.set_scale(ctx.name, scale.detach().cpu())
        return grad, None


class MixedPrecisionQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, bits, stochastic=False, inplace=False):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        with torch.no_grad():
            B = (2 ** bits - 1).unsqueeze(1)
            x_shape = output.shape
            output = output.view(x_shape[0], -1)
            # print(x_shape, output.shape)
            mn = pytorch_minimax.min(output).unsqueeze(1) - 1e-8
            mx = pytorch_minimax.max(output).unsqueeze(1) + 1e-8
            # print(mn[:5], mx[:5], B[:5])
            scale = B / (mx - mn)
            output = (output - mn) * scale

            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)

            output = F.relu(output)
            output = torch.min(output, B).round_()

            output = output / scale + mn
            output = output.view(*x_shape)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None


# Quantized Activation Layers
class QF:
    @staticmethod
    def init(num_samples):
        # print('Initialized batch size ', num_samples)
        QF.num_samples = num_samples
        QF.scales = {}
        QF.update_scale = True
        QF.training = True

    @staticmethod
    def set_current_batch(ids):
        # print('Set current batch ', ids)
        QF.ids = ids

    @staticmethod
    def get_scale(name):
        if not name in QF.scales:
            QF.scales[name] = torch.ones(QF.num_samples)

        return QF.scales[name][QF.ids]

    @staticmethod
    def set_scale(name, scale):
        if QF.update_scale and QF.training:
            if not name in QF.scales:
                QF.scales[name] = torch.ones(QF.num_samples)

            QF.scales[name][QF.ids] = scale
            # print('Set scale ', name, scale)

    @staticmethod
    def quantize(input, name):
        if config.activation_compression_bits >= 32 or not QF.training:
            # print('Skipping')
            return input

        N = input.shape[0]
        D = input.shape[1]
        input_flatten = input.view(N, -1)

        if config.alg == 'greedy':
            grad_sum = QF.get_scale(name).cpu()
            mn = pytorch_minimax.min(input_flatten).cpu()
            mx = pytorch_minimax.max(input_flatten).cpu()
            Range = mx - mn
            C = D / 4 * Range ** 2 * grad_sum
            b = torch.ones(N, dtype=torch.int32) * config.initial_bits
            b = calc_precision(b, C, config.activation_compression_bits * N).float()

            mask = 1.0
        else:   # DP. only work for convolution
            grad_sum = QF.get_scale(name).cpu()
            input_sum = (input_flatten ** 2).sum(1).cpu()

            mn = pytorch_minimax.min(input_flatten).cpu()
            mx = pytorch_minimax.max(input_flatten).cpu()
            Range = mx - mn

            C = D / 4 * Range ** 2 * grad_sum
            A = input_sum * grad_sum

            b, keep_prob = calc_precision_dp(A, C, config.initial_bits, config.activation_compression_bits, 2)
            mask = torch.distributions.Bernoulli(probs=keep_prob).sample() / keep_prob

            with torch.no_grad():
                mask = mask.cuda()
                if len(input.shape) == 2:
                    mask = mask.unsqueeze(1)
                else:
                    mask = mask.view(N, 1, 1, 1)

        output = MixedPrecisionQuantize().apply(input, b.cuda(), True, False) * mask
        return output

    @staticmethod
    def conv2d(input, weight, bias, stride, padding, dilation, groups, name):
        assert(bias is None)
        # Correct output, and correct L'input, incorrect L'weight
        output = F.conv2d(input, weight.detach(), bias, stride, padding, dilation, groups)
        if not QF.training:
            return output

        qinput = QF.quantize(input, name)
        # Incorrect output, incorrect L'input, correct L'weight
        fake_output = F.conv2d(qinput.detach(), weight, bias, stride, padding, dilation, groups)
        fake_output = GradRecorder().apply(fake_output, name)
        # output = GradRecorder().apply(output, name)

        return output + fake_output - fake_output.detach()

##############
class MyLinear_apply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None, name=None):

        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        _input = QF.quantize(input.detach(), name)
        ctx.save_for_backward(_input, weight, bias)
        ctx.name = name

        # output = GradRecorder().apply(output, name)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        # Record gradient
        scale = (grad_output ** 2).sum(1)
        QF.set_scale(ctx.name, scale.detach().cpu())

        grad_input = grad_weight = grad_bias = None

        grad_input = grad_output.mm(weight)

        grad_weight = grad_output.t().mm(input)
        if bias is not None:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None # The name does not need gradient.


class batch_norm(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias,
                training, exponential_average_factor, eps, name):
        output = F.batch_norm(
            input, running_mean, running_var, weight, bias,
            training, exponential_average_factor, eps)

        batch_mean = input.mean((0, 2, 3), keepdim=True)
        batch_std = torch.sqrt(input.var((0, 2, 3), keepdim=True) + eps)

        normalized = (input - batch_mean) / batch_std

        weight = weight.view(1, -1, 1, 1)
        bias = bias.view(1, -1, 1, 1)

        # ctx.save_for_backward(batch_std, weight, normalized)
        ctx.batch_std = batch_std
        ctx.weight = weight
        ctx.normalized = QF.quantize(normalized, name)
        ctx.name = name

        # my_output = normalized * weight + bias
        # print(my_output.norm())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Record gradient
        scale = (grad_output.view(grad_output.shape[0], -1) ** 2).sum(1)
        QF.set_scale(ctx.name, scale.detach().cpu())

        # batch_std, weight, normalized = ctx.saved_tensors
        batch_std = ctx.batch_std
        weight = ctx.weight
        normalized = ctx.normalized

        grad_weight = (grad_output * normalized).sum((0, 2, 3))
        grad_bias = grad_output.sum((0, 2, 3))
        grad_normalized = grad_output * weight

        mean_grad_normalized = grad_normalized.mean((0, 2, 3), keepdim=True)
        grad_input = grad_normalized - mean_grad_normalized - normalized * \
                     (normalized * grad_normalized).mean((0, 2, 3), keepdim=True)
        grad_input = grad_input / batch_std
        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None

class my_layer_norm(Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps, name):
        mean = input.mean(dim=-1, keepdim=True)
        var = ((input - mean) ** 2).mean(dim=-1, keepdim=True)
        std = var.sqrt().add_(eps)

        normalized = (input - mean) / std 
        ctx.mean = mean
        ctx.std = std 
        ctx.normalized = QF.quantize(normalized.detach(), name)
        ctx.weight = weight
        ctx.name = name


        output = normalized * output + bias 

        return output 

    def backward(ctx, grad_output):
        # we may do not need scale here
        scale = (grad_output.view(grad_output.shape[0], -1) ** 2).sum(1) # for NLP, input is seq_length * batch_size * token_size, this one is wrong.
        QF.set_scale(ctx.name, scale.detach().cpu())

        std = ctx.std 


class qsoftmax(Function):
    @staticmethod
    def forward(ctx, input, dim, name):
        output = F.softmax(input)
        ctx.dim = dim
        ctx.name = name
        ctx.output = QF.quantize(output, name)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Record gradient
        scale = (grad_output.view(grad_output.shape[0], -1) ** 2).sum(1)
        QF.set_scale(ctx.name, scale.detach().cpu())

        output = ctx.output
        grad = output * (grad_output - (output * grad_output).sum(ctx.dim, keepdim=True))
        return grad, None, None


if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
