from collections import namedtuple
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.utils.cpp_extension import load
from torch.autograd.function import Function
import pytorch_minimax

from quantize.conf import config
from C import calc_precision_dp, calc_precision, calc_avg_bits


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


class QScheme:

    num_samples = 0
    num_layers = 0
    batch = None
    update_scale = True
    layers = []

    def __init__(self, num_locations=1):
        self.initial_bits = config.initial_bits
        self.bits = config.activation_compression_bits
        assert QScheme.num_samples > 0
        self.scales = torch.ones(QScheme.num_samples)
        QScheme.layers.append(self)
        self.C = None
        self.dim = None
        self.num_locations = num_locations

        # debug
        self.name = 'layer_{}'.format(QScheme.num_layers)
        QScheme.num_layers += 1

    def get_scale(self):
        assert QScheme.batch is not None
        return self.scales[QScheme.batch]

    def set_scale(self, grad):
        if QScheme.update_scale:
            assert QScheme.batch is not None
            scale = (grad.view(grad.shape[0], -1) ** 2).sum(1).detach().cpu()
            # print(scale)
            self.scales[QScheme.batch] = scale

    def compute_quantization_bits(self, input):
        N = input.shape[0]
        D = input.shape[1]
        input_flatten = input.view(N, -1)

        # greedy
        grad_sum = self.get_scale().cuda()
        # print('grad sum', grad_sum)
        mn = pytorch_minimax.min(input_flatten)
        mx = pytorch_minimax.max(input_flatten)
        Range = mx - mn
        C = self.num_locations * D / 4 * Range ** 2 * grad_sum
        # print(C)
        self.C = C.cpu()
        self.dim = input.numel() // N
        b = torch.ones(N, dtype=torch.int32) * self.initial_bits
        w = torch.ones(N, dtype=torch.int32)
        b = calc_precision(b, self.C, w, int(self.bits * N))
        # print(self.initial_bits, b, self.C, w, int(self.bits * N))

        # TODO hack
        B = 2 ** b - 1
        self.var = (self.C / B**2).sum()

        if config.simulate:
            mn = mn.unsqueeze(1)
            mx = mx.unsqueeze(1)

        return b, mn, mx

    @staticmethod
    def allocate_perlayer():
        layers = QScheme.layers
        L = len(layers)

        Cs = [layer.C for layer in layers]
        C = torch.cat(Cs, 0)

        N = Cs[0].shape[0]

        Ws = [torch.ones(N, dtype=torch.int32) * layer.dim for layer in layers]
        w = torch.cat(Ws, 0)

        total_bits = w.sum() * config.activation_compression_bits
        b = torch.ones(N * L, dtype=torch.int32) * config.initial_bits
        b = calc_precision(b, C, w, total_bits)
        for i in range(L):
            bs = b[i*N : (i+1)*N]
            print(i, w[i*N], Cs[i].mean(), bs.float().mean(), bs)
            layers[i].bits = bs.float().mean()


##################################


class conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, scheme=None):
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
        ctx.saved = (q_input, q_bits, q_scale, q_min, weight, bias, batch_std, input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # TODO save_for_backward is not working, get RuntimeError: No grad accumulator for a saved leaf!
        # if config.swap:
        #     q_input, q_bits, q_scale, q_min, weight, batch_std = [x.cuda() if x is not None else x for x in ctx.saved_tensors]
        # else:
        #     q_input, q_bits, q_scale, q_min, weight, batch_std = ctx.saved_tensors
        q_input, q_bits, q_scale, q_min, weight, bias, batch_std, input = ctx.saved
        q_input_shape = ctx.other_args
        normalized = dequantize_mixed_precision(q_input, q_input_shape, q_bits, q_scale, q_min)

        # TODO: fused batch_norm
        grad_weight = (grad_output * normalized).sum((0, 2, 3))
        grad_bias = grad_output.sum((0, 2, 3))
        grad_normalized = grad_output * weight

        mean_grad_normalized = grad_normalized.mean((0, 2, 3), keepdim=True)
        mean_grad = (normalized * grad_normalized).mean((0, 2, 3), keepdim=True)
        grad_input = grad_normalized - mean_grad_normalized - normalized * mean_grad
        grad_input = grad_input / batch_std

        ctx.scheme.set_scale(mean_grad)

        # print('Saving')
        # torch.save([input, weight, bias, grad_output, grad_input], ctx.scheme.name + '.pt')

        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None


if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
