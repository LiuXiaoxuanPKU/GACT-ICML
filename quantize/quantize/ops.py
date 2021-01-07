from collections import namedtuple
from functools import reduce
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.utils.cpp_extension import load
from torch.autograd.function import Function
import torch.distributed as dist


import pytorch_minimax
from quantize.conf import config

# Load cuda extensions
dirname = os.path.dirname(__file__)
ext_backward_func = load(name="ext_backward_func",
        sources=[os.path.join(dirname, "ext_backward_func.cc")], verbose=True)
ext_quantization = load(name="ext_quantization",
        sources=[os.path.join(dirname, "ext_quantization.cc"),
                 os.path.join(dirname, "ext_quantization_cuda_kernel.cu")], verbose=True)


QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

def quantize_mixed_precision(data, bits, mn, mx):
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
        output = (output - mn) * scale

        if config.stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)

        output = F.relu(output)
        output = torch.min(output, B.float()).round_().int()
    else:
        bits = bits.cuda()
        output, scale = ext_quantization.pack_mixed_precision(output, mn, mx, bits, config.stochastic)

    return output, bits, scale


def dequantize_mixed_precision(data, shape, bits, scale, mn):
    if not config.compress_activation:
        return data

    if config.simulate:
        data = data / scale + mn
    else:
        N = shape[0]
        num_features = int(np.prod(shape[1:]))
        group_size = config.group_size
        # Pad to group_size
        num_features = (num_features + (group_size - num_features % group_size) % group_size)

        #assert other_dim % config.group_size == 0, f"{shape}, {other_dim}, {config.group_size}"
        data = ext_quantization.unpack_mixed_precision(data, bits,
                scale, mn, N, num_features // config.group_size, config.group_size)
    return data


def quantize_activation(input, scheme):
    N = input.shape[0]
    input_groups, q_bits, q_min, mx = scheme.compute_quantization_bits(input)
    q_input, q_bits, q_scale = \
        quantize_mixed_precision(input_groups, q_bits, q_min, mx)

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
            output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
            quantized = quantize_activation(input, scheme)

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

        ctx.scheme.if_allocate_perlayer()
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

        ctx.scheme.if_allocate_perlayer()
        return grad_input, grad_weight, grad_bias, None


class batch_norm(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias,
                training, exponential_average_factor, eps, scheme):
        with torch.no_grad():
            output = F.batch_norm(
                input, running_mean, running_var, weight, bias,
                training, exponential_average_factor, eps)

            batch_mean = input.mean((0, 2, 3))      # TODO: compute these with cuDNN
            batch_std = torch.sqrt(input.var((0, 2, 3)) + eps)

            quantized = quantize_activation(input, scheme)

        ctx.scheme = scheme
        # TODO save_for_backward is not working, get RuntimeError: No grad accumulator for a saved leaf!
        ctx.other_args = input.shape
        ctx.saved = (quantized, weight, running_mean, running_var, batch_mean, batch_std, bias, eps)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # TODO save_for_backward is not working, get RuntimeError: No grad accumulator for a saved leaf!
        quantized, weight, running_mean, running_var, batch_mean, batch_std, bias, eps = ctx.saved
        q_input_shape = ctx.other_args
        input = dequantize_activation(quantized, q_input_shape)

        grad_input, grad_weight, grad_bias = ext_backward_func.cudnn_batch_norm_backward(
            input, grad_output, weight, running_mean, running_var, batch_mean, 1 / batch_std, eps, torch.Tensor()
        )

        ctx.scheme.if_allocate_perlayer()
        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None


class sync_batch_norm(Function):
    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size, scheme):
        input = input.contiguous()

        count = torch.empty(1,
                            dtype=running_mean.dtype,
                            device=input.device).fill_(input.numel() // input.size(1))

        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)

        num_channels = input.shape[1]
        # C, C, 1 -> (2C + 1)
        combined = torch.cat([mean, invstd, count], dim=0)
        # world_size * (2C + 1)
        combined_list = [
            torch.empty_like(combined) for k in range(world_size)
        ]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(combined_list, combined, process_group, async_op=False)
        combined = torch.stack(combined_list, dim=0)
        # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
        mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)

        size = count_all.view(-1).long().sum()
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1)
        )

        quantized = quantize_activation(input, scheme)
        self.saved = quantized
        self.save_for_backward(weight, mean, invstd, count_all)
        self.scheme = scheme
        self.other_args = input.shape
        self.process_group = process_group

        # apply element-wise normalization
        out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        return out

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        quantized = self.saved
        q_input_shape = self.other_args
        saved_input = dequantize_activation(quantized, q_input_shape)
        del quantized, self.saved          # TODO, lianmin: check

        weight, mean, invstd, count_tensor = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group

        # calculate local stats as well as grad_weight / grad_bias
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            self.needs_input_grad[0],
            self.needs_input_grad[1],
            self.needs_input_grad[2]
        )

        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            # TODO: move div_ into batch_norm_backward_elemt kernel
            num_channels = sum_dy.shape[0]
            combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
            torch.distributed.all_reduce(
                combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
            sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

            divisor = count_tensor.sum()
            mean_dy = sum_dy / divisor
            mean_dy_xmu = sum_dy_xmu / divisor
            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                mean_dy,
                mean_dy_xmu
            )

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        self.scheme.if_allocate_perlayer()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated


if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
