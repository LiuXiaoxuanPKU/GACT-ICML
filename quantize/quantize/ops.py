from collections import namedtuple
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.utils.cpp_extension import load
from torch.autograd.function import Function
import torch.distributed as dist

from quantize.conf import config
from quantize.utils import get_memory_usage, compute_tensor_bytes, empty_cache

# Load cuda extensions
dirname = os.path.dirname(__file__)
ext_backward_func = load(name="ext_backward_func",
    sources=[os.path.join(dirname, "ext_backward_func.cc")], verbose=False)
ext_quantization = load(name="ext_quantization",
    sources=[os.path.join(dirname, "ext_quantization.cc"),
             os.path.join(dirname, "ext_quantization_cuda_kernel.cu")], verbose=False)
ext_minimax = load(name="ext_minimax",
    sources=[os.path.join(dirname, "ext_minimax.cc"),
             os.path.join(dirname, "ext_minimax_cuda_kernel.cu")], verbose=False)


QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

def quantize_mixed_precision(data, bits, mn, mx):
    if config.simulate:
        N = data.shape[0]
        output = data   # N, groups, group_dim

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
        output, scale = ext_quantization.pack_mixed_precision(data, mn, mx, bits, config.stochastic)
        if config.swap:
            output = output.cpu()

    return output, scale


def dequantize_mixed_precision(data, shape, bits, scale, mn):
    if config.simulate:
        data = data / scale + mn
    else:
        if config.swap:
            data = data.cuda()
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
    if not config.compress_activation:
        return input, None, None, None

    N = input.shape[0]
    input_groups, q_bits, q_min, mx = scheme.compute_quantization_bits(input)
    q_input, q_scale = quantize_mixed_precision(input_groups, q_bits, q_min, mx)

    if q_scale is None:
        return q_input, q_bits

    return q_input, q_bits, q_scale.to(torch.bfloat16), q_min.to(torch.bfloat16)  # TODO convert q_bits to int8


def dequantize_activation(quantized, q_input_shape):
    if not config.compress_activation:
        return quantized[0]

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
    num_features = np.prod(q_input_shape[1:])
    input = input.view(N, -1)[:, :num_features]
    input = input.view(*q_input_shape)
    return input

conv2d_layer_ct = 0
bn_layer_ct = 0
total_act_mem = 0

class conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, scheme=None):
        # if not ctx.needs_input_grad[1]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]
        #     return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        quantized = quantize_activation(input, scheme)

        ctx.scheme = scheme
        ctx.saved = quantized, weight, bias
        ctx.other_args = (input.shape, stride, padding, dilation, groups)

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_forward:
            global conv2d_layer_ct, total_act_mem
            print("========== conv2d forward %d ==========" % conv2d_layer_ct)
            get_memory_usage(True)
            conv2d_layer_ct += 1
            total_act_mem += compute_tensor_bytes(quantized)
            print("Act mem: %.2f MB" % (total_act_mem / 1024 ** 2))

        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        # if not ctx.needs_input_grad[1]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]
        #     return None, None, None, None, None, None, None, None
        ctx.scheme.set_scale(grad_output)

        q_input_shape, stride, padding, dilation, groups = ctx.other_args
        padding = _pair(padding)
        stride = _pair(stride)
        dilation = _pair(dilation)

        quantized, weight, bias = ctx.saved
        input = dequantize_activation(quantized, q_input_shape)
        del quantized, ctx.saved

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_backward:
            global conv2d_layer_ct
            print("========== conv2d backward %d ==========" % conv2d_layer_ct)
            get_memory_usage(True)
            conv2d_layer_ct += 1
            print("WS: %.2f MB" % (compute_tensor_bytes([grad_output, input, input]) / 1024 ** 2))

        ws_mem = compute_tensor_bytes([grad_output, input, input])
        if (config.pipeline_threshold and ws_mem > config.pipeline_threshold and
                ctx.needs_input_grad[1] and ctx.needs_input_grad[0]):
            use_pipeline = True
        else:
            use_pipeline = False

        if use_pipeline:
            micro_batch_size = (ws_mem + config.pipeline_threshold) // config.pipeline_threshold
            raw_input = input
            raw_grad_output = grad_output
            input = torch.chunk(input, micro_batch_size)
            grad_output = torch.chunk(grad_output,  micro_batch_size)
            grad_weight = None

            for i in range(micro_batch_size):
                input[i][:], grad_weight_tmp = ext_backward_func.cudnn_convolution_backward(
                        input[i], grad_output[i], weight, padding, stride, dilation, groups,
                        False, False, False, [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])
                if grad_weight is None:
                    grad_weight = grad_weight_tmp
                else:
                    grad_weight += grad_weight_tmp
            grad_input = raw_input
            grad_output = raw_grad_output
        else:
            grad_input, grad_weight = ext_backward_func.cudnn_convolution_backward(
                input, grad_output, weight, padding, stride, dilation, groups,
                False, False, False, [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3])
        else:
            grad_bias = None

        ctx.scheme.if_allocate_perlayer()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, scheme=None):
        quantized = quantize_activation(input, scheme)

        empty_cache(config.empty_cache_threshold)

        ctx.scheme = scheme
        ctx.saved = quantized, weight, bias
        ctx.other_args = input.shape

        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        ctx.scheme.set_scale(grad_output)

        quantized, weight, bias = ctx.saved
        q_input_shape = ctx.other_args

        input = dequantize_activation(quantized, q_input_shape)
        del quantized, ctx.saved

        empty_cache(config.empty_cache_threshold)

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
        # if not ctx.needs_input_grad[3]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]
        #     return ext_backward_func.cudnn_batch_norm(
        #         input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)[0]

        quantized = quantize_activation(input, scheme)

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_forward:
            global bn_layer_ct, total_act_mem
            print("========== bn forward %d ==========" % bn_layer_ct)
            get_memory_usage(True)
            bn_layer_ct += 1
            total_act_mem += compute_tensor_bytes(quantized)
            print("Act mem: %.2f MB" % (total_act_mem / 1024 ** 2))

        if training:
            output, save_mean, save_var, reserve = ext_backward_func.cudnn_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
        else:
            output, save_mean, save_var = ext_backward_func.native_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
            reserve = None

        ctx.scheme = scheme
        ctx.other_args = input.shape
        ctx.saved = (quantized, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # if not ctx.needs_input_grad[3]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]
        #     return None, None, None, None, None, None, None, None, None
        quantized, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve = ctx.saved

        q_input_shape = ctx.other_args

        input = dequantize_activation(quantized, q_input_shape)
        del quantized, ctx.saved

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_backward:
            global bn_layer_ct
            print("========== bn backward %d ==========" % bn_layer_ct)
            get_memory_usage(True)
            bn_layer_ct += 1

        if training:
            grad_input, grad_weight, grad_bias = ext_backward_func.cudnn_batch_norm_backward(
                input, grad_output, weight, running_mean, running_var, save_mean, save_var, eps, reserve)
        else:
            grad_input, grad_weight, grad_bias = ext_backward_func.native_batch_norm_backward(
                grad_output, input, weight, running_mean, running_var, save_mean, save_var, training, eps,
                [ctx.needs_input_grad[0], ctx.needs_input_grad[3], ctx.needs_input_grad[4]]
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
        return torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()

        quantized = self.saved
        q_input_shape = self.other_args
        saved_input = dequantize_activation(quantized, q_input_shape)
        del quantized, self.saved

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


class adaptive_avg_pool2d(Function):
    @staticmethod
    def forward(ctx, input, output_size):
        assert output_size == (1, 1)
        ctx.saved = input.shape
        return torch.mean(input, dim=[2, 3], keepdim=True)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.saved
        repeat_size = [int(x / y) for x, y in zip(input_shape, grad_output.shape)]
        return grad_output.repeat(repeat_size) / np.prod(repeat_size), None


if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
