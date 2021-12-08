from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from actnn.conf import config
import actnn.cpp_extension.quantization as ext_quantization


def no_scheme_quantize_pack(input, q_bit):
    N = input.shape[0]
    input_flatten = input.view(N, -1)
    num_features = input_flatten.shape[1]

    # Compute min, max by groups
    if num_features % config.group_size != 0:
        # Padding
        new_num_features = (num_features // config.group_size + 1) * config.group_size
        delta = new_num_features - num_features
        input_flatten = torch.cat(
            [
                input_flatten,
                torch.zeros([N, delta], dtype=input.dtype, device=input.device),
            ],
            1,
        )

    input_groups = input_flatten.view(N, -1, config.group_size).contiguous()

    pack_func = ext_quantization.minimax_quantize_single_precision
    q_input, q_scale, q_min = pack_func(input_groups, q_bit)
    return q_input, q_scale, q_min


def dequantize_and_unpack(data, shape, bits, scale, mn, tid=-1):
    if config.simulate:
        data = data / scale + mn
    else:
        # Pad to group_size
        N = shape[0]
        num_features = int(np.prod(shape[1:]))
        group_size = config.group_size
        num_features = (
            num_features + (group_size - num_features % group_size) % group_size
        )

        # Unpack bitstream
        if isinstance(bits, int):
            unpack_func = ext_quantization.unpack_single_precision
        else:
            unpack_func = ext_quantization.unpack_mixed_precision
        data = unpack_func(
            data, bits, scale, mn, N, num_features // group_size, group_size
        )
    return data


def op_quantize(input, q_bit):
    q_input, q_scale, q_min = no_scheme_quantize_pack(input, q_bit)
    return [q_input, q_bit, q_scale, q_min]


def op_dequantize(input, input_shape):
    q_input, q_bit, q_scale, q_min = input
    input = dequantize_and_unpack(q_input, input_shape, q_bit, q_scale, q_min)

    # Remove padding
    N = input_shape[0]
    num_features = np.prod(input_shape[1:])
    input = input.view(N, -1)[:, :num_features]
    input = input.reshape(*input_shape).contiguous()
    return input
