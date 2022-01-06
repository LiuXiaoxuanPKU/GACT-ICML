from collections import namedtuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from actnn.conf import config
import actnn.cpp_extension.quantization as ext_quantization
import torch.utils.checkpoint as checkpoint


def no_scheme_quantize_pack(input, q_bit):
    N = input.shape[0]
    input_flatten = input.reshape(N, -1)
    num_features = input_flatten.shape[1]

    # Compute min, max by groups
    if num_features % config.group_size != 0:
        # Padding
        new_num_features = (
            num_features // config.group_size + 1) * config.group_size
        delta = new_num_features - num_features
        input_flatten = torch.cat(
            [
                input_flatten,
                torch.zeros([N, delta], dtype=input.dtype,
                            device=input.device),
            ],
            1,
        )

    input_groups = input_flatten.view(N, -1, config.group_size).contiguous()

    pack_func = ext_quantization.minimax_quantize_single_precision
    if q_bit == 32:  # TODO, use kernel to optimize this
        q_min = input_groups.min(dim=-1).values
        q_scale = input_groups.max(dim=-1).values - q_min
        q_input = input_groups
    else:
        q_input, q_scale, q_min = pack_func(input_groups, q_bit)
    return q_input, q_scale, q_min


def dequantize_and_unpack(data, shape, q_bit, scale, mn):
    # Pad to group_size
    N = shape[0]
    num_features = int(np.prod(shape[1:]))
    group_size = config.group_size
    num_features = (
        num_features + (group_size - num_features %
                        group_size) % group_size
    )

    if not isinstance(q_bit, int):
        print("bits must be intergers, now bits ", q_bit)
        assert(False)

    if q_bit == 32:
        pass
    else:
        unpack_func = ext_quantization.unpack_single_precision
        data = unpack_func(
            data, q_bit, scale, mn, N, num_features // group_size, group_size
        )
    return data


def op_quantize(input, q_bit):
    q_input, q_scale, q_min = no_scheme_quantize_pack(input, q_bit)
    return [q_input, q_bit, q_scale, q_min]


def op_dequantize(input, input_shape, inject_noise):
    q_input, q_bit, q_scale, q_min = input
    input = dequantize_and_unpack(
        q_input, input_shape, q_bit, q_scale, q_min)

    if inject_noise:    # Save and load rng state
        rng_state = torch.get_rng_state()
        noise = torch.randn_like(input)
        torch.set_rng_state(rng_state)
        bin_size = q_scale / 4
        # print(bin_size.shape, q_scale.shape, input.shape)
        input = input + noise * bin_size

    # Remove padding
    N = input_shape[0]
    num_features = np.prod(input_shape[1:])
    input = input.view(N, -1)[:, :num_features]
    input = input.reshape(*input_shape).contiguous()

    return input


def self_atten(dropout_p, query_layer, key_layer, value_layer,
               q_chunk_size, k_chunk_size, use_checkpoint=True):
    batch_size, num_heads, seq_len, q_features = query_layer.shape
    batch_size, num_heads, seq_len, k_features = key_layer.shape
    batch_size, num_heads, seq_len, v_features = value_layer.shape
    q_chunk_size = min(q_chunk_size, seq_len)
    dropout = nn.Dropout(dropout_p)

    def _query_chunk_attention(query, key, value):
        batch_size, num_heads, num_kv, k_features = key.shape
        v_features = value.shape[-1]
        key_chunk_size = min(k_chunk_size, num_kv)
        num_key_chunk = math.ceil(num_kv / key_chunk_size)
        query = query / math.sqrt(k_features)

        def summarize_chunk(query, key, value):
            attn_weights = torch.einsum('bhqd,bhkd->bhqk', query, key)
            max_score = torch.max(attn_weights, axis=-1, keepdims=True).values
            max_score = max_score.detach()
            exp_weights = torch.exp(attn_weights - max_score)
            exp_values = torch.einsum('bhvf,bhqv->bhqf', value, exp_weights)
            exp_values = dropout(exp_values)
            return (exp_values, exp_weights.sum(axis=-1), max_score.squeeze())

        chunk_values = None
        chunk_weights = None
        global_max = None

        def batch_dot(m1, m2):
            feature_size = m1.shape[-1]
            v = m1.reshape(-1, feature_size) * m2.reshape(-1, 1)
            return v.reshape(m1.shape)

        for i in range(num_key_chunk):
            key_chunk = key[:, :, i *
                            key_chunk_size: (i+1) * key_chunk_size, :]
            value_chunk = value[:, :, i *
                                key_chunk_size: (i+1) * key_chunk_size, :]
            if use_checkpoint:
                chunk_value, chunk_weight, chunk_max = \
                    torch.utils.checkpoint.checkpoint(
                        summarize_chunk, query, key_chunk, value_chunk)
            else:
                chunk_value, chunk_weight, chunk_max = summarize_chunk(
                    query, key_chunk, value_chunk)

            # with torch.no_grad():
            if global_max is None:
                global_max = chunk_max
                chunk_values = chunk_value
                chunk_weights = chunk_weight
            else:
                old_max = global_max
                global_max = torch.maximum(chunk_max, global_max).detach()

                diff1 = torch.exp(chunk_max - global_max).detach()
                chunk_value = batch_dot(chunk_value, diff1)
                chunk_weight *= diff1

                diff2 = torch.exp(old_max - global_max).detach()
                chunk_values = batch_dot(chunk_values, diff2)
                chunk_weights *= diff2

                chunk_values += chunk_value
                chunk_weights += chunk_weight

        chunk_values = chunk_values.reshape(-1, chunk_values.shape[-1])
        chunk_weights = chunk_weights.reshape(-1, 1)
        return chunk_values / chunk_weights

    num_q_chunk = math.ceil(query_layer.shape[2] / q_chunk_size)
    res = torch.zeros(query_layer.shape).cuda()
    for i in range(num_q_chunk):
        r = _query_chunk_attention(query_layer[:, :, i*q_chunk_size:(i+1)*q_chunk_size, :],
                                   key_layer, value_layer)
        res[:, :, i*q_chunk_size:(i+1)*q_chunk_size, :] = r.reshape(
            batch_size, num_heads, q_chunk_size, q_features)
    return res
