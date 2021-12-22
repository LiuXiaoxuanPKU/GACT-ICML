from collections import namedtuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from actnn.conf import config
import actnn.cpp_extension.quantization as ext_quantization
from actnn.utils import compute_tensor_bytes, empty_cache

import torch.utils.checkpoint as checkpoint

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
    empty_cache(2)
    peak_mem = torch.cuda.max_memory_allocated()
    alloc_mem = torch.cuda.memory_allocated()
    tensor_size = compute_tensor_bytes(input)
    del input
    print("-------Peak", peak_mem / 1024 / 1024)
    print("Alloc mem ", alloc_mem / 1024 / 1024)
    print("Input size ",  tensor_size / 1024 / 1024)
    torch.cuda.reset_peak_memory_stats()
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


def self_atten(query_layer, key_layer, value_layer, q_chunk_size, k_chunk_size):
    batch_size, num_heads, seq_len, q_features = query_layer.shape
    batch_size, num_heads, seq_len, k_features = key_layer.shape
    batch_size, num_heads, seq_len, v_features = value_layer.shape

    def _query_chunk_attention(query, key, value):
        batch_size, num_heads, num_kv, k_features = key.shape
        v_features = value.shape[-1]
        key_chunk_size = min(k_chunk_size, num_kv)
        num_key_chunk = math.ceil(num_kv / key_chunk_size)
        query = query / math.sqrt(k_features)

        def summarize_chunk(query, key, value):
            attn_weights = torch.einsum('bhqd,bhkd->bhqk', query, key) 
            # attn_weights = torch.einsum('qhd,khd->qhk', query, key) 
            max_score = torch.max(attn_weights, axis=-1, keepdims=True).values
            max_score = max_score.detach()
            exp_weights = torch.exp(attn_weights - max_score)
            exp_values = torch.einsum('bhvf,bhqv->bhqf', value, exp_weights)
            # exp_values = torch.einsum('vhf,qhv->qhf', value, exp_weights)
            return (exp_values, exp_weights.sum(axis=-1),
                max_score.reshape((query.shape[0], query.shape[1], query.shape[2])))

        chunk_values = None
        chunk_weights = None
        for i in range(num_key_chunk):
            key_chunk = key[:, :, i * key_chunk_size : (i+1) * key_chunk_size, :]
            value_chunk = value[:, :, i * key_chunk_size : (i+1) * key_chunk_size, :]
            chunk_value, chunk_weight, chunk_max =  checkpoint.checkpoint(
                    summarize_chunk, query, key_chunk, value_chunk)# TODO
            print("chunk_value", chunk_value.shape)
            print("chunk_weight", chunk_weight.shape)
            print("chunk_max", chunk_max.shape)
            global_max = torch.max(chunk_max, axis=2, keepdims=True).values
            max_diffs = torch.exp(chunk_max - global_max) 
            
            chunk_value = chunk_value.reshape(-1, k_feature)
            max_diffs = max_diffs.reshape((-1, 1))
            chunk_value *= max_diffs
            chunk_value = chunk_value.reshape(query.shape)
            max_diffs = max_diffs.reshape(query.shape[0], query.shape[1], query.shape[2])
            chunk_weight *= max_diffs

            if chunk_values is None:
                chunk_values = chunk_value
            else:
                chunk_values += chunk_value
            
            if chunk_weights is None:
                chunk_weights = chunk_weight
            else:
                chunk_weights += chunk_weight

        chunk_values = chunk_values.reshape(-1, chunk_values.shape[-1])
        chunk_weights = chunk_weights.reshape(-1, 1)
        return chunk_values / chunk_weights

    num_q_chunk = math.ceil(query_layer.shape[2] / q_chunk_size)
    res = []
    for i in range(num_q_chunk):
        r = _query_chunk_attention(query_layer[:, :, i*q_chunk_size:(i+1)*q_chunk_size, :], 
                    key_layer, value_layer)
        res.append(r)
    re = torch.stack(res)
    return re.reshape(bz, num_heads, seq_len, k_features)

if __name__ == "__main__":
    bz = 8
    num_head = 12
    seq_len = 512
    q_feature = k_feature = v_feature = 64
    q = torch.rand((bz, num_head, seq_len, q_feature)).cuda()
    k = torch.rand((bz, num_head, seq_len, k_feature)).cuda()
    v = torch.rand((bz, num_head, seq_len, v_feature)).cuda()

    def ref_imp(q, k, v):
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(k_feature)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, v)
        return context_layer

    def self_atten_imp(q, k, v):
        return self_atten(q, k, v, q_chunk_size=64, k_chunk_size=64)
    
    ref_result = ref_imp(q, k, v)
    op_result = self_atten_imp(q, k, v)


    np.allclose(ref_result.cpu(), op_result.cpu())

    def test_implementation(func):
        bz = 8
        num_head = 12
        seq_len = 512
        q_feature = k_feature = v_feature = 64
        q = torch.rand((bz, num_head, seq_len, q_feature)).cuda().requires_grad_()
        k = torch.rand((bz, num_head, seq_len, k_feature)).cuda().requires_grad_()
        v = torch.rand((bz, num_head, seq_len, v_feature)).cuda().requires_grad_()

        before = torch.cuda.memory_allocated()

        for i in range(10):
            data = func(q, k, v)

        after = torch.cuda.memory_allocated()

        return after - before
    
    usage_ref = test_implementation(ref_imp)
    usage_us = test_implementation(self_atten_imp)

    print("Reference.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
    print("Optimized. Usage: %.2f MB" % (usage_us / 2 ** 20))