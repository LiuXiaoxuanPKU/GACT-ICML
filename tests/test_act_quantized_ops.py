"""Test the activation quantized ops"""

import math
import random

import numpy as np
import torch
from torch import nn, autograd
from torch.nn import init, functional as F
from torch.autograd.function import Function

from timeit_v2 import py_benchmark

from actnn.utils import get_memory_usage, compute_tensor_bytes
from actnn.ops import ext_quantization, op_quantize, op_dequantize
from actnn.ops import self_atten
from utils import setup_seed, error_rate

def test_quantize_mask():
    print("==========  Mask Quantization Eror Rate Test ==========")
    input = F.one_hot(torch.arange(0, 100) % 77).to(torch.uint8).to('cuda')
    d_input = op_dequantize(op_quantize(input.to(torch.float32), 1), input.shape)
    e = error_rate(d_input.to(torch.uint8), input)
    assert(e < 1e-10)

def test_quantize_error():
    input_shape = (64, 3, 224, 224)
    input = torch.rand(input_shape).to("cuda")
    ones_input = torch.ones(input_shape).to("cuda")
    print("==========  Quantization Error Rate Test ==========")
    print("1 bit error rate (value between 0 and 1): ")
    error_rate(op_dequantize(op_quantize(input, 1), input_shape), input)
    print("1 bit error rate (value = 0 or 1): ")
    error_rate(op_dequantize(op_quantize(ones_input, 1), input_shape), ones_input)
    print("2 bit error rate: ")
    error_rate(op_dequantize(op_quantize(input, 2), input_shape), input)
    print("4 bit error rate: ")
    error_rate(op_dequantize(op_quantize(input, 4), input_shape), input)
    print("8 bit error rate: ")
    error_rate(op_dequantize(op_quantize(input, 8), input_shape), input)

def test_quantize_big_min():
    input_shape = (65535 * 10, 256)
    input = torch.rand(input_shape).to("cuda")
    q_input, q_bit, q_scale, q_min = op_quantize(input, 4)
    ref_min = torch.min(input, dim=1).values
    q_min = q_min.reshape(1, -1)
    ref_min = ref_min.reshape(1, -1)
    np.testing.assert_allclose(q_min.cpu(), ref_min.cpu())

def test_quantize_big_error():
    input_shape = (65535 * 10, 512)
    input = torch.rand(input_shape).to("cuda")
    ones_input = torch.ones(input_shape).to("cuda")
    print("==========  Quantization Big Error Rate Test ==========")
    print("1 bit error rate (value between 0 and 1): ")
    error_rate(op_dequantize(op_quantize(input, 1), input_shape), input)
    print("1 bit error rate (value = 0 or 1): ")
    error_rate(op_dequantize(op_quantize(ones_input, 1), input_shape), ones_input)
    print("2 bit error rate: ")
    error_rate(op_dequantize(op_quantize(input, 2), input_shape), input)
    print("4 bit error rate: ")
    error_rate(op_dequantize(op_quantize(input, 4), input_shape), input)
    print("8 bit error rate: ")
    error_rate(op_dequantize(op_quantize(input, 8), input_shape), input)

def test_relu_correctness():
    print("========== ReLU Correctness Test ==========")

    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")
        data_np = np.random.randn(128, 56, 56, 31).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            output = func(data)
            output.backward(torch.ones_like(output))

            return [x.detach().cpu().numpy() for x in [output, data.grad]]

        output_ref, grad_data_ref = test_implementation(F.relu)
        output_us, grad_data_us = test_implementation(
            ext_quantization.act_quantized_relu)

        np.testing.assert_allclose(output_ref, output_us)
        np.testing.assert_allclose(grad_data_ref, grad_data_us)

def test_relu_memory():
    print("========== ReLU Memory Test ==========")

    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")
        data_np = np.random.randn(128, 56, 56, 32).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            before = get_memory_usage()

            for i in range(10):
                data = func(data)

            after = get_memory_usage()

            return after - before

        usage_ref = test_implementation(F.relu)
        usage_us = test_implementation(ext_quantization.act_quantized_relu)

        print("Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
        print("Quantized. Usage: %.2f MB" % (usage_us / 2 ** 20))

def test_relu_speed():
    print("========== ReLU Speed Test ==========")

    for dtype in ['float32', 'float16']:
        print(f"test {dtype}...")

        data_np = np.random.randn(256, 56, 56, 32).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            stmt = "func(data)"
            t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            output = func(data)
            head = torch.ones_like(output)
            stmt = "output.backward(head, retain_graph=True)"
            t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            return t_forward, t_backward

        forward_ref, backward_ref = test_implementation(F.relu)
        forward_us, backward_us = test_implementation(
            ext_quantization.act_quantized_relu)

        print("Exact.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
        print("Quantized. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))

def self_atten_ref_imp(dropout_p, q, k, v):
    k_feature = k.shape[-1]
    attention_scores = torch.matmul(q, k.transpose(-1, -2))
    # save tensor 1: attention score
    attention_scores = attention_scores / math.sqrt(k_feature)
    # save tensor 2: attention probility
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    dropout = nn.Dropout(dropout_p)
    attention_probs = dropout(attention_probs)
    context_layer = torch.matmul(attention_probs, v)
    return context_layer

def self_atten_opt_imp(dropout_p, q, k, v):
    return self_atten(dropout_p, q, k, v, q_chunk_size=512, k_chunk_size=256)

def test_self_atten_correctness():
    print("========== Self Multihead Attention Correctness Test ==========")
    def test_implementation(func):
        bz = 8
        num_head = 12
        seq_len = 512
        q_feature = k_feature = v_feature = 64
        torch.manual_seed(0)
        q = torch.rand((bz, num_head, seq_len, q_feature), dtype=torch.double).cuda().requires_grad_()
        k = torch.rand((bz, num_head, seq_len, k_feature), dtype=torch.double).cuda().requires_grad_()
        v = torch.rand((bz, num_head, seq_len, v_feature), dtype=torch.double).cuda().requires_grad_()
        dropout_p = 0

        output = func(dropout_p, q, k, v)
        head = torch.ones_like(output)
        output.backward(head, retain_graph=True)

        return q.grad, k.grad, v.grad, output

    setup_seed(0)
    ref_q_grad, ref_k_grad, ref_v_grad, ref_output = test_implementation(self_atten_ref_imp)
    setup_seed(0)
    opt_q_grad, opt_k_grad, opt_v_grad, opt_output = test_implementation(self_atten_opt_imp)

    # np.testing.assert_allclose(ref_output.detach().cpu(), opt_output.detach().cpu(), rtol=1e-5)
    np.testing.assert_allclose(ref_q_grad.cpu(), opt_q_grad.cpu(), rtol=1e-5)
    np.testing.assert_allclose(ref_k_grad.cpu(), opt_k_grad.cpu(), rtol=1e-5)
    np.testing.assert_allclose(ref_v_grad.cpu(), opt_v_grad.cpu(), rtol=1e-5)

def test_self_atten_memory():
    print("========== Self Multihead Attention Memory Test ==========")
    bz = 8
    num_head = 12
    seq_len = 2**11
    q_feature = k_feature = v_feature = 128
    q = torch.rand((bz, num_head, seq_len, q_feature)).cuda().requires_grad_()
    k = torch.rand((bz, num_head, seq_len, k_feature)).cuda().requires_grad_()
    v = torch.rand((bz, num_head, seq_len, v_feature)).cuda().requires_grad_()

    def test_implementation(func):
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        dropout_p = 0.2

        data = func(dropout_p, q, k, v)

        after = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        del data
        return after - before, peak_mem
    
    usage_us, peak_mem_op = test_implementation(self_atten_opt_imp)
    usage_ref, peak_mem_ref = test_implementation(self_atten_ref_imp)

    print("Reference. Usage: %.2f MB, Peak: %.2f" % (usage_ref / 2 ** 20, peak_mem_ref / 2 ** 20))
    print("Optimized. Usage: %.2f MB, Peak: %.2f" % (usage_us / 2 ** 20, peak_mem_op / 2 ** 20))

def test_self_atten_speed():
    print("========== Self Multihead Attention Speed Test ==========")
    def test_implementation(func):
        bz = 8
        num_head = 12
        seq_len = 1024
        q_feature = k_feature = v_feature = 128
        q = torch.rand((bz, num_head, seq_len, q_feature)).cuda().requires_grad_()
        k = torch.rand((bz, num_head, seq_len, k_feature)).cuda().requires_grad_()
        v = torch.rand((bz, num_head, seq_len, v_feature)).cuda().requires_grad_()
        dropout_p = 0.2

        stmt = "func(dropout_p, q, k, v)"
        t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

        output = func(dropout_p, q, k, v)
        head = torch.ones_like(output)
        stmt = "output.backward(head, retain_graph=True)"
        t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

        return t_forward, t_backward

    forward_ref, backward_ref = test_implementation(self_atten_ref_imp)
    forward_us, backward_us = test_implementation(self_atten_opt_imp)

    print("Ref.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
    print("Memory Optimized. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))

def test_self_atten_saved_tensors():
    print("========== Self Multihead Attention Saved Tensor Test ==========")
    def test_implementation(func):
        bz = 8
        num_head = 12
        seq_len = 512
        q_feature = k_feature = v_feature = 64
        torch.manual_seed(0)
        q = torch.rand((bz, num_head, seq_len, q_feature), dtype=torch.double).cuda().requires_grad_()
        k = torch.rand((bz, num_head, seq_len, k_feature), dtype=torch.double).cuda().requires_grad_()
        v = torch.rand((bz, num_head, seq_len, v_feature), dtype=torch.double).cuda().requires_grad_()
        dropout_p = 0

        output = func(dropout_p, q, k, v)
        head = torch.ones_like(output)
        output.backward(head, retain_graph=True)

        return q.grad, k.grad, v.grad, output

    def pack_hook(tensor):  # quantize hook
        if tensor.requires_grad:
            torch.cuda.synchronize()
            cur_mem = torch.cuda.memory_allocated()
            print("[Pack]", tensor.shape, "mem: ", cur_mem / 1024 / 1024, type(tensor.grad_fn))
        return tensor

    def unpack_hook(tensor):  # dequantize hook
        if tensor.requires_grad:
            torch.cuda.synchronize()
            cur_mem = torch.cuda.memory_allocated()
            print("[Unpack]", tensor.shape, "mem: ", cur_mem / 1024 / 1024, type(tensor.grad_fn))
        return tensor


    # setup_seed(0)
    # print("--------- Reference Implementation -----------------")
    # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    #     ref_q_grad, ref_k_grad, ref_v_grad, ref_output = test_implementation(self_atten_ref_imp)


    setup_seed(0)
    print("--------- Optimized Implementation -----------------")
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        opt_q_grad, opt_k_grad, opt_v_grad, opt_output = test_implementation(self_atten_opt_imp)


if __name__ == "__main__":
    test_quantize_mask()
    # test_quantize_big_min()
    # test_quantize_error()
    # test_quantize_big_error()
    # test_relu_correctness()
    # test_relu_memory()
    # test_relu_speed()
    # test_self_atten_correctness()
    # test_self_atten_memory()
    # test_self_atten_speed()
    # test_self_atten_saved_tensors()
