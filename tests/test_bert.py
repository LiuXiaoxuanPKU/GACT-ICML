import torch
import torch.nn as nn
import numpy as np
import random
import actnn 
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer, BertAttention, BertModel
from timeit_v2 import py_benchmark

def error_rate(grad_ref, grad_q):
    return ((grad_q - grad_ref)**2).sum() / ((grad_ref**2).sum() + 1e-10)

def test_layernorm_quantize():
    print("===============Quantize Layernorm Correctness==================")
    bz, seq_len, feature = 8, 512, 768
    np.random.seed(0)
    data_np = np.random.randn(bz, seq_len, feature).astype("float32")

    def test_implementation(func):
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        output = func(data)
        output.backward(torch.ones_like(output))
        return  [x.detach().cpu().numpy() for x in [output, data.grad, func.weight.grad, func.bias.grad]]

    layernorm_org = nn.LayerNorm(feature,eps=0).cuda()
    output_org, grad_org, grad_wei_org, grad_bias_org = test_implementation(layernorm_org)

    def pack_hook(input):
        return controller.quantize(input)
    def unpack_hook(input):
        return controller.dequantize(input)

    layernorm_q = nn.LayerNorm(feature,eps=0).cuda()
    controller = actnn.controller.Controller(default_bit=8, swap=False, debug=False, prefetch=False)
    controller.filter_tensors(layernorm_q.named_parameters())
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        output_q, grad_q, grad_wei_q, grad_bias_q = test_implementation(layernorm_q)
    
    np.testing.assert_allclose(grad_org, grad_q, atol=1e-2, rtol=1e-2)
    # np.testing.assert_allclose(grad_wei_org, grad_wei_q, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(grad_bias_org, grad_bias_q, atol=1e-2, rtol=1e-2)

def test_self_atten_correctness():
    print("===============Self Attention Correctness==================")
    bz = 8
    seq_len = 512
    feature = 768
    data_np = np.random.randn(bz, seq_len, feature).astype("float32")

    def test_implementation(func):
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        output = func(data)
        output[0].backward(torch.ones_like(output[0]))
        return  [x.detach().cpu().numpy() for x in [output[0], data.grad]]

    model_name = "bert-base-uncased"
    config_opt = AutoConfig.from_pretrained(model_name, num_labels=2)
    config_opt.efficient_softmax = True
    config_opt.nested_checkpoint = False
    torch.manual_seed(0)
    layer_opt = BertSelfAttention(config_opt, position_embedding_type=None).cuda()

    config_org = AutoConfig.from_pretrained(model_name, num_labels=2)
    config_org.efficient_softmax = False
    config_org.nested_checkpoint = False
    torch.manual_seed(0)
    layer_org = BertSelfAttention(config_org, position_embedding_type=None).cuda()

    output_opt, grad_opt = test_implementation(layer_opt)
    output_org, grad_org = test_implementation(layer_org)

    np.testing.assert_allclose(output_opt, output_org, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_opt, grad_org, rtol=1e-5, atol=1e-5)

    #======================= Checkpoint ================================
    config_ckpt = AutoConfig.from_pretrained(model_name, num_labels=2)
    torch.manual_seed(0)
    config_opt.efficient_softmax = False
    config_opt.nested_checkpoint = True
    layer_ckpt = BertSelfAttention(config_opt, position_embedding_type=None).cuda()
    output_ckpt, grad_ckpt = test_implementation(layer_ckpt)
    np.testing.assert_allclose(output_ckpt, output_org, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(grad_ckpt, grad_org, atol=1e-5, rtol=1e-5)

def test_bert_layer_correctness():
    print("===============Bert Layer Correctness==================")
    bz, seq_len, feature = 8, 512, 768
    data_np = np.random.randn(bz, seq_len, feature).astype("float32")
    
    def test_implementation(func):
        torch.manual_seed(0)
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        output = func(data)
        output[0].backward(torch.ones_like(output[0]))
        return  [x.detach().cpu().numpy() for x in [output[0], data.grad]]

    model_name = "bert-base-uncased"
    config_effi = AutoConfig.from_pretrained(model_name, num_labels=2)
    torch.manual_seed(0)
    config_effi.efficient_softmax = True
    config_effi.nested_checkpoint = False
    layer_effi = BertLayer(config_effi).cuda()

    config_ckpt = AutoConfig.from_pretrained(model_name, num_labels=2)
    torch.manual_seed(0)
    config_ckpt.efficient_softmax = False
    config_ckpt.nested_checkpoint = True
    layer_ckpt = BertLayer(config_ckpt).cuda()

    config_org = AutoConfig.from_pretrained(model_name, num_labels=2)
    torch.manual_seed(0)
    config_org.efficient_softmax = False
    config_org.nested_checkpoint = False
    layer_org = BertLayer(config_org).cuda()

    output_effi, grad_effi = test_implementation(layer_effi)
    output_ckpt, grad_ckpt = test_implementation(layer_ckpt)
    output_org, grad_org = test_implementation(layer_org)

    np.testing.assert_allclose(output_effi, output_org, atol=1e-5)
    np.testing.assert_allclose(grad_effi, grad_org, atol=1e-5)
    np.testing.assert_allclose(output_ckpt, output_org, rtol=1e-7)
    np.testing.assert_allclose(grad_ckpt, grad_org, rtol=1e-7)

def test_atten_quantize_correctness():
    print("===============Quantize Self Attention Layer Correctness==================")
    bz, seq_len, feature = 8, 512, 768
    np.random.seed(0)
    data_np = np.random.randn(bz, seq_len, feature).astype("float32")
    
    def test_implementation(func):
        torch.manual_seed(0)
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        output = func(data)
        output[0].backward(torch.ones_like(output[0]))
        return  [x.detach().cpu().numpy() for x in [output[0], data.grad]]

    model_name = "bert-base-uncased"
    config_org = AutoConfig.from_pretrained(model_name, num_labels=2)
    config_org.efficient_softmax = False
    config_org.nested_checkpoint = False
    torch.manual_seed(0)
    layer_org = BertSelfAttention(config_org).cuda()
    output_org, grad_org = test_implementation(layer_org)

    config_opt = AutoConfig.from_pretrained(model_name, num_labels=2)
    config_opt.efficient_softmax = True
    config_opt.nested_checkpoint = False
    torch.manual_seed(0)
    layer_opt = BertSelfAttention(config_opt).cuda()

    def pack_hook(input):
        return controller.quantize(input)

    def unpack_hook(input):
        return controller.dequantize(input)

    bit = 4
    controller = actnn.controller.Controller(default_bit=bit, swap=False, debug=False, prefetch=False)
    controller.filter_tensors(layer_opt.named_parameters())

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        output_opt, grad_opt = test_implementation(layer_opt)

    np.testing.assert_allclose(output_org, output_opt, atol=1e-1, rtol=1e-2)
    np.testing.assert_allclose(grad_org, grad_opt, atol=1e-1, rtol=1e-2)

def test_bert_layer_quantize_correctness():
    print("===============Quantize Bert Layer Correctness==================")
    bz, seq_len, feature = 8, 512, 768
    np.random.seed(0)
    data_np = np.random.randn(bz, seq_len, feature).astype("float32")
    
    def test_implementation(func):
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        output = func(data)
        output[0].backward(torch.ones_like(output[0]))
        return  [x.detach().cpu().numpy() for x in [output[0], data.grad]]

    model_name = "bert-base-uncased"
    config_opt = AutoConfig.from_pretrained(model_name, num_labels=2)
    config_opt.efficient_softmax = True
    config_opt.nested_checkpoint = False
    torch.manual_seed(0)
    layer_opt = BertLayer(config_opt).cuda()

    config_org = AutoConfig.from_pretrained(model_name, num_labels=2)
    config_opt.efficient_softmax = True
    config_opt.nested_checkpoint = False
    torch.manual_seed(0)
    layer_org = BertLayer(config_org).cuda()

    output_org, grad_org = test_implementation(layer_org)

    def pack_hook(input):
        # return input
        return controller.quantize(input)

    def unpack_hook(input):
        # return input
        return controller.dequantize(input)

    bit = 4
    controller = actnn.controller.Controller(default_bit=bit, swap=False, debug=False, prefetch=False)
    controller.filter_tensors(layer_opt.named_parameters())

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        output_opt, grad_opt = test_implementation(layer_opt)

    # Layernorm is very sensitive to quantization
    # if Layernorm is deleted, the following test should pass
    np.testing.assert_allclose(output_org, output_opt, atol=1e-1, rtol=1e-2)
    np.testing.assert_allclose(grad_org, grad_opt, atol=1e-1, rtol=1e-2)

def test_bert_layer_speed():
    print("========== Bert Layer Speed Test ==========")
    bz, seq_len, feature = 8, 512, 768
    np.random.seed(0)
    data_np = np.random.randn(bz, seq_len, feature).astype("float32")

    def test_implementation(func):
        data = torch.tensor(data_np).to("cuda").requires_grad_()

        stmt = "func(data)"
        t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

        output = func(data)[0]
        head = torch.ones_like(output)
        stmt = "output.backward(head, retain_graph=True)"
        t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

        return t_forward, t_backward

    model_name = "bert-base-uncased"
    config_opt = AutoConfig.from_pretrained(model_name, num_labels=2)
    config_opt.efficient_softmax = True
    config_opt.nested_checkpoint = False
    torch.manual_seed(0)
    layer_opt = BertLayer(config_opt).cuda()
    forward_us, backward_us = test_implementation(layer_opt)

    config_org = AutoConfig.from_pretrained(model_name, num_labels=2)
    config_org.efficient_softmax = False
    config_org.nested_checkpoint = False
    torch.manual_seed(0)
    layer_org = BertLayer(config_org).cuda()
    forward_ref, backward_ref = test_implementation(layer_org)

    print("Ref.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
    print("Memory Optimized. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))

if __name__ == "__main__":
    test_layernorm_quantize()
    test_self_atten_correctness()
    test_bert_layer_correctness()
    test_bert_layer_speed()
    test_atten_quantize_correctness()
    test_bert_layer_quantize_correctness()