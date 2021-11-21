import time
from functools import total_ordering
import torch
from actnn.ops import op_quantize
from actnn.ops import op_dequantize
from actnn.autoprec import AutoPrecision


class Controller:
    '''
    default_bit: if auto_prec = False, default_bit is the fix number of quantize bits
                 if auto_prec = True, default_bit is the average number of quantize bits
    auto_prec: if auto_prec = True, auto precision is turned on.
    single_quantize: if single_quantize = True, tensors that share the same storage will only be quantized once
    verbose: print debug log
    '''

    def __init__(self, default_bit=4, verbose=False, swap=False, debug=False, prefetch=False):
        self.unrelated_tensors = set()
        self.verbose = verbose
        self.default_bit = default_bit
        self.debug = debug

        self.swap = swap
        self.swap_out_stream = torch.cuda.Stream()
        self.swap_in_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.current_stream()
        self.ptr_qtensor_map = {}
        self.prefetch = prefetch
        self.layer_key_map = {}
        self.tid = 0

        self.start_bwd = True

    def filter_tensors(self, pairs):
        for k, v in pairs:
            self.unrelated_tensors.add(v.data_ptr())

    def check_quantize(self, input_tensor):
        if input_tensor.dtype != torch.float32:
            return False
        if input_tensor.requires_grad is False:
            return False
        if (len(input_tensor.shape) != 3) and (len(input_tensor.shape) != 4):
            return False
        if input_tensor.data_ptr() in self.unrelated_tensors:
            return False
        return True

    def iterate(self):
        del self.ptr_qtensor_map
        self.ptr_qtensor_map = {}
        self.tid = 0

    def quantize(self, input):
        if not self.check_quantize(input):
            return False, input

        if self.debug:
            ret = op_quantize(input, self.default_bit)
            return True, ret, input.shape

        tid = self.tid
        input_shape = input.shape
        key = (input.data_ptr() + input.sum(), input._version)
        self.layer_key_map[tid] = key
        self.tid += 1
        if key not in self.ptr_qtensor_map:
            # quantize
            q_inputs = op_quantize(input, self.default_bit)
            if self.swap:
                q_input_cpu = torch.empty(
                    q_inputs[0].shape, dtype=q_inputs[0].dtype, device='cpu', pin_memory=True)
                q_input_cpu.copy_(q_inputs[0], non_blocking=True)
                q_input_gpu = q_inputs[0]
                del q_input_gpu
                q_inputs[0] = q_input_cpu
            self.ptr_qtensor_map[key] = q_inputs

        return True, key, input_shape, tid

    def dequantize(self, input):
        quantized = input[0]
        if not quantized:
            return input[1]

        if self.debug:
            _, q_inputs, input_shape = input
            ret = op_dequantize(q_inputs, input_shape)
            return ret

        _, key, input_shape, tid = input
        q_inputs = self.ptr_qtensor_map[key]
        if not q_inputs[0].is_cuda:
            self.ptr_qtensor_map[key][0] = q_inputs[0].cuda(non_blocking=True)

        # prefetch previous layer
        with torch.cuda.stream(self.swap_in_stream):
            if tid > 0 and self.prefetch:
                previous_key = self.layer_key_map[tid-1]
                q_previous_inputs = self.ptr_qtensor_map[previous_key]
                if not q_previous_inputs[0].is_cuda:
                    self.ptr_qtensor_map[previous_key][0] = q_previous_inputs[0].cuda(
                        non_blocking=True)

        ret = op_dequantize(q_inputs, input_shape)
        return ret
