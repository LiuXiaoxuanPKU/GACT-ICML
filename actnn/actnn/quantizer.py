import torch
from actnn.ops import op_quantize, op_dequantize
from actnn.utils import random_sample, uniform_sample


class Quantizer:
    """
    default_bit: the number of bits used to quantize
    swap: if turned on, swap activation memory to CPU
    prefetch: if turned on, activation of the previous layer will be prefetched. the parameter is meaningful only when swap is True
    debug: if turned on, the same tensor that is saved twice will be quantized twice, which introduces memory overhead
    verbose: print debug log
    """

    def __init__(self, default_bit=4, swap=False, debug=False, prefetch=False, verbose=False):
        self.unrelated_tensors = set()
        self.default_bit = default_bit
        self.debug = debug

        self.swap = swap
        if swap:
            self.swap_out_stream = torch.cuda.Stream()
            self.swap_in_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.current_stream()
        self.ptr_qtensor_map = {}
        self.prefetch = prefetch
        if prefetch:
            self.start_prefetch_event = torch.cuda.Event(blocking=True)
            self.end_prefetch_event = torch.cuda.Event(blocking=True)
        self.layer_key_map = {}
        self.tid = 0
        self.start_bwd = True

        # data collected for auto precision
        self.inject_noises = []
        self.seeds = []
        self.bits = []
        self.dims = []

        self.iter = 0
        self.bwd_fns = []

    def filter_tensors(self, pairs):
        for k, v in pairs:
            self.unrelated_tensors.add(v.data_ptr())

    '''
    return should_be_quantized, is_dropout_mask
    '''

    def check_quantize(self, input_tensor):
        if input_tensor.numel() > 0 and input_tensor.dtype == torch.uint8:
            if (input_tensor.max() == 1) and (input_tensor.min() == 0):
                return True, True
            return False, False
        if input_tensor.dtype != torch.float32:
            return False, False
        if input_tensor.requires_grad is False:
            return False, False
        if ((len(input_tensor.shape) != 2)
            and (len(input_tensor.shape) != 3)
            and (len(input_tensor.shape) != 4)
            ):
            return False, False
        if input_tensor.data_ptr() in self.unrelated_tensors:
            return False, False
        return True, False

    def __del__(self):
        del self.ptr_qtensor_map
        del self.layer_key_map
        del self.unrelated_tensors

    def iterate(self):
        del self.ptr_qtensor_map
        del self.layer_key_map
        self.ptr_qtensor_map = {}
        self.layer_key_map = {}
        self.tid = 0
        self.start_bwd = True
        self.iter += 1
        # print("bits")
        # print(self.bits)

    def generate_tensor_key(self, t, tid):
        return tid
        if not t.is_contiguous():
            return (tid)
        # sample 100 elements data pointer as the key
        sample_cnt = min(100, t.numel())
        key = uniform_sample(t, sample_cnt, add_dataptr=True)
        key.append(t.sum().item())
        return tuple(key)

    def quantize(self, input):
        quantize, is_dropout_mask = self.check_quantize(input)

        if not quantize:
            return False, input

        if self.debug:
            # debug does not quantize dropout mask
            if is_dropout_mask:
                return False, input
            ret = op_quantize(input, self.default_bit, 0)
            return True, ret, input.shape

        # special case: use 1 bit to quantize dropout mask
        if is_dropout_mask:
            input = input.to(torch.float32)
            q_inputs = op_quantize(input, 1, 0)
            return True, is_dropout_mask, q_inputs, input.shape

        if self.iter == 0:
            self.dims.append(input.numel())
            self.bits.append(int(self.default_bit))
            self.inject_noises.append(False)
            self.seeds.append(self.tid)
            print('Tensor ', self.tid, input.shape, input.grad_fn, input.dtype)

        tid = self.tid
        self.tid += 1
        input_shape = input.shape

        bit = self.bits[tid]
        key = self.generate_tensor_key(input, tid)
        self.layer_key_map[tid] = key
        skip_quantize = key in self.ptr_qtensor_map

        self.bwd_fns.append(str(type(input.grad_fn)))
        if not skip_quantize:
            # quantize
            # use tid as quantize seed
            q_inputs = op_quantize(input, bit, self.seeds[tid])
            if self.swap:
                with torch.cuda.stream(self.swap_out_stream):
                    self.swap_out_stream.wait_stream(self.compute_stream)
                    q_input_cpu = torch.empty(
                        q_inputs[0].shape,
                        dtype=q_inputs[0].dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    q_input_cpu.copy_(q_inputs[0], non_blocking=True)
                    q_input_gpu = q_inputs[0]
                    del q_input_gpu
                    q_inputs[0] = q_input_cpu
            self.ptr_qtensor_map[key] = [q_inputs, 1, tid]
        else:
            # print("===============Skip Quantize", tid)
            # increase the ref count
            self.ptr_qtensor_map[key][1] += 1
        return True, is_dropout_mask, key, input_shape, tid

    def dequantize(self, input):
        quantized = input[0]
        if not quantized:
            return input[1]

        if self.debug:
            _, q_inputs, input_shape = input
            ret = op_dequantize(q_inputs, input_shape, False)
            return ret

        is_dropout_mask = input[1]
        if is_dropout_mask:
            _, is_dropout_mask, q_inputs, input_shape = input
            ret = op_dequantize(q_inputs, input_shape, False)
            ret = ret.to(torch.uint8)
            return ret

        _, _, key, input_shape, tid = input
        q_inputs, ref_cnt, key_tid = self.ptr_qtensor_map[key]

        if self.start_bwd and self.swap:
            self.compute_stream.wait_stream(self.swap_out_stream)
            self.start_bwd = False

        # compute waits until prefetch finishes
        if self.prefetch and self.swap:
            self.end_prefetch_event.wait(self.compute_stream)

        if not q_inputs[0].is_cuda:
            q_inputs[0] = q_inputs[0].cuda(non_blocking=False)

        # prefetch previous layer
        if self.prefetch and self.swap:
            # event: start_prefetch
            self.start_prefetch_event.record()
            with torch.cuda.stream(self.swap_in_stream):
                if tid > 0:
                    self.start_prefetch_event.wait(self.swap_in_stream)
                    previous_key = self.layer_key_map[tid - 1]
                    q_previous_inputs, _, _ = self.ptr_qtensor_map[previous_key]
                    if not q_previous_inputs[0].is_cuda:
                        q_previous_inputs[0] = q_previous_inputs[0].cuda(
                            non_blocking=True
                        )
                    self.end_prefetch_event.record()

        ret = op_dequantize(q_inputs, input_shape, self.inject_noises[tid])

        ref_cnt -= 1
        if ref_cnt < 0:
            print("[Error] Ref count < 0", key, ref_cnt)
            exit(0)
        elif ref_cnt == 0:
            del self.ptr_qtensor_map[key]
        else:
            self.ptr_qtensor_map[key] = [q_inputs, ref_cnt, key_tid]
        return ret
