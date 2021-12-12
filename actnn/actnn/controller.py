import torch
from actnn.ops import op_quantize
from actnn.ops import op_dequantize
from .utils import compute_tensor_bytes


class Controller:
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
        self.swap_out_stream = torch.cuda.Stream()
        self.swap_in_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.current_stream()
        self.ptr_qtensor_map = {}
        self.prefetch = prefetch
        self.start_prefetch_event = torch.cuda.Event(blocking=True)
        self.end_prefetch_event = torch.cuda.Event(blocking=True)
        self.layer_key_map = {}
        self.tid = 0
        self.start_bwd = True

        # debug purpose
        self.verbose = verbose
        self.unquantize_param_size = 0
        self.unquantize_not_float32_size = 0
        self.unquantize_not_require_grad_size = 0
        self.unquantize_shape_size = 0
        self.quantize_size = 0
        self.quantize_twice_size = 0

    def filter_tensors(self, pairs):
        for k, v in pairs:
            self.unrelated_tensors.add(v.data_ptr())

    def check_quantize(self, input_tensor):
        if input_tensor.dtype != torch.float32:
            if self.verbose:
                self.unquantize_not_float32_size += compute_tensor_bytes([input_tensor])
            return False
        if input_tensor.requires_grad is False:
            if self.verbose:
                self.unquantize_not_require_grad_size += compute_tensor_bytes(
                    [input_tensor]
                )
            return False
        if input_tensor.numel() < 1024: # tensor size < 4 KB
            return False
        if ((len(input_tensor.shape) != 2)
            and (len(input_tensor.shape) != 3)
            and (len(input_tensor.shape) != 4)
        ):
            if self.verbose:
                self.unquantize_shape_size += compute_tensor_bytes([input_tensor])
            return False
        if input_tensor.data_ptr() in self.unrelated_tensors:
            if self.verbose:
                self.unquantize_param_size += compute_tensor_bytes([input_tensor])
            return False
        if self.verbose:
            self.quantize_size += compute_tensor_bytes([input_tensor])
        return True

    def iterate(self):
        del self.ptr_qtensor_map
        del self.layer_key_map
        self.ptr_qtensor_map = {}
        self.layer_key_map = {}
        self.tid = 0
        self.start_bwd = True

        if self.verbose:
            print(
                "Not quantize dtype %f MB"
                % (self.unquantize_not_float32_size / 1024 / 1024)
            )
            print("Not quantize param %f MB" % (self.unquantize_param_size / 1024 / 1024))
            print(
                "Not quantize not require grad %f MB"
                % (self.unquantize_not_require_grad_size / 1024 / 1024)
            )
            print("Not quantize shape %f MB" % (self.unquantize_shape_size / 1024 / 1024))
            print("Quantize size %f MB" % (self.quantize_size / 1024 / 1024))

    def generate_tensor_key(self, t, tid):
        if not t.is_contiguous():
            return (tid)
        # sample 20 elements data pointer as the key
        sample_cnt = 20
        step = max(torch.numel(t) // sample_cnt, 1)
        ptrs = [t.data_ptr()]
        for i in range(min(sample_cnt, torch.numel(t))):
            idx = i * step
            ptrs.append(t.view(-1)[idx].item())
        return tuple(ptrs)

    def quantize(self, input):
        if not self.check_quantize(input):
            return False, input

        if self.debug:
            ret = op_quantize(input, self.default_bit)
            return True, ret, input.shape

        tid = self.tid
        input_shape = input.shape
        key = self.generate_tensor_key(input, tid)
        self.layer_key_map[tid] = key
        self.tid += 1
        skip_quantize = key in self.ptr_qtensor_map
        if not skip_quantize:
            # quantize
            q_inputs = op_quantize(input, self.default_bit)
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
            self.quantize_twice_size += 1
            print("Same tensor", key, input.shape, self.quantize_twice_size)
            # increase the ref count
            self.ptr_qtensor_map[key][1] += 1
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
        q_inputs, ref_cnt, key_tid = self.ptr_qtensor_map[key]

        if self.start_bwd and self.verbose:
            store_size = 0
            total_eles = 0
            for k in self.ptr_qtensor_map:
                total_eles += self.ptr_qtensor_map[k][0][0].numel() \
                            + self.ptr_qtensor_map[k][0][2].numel() \
                            + self.ptr_qtensor_map[k][0][3].numel() 
                store_size += (
                    compute_tensor_bytes(
                        [
                            self.ptr_qtensor_map[k][0][0],
                            self.ptr_qtensor_map[k][0][2],
                            self.ptr_qtensor_map[k][0][3],
                        ]
                    )
                    + 4
                    + 12
                )
            for k in self.layer_key_map:
                store_size += 4 + 12
            print("Store size %d MB, # of elems %d" % (store_size / 1024 / 1024, total_eles))
            print(
                "Quantize twice size %d" % (self.quantize_twice_size)
            )
            self.start_bwd = False
        if self.start_bwd and self.swap:
            self.compute_stream.wait_stream(self.swap_out_stream)
            self.start_bwd = False

        # swap waits until prefetch finishes
        if self.prefetch and self.swap:
            self.end_prefetch_event.wait(self.compute_stream)

        if not q_inputs[0].is_cuda:
            q_inputs[0] = q_inputs[0].cuda(non_blocking=False)

        # event: start_prefetch
        self.start_prefetch_event.record()

        # prefetch previous layer
        if self.prefetch and self.swap:
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

        ret = op_dequantize(q_inputs, input_shape)

        ref_cnt -= 1
        if ref_cnt < 0:
            print("[Error] Ref count < 0", key, ref_cnt)
            exit(0)
        elif ref_cnt == 0:
            del self.ptr_qtensor_map[key]
        else:
            self.ptr_qtensor_map[key] = [q_inputs, ref_cnt, key_tid]
        return ret
