import torch
from gact.conf import config
from gact.ops import op_quantize, op_dequantize, op_quantize_mask, op_dequantize_mask
from gact.utils import uniform_sample, compute_tensor_bytes


class Quantizer:
    """
    default_bit: the number of bits used to quantize
    swap: if turned on, swap activation memory to CPU
    prefetch: if turned on, activation of the previous layer will be prefetched. the parameter is meaningful only when swap is True
    """

    def __init__(self, default_bit, swap, prefetch):
        self.unrelated_tensors = set()  # record the tensors that should not be quantized
        self.default_bit = default_bit

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
        self.seeds = {}
        self.bits = {}
        self.dims = {}

        self.iter = 0  # total number of iterations, including the extra inter for auto precision
        # iteration for seed, share the same seed_iter for the same auto precision adaptive step
        self.seed_iter = 0

    def filter_tensors(self, pairs):
        for _, v in pairs:
            self.unrelated_tensors.add(v.data_ptr())

    # return should_be_quantized, is_dropout_mask
    # treat dropout mask differently because it can be quantized with 1 bit with a specialized kernel
    def check_quantize(self, input_tensor):
        # does not quantize parameters
        if input_tensor.data_ptr() in self.unrelated_tensors:
            return False, False
        # special check for saved mask
        if input_tensor.numel() > 0 and input_tensor.dtype == torch.uint8:
            if (input_tensor.max() == 1) and (input_tensor.min() == 0):
                return True, True
            return False, False
        # only quantize float16 and float32
        if input_tensor.dtype not in [torch.float32, torch.float16]:
            return False, False
        # only quantize activation that requires gradient
        # for example: BN statistics (running mean/var) should not be quantized
        if input_tensor.requires_grad is False:
            return False, False
        # only quantize 2/3/4D tensors for now
        if ((len(input_tensor.shape) != 2)
            and (len(input_tensor.shape) != 3)
            and (len(input_tensor.shape) != 4)
            ):
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

    def generate_tensor_key(self, t, tid):
        if config.check_dup:
            # sample 100 elements data pointer + tensor.sum() as the key
            sample_cnt = min(100, t.numel())
            key = uniform_sample(t, sample_cnt, add_dataptr=True)
            key.append(t.sum().item())
            return tuple(key)
        else:
            return (tid)

    def quantize(self, input):
        quantize, is_dropout_mask = self.check_quantize(input)

        if not quantize:
            return False, input

        # special case: use 1 bit to quantize dropout mask
        if is_dropout_mask:
            q_inputs = op_quantize_mask(input)
            return True, is_dropout_mask, q_inputs

        tid = self.tid
        self.tid += 1
        input_shape = input.shape

        key = self.generate_tensor_key(input, tid)
        self.layer_key_map[tid] = key
        skip_quantize = key in self.ptr_qtensor_map

        if not skip_quantize:
            if self.iter == 0:
                bit = self.default_bit
                self.bits[tid] = bit
                self.dims[tid] = input.numel()
                self.seeds[tid] = tid
            else:
                bit = self.bits[tid]
            # quantize
            q_inputs = op_quantize(
                input, bit, self.seeds[tid] + self.seed_iter)
            if self.swap:
                #  with torch.cuda.stream(self.swap_out_stream):
                # self.swap_out_stream.wait_stream(self.compute_stream)
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
            # increase the ref count
            self.ptr_qtensor_map[key][1] += 1
        return True, is_dropout_mask, key, input_shape, tid

    def dequantize(self, input):
        quantized = input[0]
        if not quantized:
            return input[1]

        is_dropout_mask = input[1]
        if is_dropout_mask:
            _, is_dropout_mask, q_inputs = input
            ret = op_dequantize_mask(q_inputs)
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
                    if previous_key in self.ptr_qtensor_map:
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
            exit(-1)
        elif ref_cnt == 0:
            del self.ptr_qtensor_map[key]
        else:
            self.ptr_qtensor_map[key] = [q_inputs, ref_cnt, key_tid]
        return ret
