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

    def __init__(self, default_bit=4, auto_prec=True, single_quantize=True, verbose=False, use_error=True, swap=False):
        self.unrelated_tensors = set()
        self.verbose = verbose
        self.single_quantize = single_quantize
        self.tensor_versions = {}
        # tensors already quantized and referenced by other tensors
        # key: tensor_id, value: tensor data
        self.quantized_tensors = {}
        # tensors that do not need to be quantized
        # the quantized results can be looked up in the self.quantized_tensors
        # key: tensor_id, value: reference_tensor_id
        self.not_quantize_tensors = {}

        self.tensor_id = 0  # tensor_id starts from 0
        self.init_iter = True
        self.errors = []
        self.all_tensors = {}

        self.auto_prec = auto_prec
        self.ap = None
        self.default_bit = default_bit
        self.groups = []
        self.dims = []
        self.id2group = {}
        self.grad_fns = {}
        self.gcnt = 0
        self.use_error = use_error

        self.swap = swap
        self.swap_out_stream = torch.cuda.Stream()
        self.swap_in_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.current_stream()

        self.iter_cnt = 0
        self.quantize_layer = -1


    def filter_tensors(self, pairs):
        for k, v in pairs:
            self.unrelated_tensors.add(v.data_ptr())

    def iterate(self, model):
        self.iter_cnt += 1
        # if self.verbose and self.iter_cnt > 2:
        #     exit(0)

        if self.init_iter and self.auto_prec:
            dims = torch.tensor(self.dims, dtype=torch.long)
            group2id = {}
            group_id = 0
            for layer_id in self.grad_fns:
                grad_fn = self.grad_fns[layer_id]
                if grad_fn not in group2id:
                    group2id[grad_fn] = group_id
                    group_id += 1

            groups = []
            for i in range(self.tensor_id):
                groups.append(group2id[self.grad_fns[i]])

            if self.auto_prec:
                self.ap = AutoPrecision(
                    self.default_bit, groups, dims, warmup_iters=10)

        if (not self.init_iter) and self.auto_prec:
            grad = []
            for param in model.parameters():
                if param.grad is not None:
                    param_grad = param.grad.detach().ravel()
                    grad.append(param_grad)
            grad = torch.cat(grad, 0)
            if self.use_error:
                delats = torch.tensor(self.errors)
            else:
                delats = 2**(-2.0 * self.ap.bits)
            gsizes = torch.ones_like(delats)
            self.ap.iterate(grad, delats, gsizes)
            print(self.ap.bits)

        if self.init_iter:
            self.init_iter = False

        self.tensor_id = 0
        self.errors = []
        self.quantized_tensors = {}
        self.all_tensors = {}

    def check_quantize(self, input_tensor):
        if input_tensor.dtype != torch.float32:
            return False
        if input_tensor.requires_grad is False:
            return False
        if (len(input_tensor.shape) != 3) and (len(input_tensor.shape) != 4):
            return False
        if input_tensor.data_ptr() in self.unrelated_tensors:
            return False
        if input_tensor.data_ptr() in self.quantized_tensors:
            return False
        return True

    def quantize(self, input):
        if not self.check_quantize(input):
            return False, input

        cur_tensor_id = self.tensor_id

        # if cur_tensor_id != self.quantize_layer:
        #     return False, input

        if self.init_iter:
            # use grad_fn to categorize groups
            # grad_fn indicates the operator of previous layer
            self.grad_fns[cur_tensor_id] = type(input.grad_fn).__name__
            dim = input.numel() // input.shape[0]
            self.dims.append(dim)

            if input.data_ptr() in self.tensor_versions and \
                    input._version == self.tensor_versions[input.data_ptr()][1]:
                self.not_quantize_tensors[cur_tensor_id] = \
                    self.tensor_versions[input.data_ptr()][0]
            else:
                self.tensor_versions[input.data_ptr()] = (
                    cur_tensor_id, input._version)
       
        # Get quantize bit
        if not self.init_iter and self.auto_prec:
            q_bit = self.ap.bits[cur_tensor_id]
            q_bit = (q_bit.item() + 1) / 2 * 2
        else:
            q_bit = self.default_bit

        if self.use_error or self.verbose:
            self.all_tensors[cur_tensor_id] = input

        # Get quantized tensor
        if not self.init_iter and self.single_quantize and cur_tensor_id in self.not_quantize_tensors:
            if self.verbose:
                print("Not quantize %d, Reference tensor %d" %
                          (cur_tensor_id, self.not_quantize_tensors[cur_tensor_id]))
            q_input = self.quantized_tensors[self.not_quantize_tensors[cur_tensor_id]]
        else:
            q_input = op_quantize(input, q_bit)
            if self.swap:
                q_input = self.swap_to_cpu(q_input)

        if not self.init_iter and self.single_quantize and cur_tensor_id in self.not_quantize_tensors.values():
            self.quantized_tensors[cur_tensor_id] = q_input

        self.tensor_id += 1
        return (True, q_input, input.shape, cur_tensor_id)

    def dequantize(self, input):
        quantized = input[0]
        if not quantized:
            return input[1]

        _, q_input, input_shape, cur_tensor_id = input
        
        if self.swap:
            q_input = self.swap_to_gpu(q_input)

        # if cur_tensor_id != self.quantize_layer:
        #     return input[1]

        r = op_dequantize(q_input, input_shape)
        if self.verbose or self.use_error:
            diff_tensor = self.all_tensors[cur_tensor_id] - r
            diff_ratio = (diff_tensor**2).sum() / \
                (self.all_tensors[cur_tensor_id]**2).sum()
            if self.verbose:
                print("layer = %d, shape %s, diff ratio = %.10f, %s" %
                      (cur_tensor_id, input_shape, diff_ratio.item(),
                       type(self.all_tensors[cur_tensor_id].grad_fn).__name__))
            self.errors.insert(0, (diff_tensor**2).sum())
        return r


    def swap_to_cpu(self, q_input):
        tensor = q_input[0]
        # fwd_values[tid] = tensor.to(float).mean()
        self.swap_out_stream.wait_stream(self.compute_stream)
        with torch.cuda.stream(self.swap_out_stream):
            tensor.record_stream(self.swap_out_stream) # tell default stream to keep tensor until swap_stream finishes swapping
            tensor_cpu = torch.empty(tensor.shape, dtype=tensor.dtype, device='cpu', pin_memory=True)
            tensor_cpu.copy_(tensor, non_blocking = True)
        q_input[0] = tensor_cpu
        return q_input


    def swap_to_gpu(self, q_input):
        q_input[0] = q_input[0].cuda()
        return q_input
