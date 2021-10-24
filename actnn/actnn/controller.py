import torch
from actnn.ops import op_quantize
from actnn.ops import op_dequantize
from actnn.autoprec import AutoPrecision


class Controller:
    def __init__(self, check_error=False):
        self.unrelated_tensors = set()
        self.quantized_tensors = {}
        self.tensor_id = 0  # tensor_id starts from 0
        self.init_iter = True
        self.check_error = check_error
        self.all_tensors = {}

        self.ap = None
        self.default_bit = 4
        self.groups = []
        self.dims = []
        self.id2group = {}
        self.gcnt = 0

    def filter_tensors(self, pairs):
        for k, v in pairs:
            self.unrelated_tensors.add(v.data_ptr())

    def iterate(self, model):
        if self.check_error and not self.init_iter:
            exit(0)

        if self.init_iter:
            dims = torch.tensor(self.dims, dtype=torch.long)
            self.ap = AutoPrecision(self.default_bit, self.groups, dims)
            self.init_iter = False
        else:
            grad = []
            for param in model.parameters():
                if param.grad is not None:
                    grad.append(param.grad.detach().ravel())
            grad = torch.cat(grad, 0)
            self.ap.iterate(grad)
        self.tensor_id = 0

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

        if self.init_iter:
            rank = len(input.shape)
            if rank not in self.id2group:
                self.gcnt += 1
                self.id2group[rank] = self.gcnt
            self.groups.append(self.id2group[rank])
            # TODO: check the correctness of dim
            dim = input.numel() // input.shape[0]
            self.dims.append(dim)
            q_bit = self.default_bit
        else:
            q_bit = self.ap.bits[self.tensor_id]
            print("Layer = %d, bit = %d" % (self.tensor_id, q_bit), flush=True)
            q_bit = 8

        cur_tensor_id = self.tensor_id
        if self.check_error:
            self.all_tensors[self.tensor_id] = input

        q_input = op_quantize(input, q_bit)
        self.tensor_id += 1

        return (True, q_input, input.shape, cur_tensor_id)

    # TODO: handle swap
    def dequantize(self, input):
        quantized = input[0]
        if not quantized:
            return input[1]

        _, q_input, input_shape, cur_tensor_id = input
        r = op_dequantize(q_input, input_shape)
        if self.check_error:
            diff_tensor = self.all_tensors[cur_tensor_id] - r
            diff_ratio = (diff_tensor**2).sum() / \
                (self.all_tensors[cur_tensor_id]**2).sum()
            print("layer = %d, shape %s, diff ratio = %.10f" %
                  (cur_tensor_id, input_shape, diff_ratio.item()))
        return r
