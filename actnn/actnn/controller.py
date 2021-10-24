import torch
from actnn.ops import op_quantize
from actnn.ops import op_dequantize
from actnn.autoprec import AutoPrecision


class Controller:
    def __init__(self, default_bit=4, check_error=True, single_quantize=False):
        self.unrelated_tensors = set()

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
        self.check_error = check_error
        self.all_tensors = {}

        self.ap = None
        self.default_bit = default_bit
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
        self.quantized_tensors = {}

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
        # Get quantize bit
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

            if input.data_ptr() in self.tensor_versions and \
                    input._version == self.tensor_versions[input.data_ptr()][1]:
                self.not_quantize_tensors[cur_tensor_id] = \
                    self.tensor_versions[input.data_ptr()][0]
            else:
                self.tensor_versions[input.data_ptr()] = (
                    cur_tensor_id, input._version)
        else:
            q_bit = self.ap.bits[self.tensor_id]
            print("Layer = %d, bit = %d" % (self.tensor_id, q_bit), flush=True)
            q_bit = max(2, q_bit.item())

        if self.check_error:
            self.all_tensors[self.tensor_id] = input

        # Get quantized tensor
        if not self.init_iter and self.single_quantize:
            if cur_tensor_id in self.not_quantize_tensors:
                print("Not quantize %d, Reference tensor %d" %
                      (cur_tensor_id, self.not_quantize_tensors[cur_tensor_id]))
                q_input = self.quantized_tensors[self.not_quantize_tensors[cur_tensor_id]]
            else:
                q_input = op_quantize(input, q_bit)
                if cur_tensor_id in self.not_quantize_tensors.values():
                    self.quantized_tensors[cur_tensor_id] = q_input
        else:
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
