/*
 * Cuda operators for quantization and mixed-precision packing
 */

#include <torch/extension.h>
#include <torch/torch.h>

#include "ext_common.h"

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;
using torch::Tensor;
using torch::IntArrayRef;

// Declarations for functions in ext_quantization_cuda_kernel.cu
// Pack and unpack
Tensor unpack_single_precision_cuda(
    Tensor data, int bits, Tensor scale, Tensor min, int64_t N, int64_t num_groups, int group_size);
tensor_list minimax_quantize_single_precision_cuda(Tensor data, int bits, uint64_t seed);

// ActQuantizedReLU
std::pair<Tensor, Tensor> act_quantized_relu_forward_cuda(Tensor data);
Tensor act_quantized_relu_backward_cuda(Tensor grad_output, Tensor mask);


// Pack/Unpack single precision
Tensor unpack_single_precision(Tensor data,
                               int bits,
                               Tensor scale,
                               Tensor min,
                               int64_t N,
                               int64_t num_groups,
                               int group_size) {
  CHECK_CUDA_TENSOR_DIM_TYPE(data, 1, torch::kInt8);
  CHECK_CUDA_TENSOR_DIM_FLOAT(scale, 3);
  CHECK_CUDA_TENSOR_DIM_FLOAT(min, 3);

  return unpack_single_precision_cuda(data, bits, scale, min,
                                      N, num_groups, group_size);
}

tensor_list minimax_quantize_single_precision(Tensor data, int bits, int seed) {
  // CHECK_CUDA_TENSOR_DIM_FLOAT(data, 2);
  // return minimax_quantize_single_precision_cuda(data, bits);
  CHECK_CUDA_TENSOR_DIM_FLOAT(data, 3);
  return minimax_quantize_single_precision_cuda(data, bits, (uint64_t)seed);
}

// Activation quantized relu: use compressed bit stream to store activation
class ActQuantizedReLU : public Function<ActQuantizedReLU> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input) {
    Tensor output, mask;
    std::tie(output, mask) = act_quantized_relu_forward_cuda(input);
    ctx->save_for_backward({mask});
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    return {act_quantized_relu_backward_cuda(grad_outputs[0], saved[0])};
  }
};

Tensor act_quantized_relu(Tensor input) {
  CHECK_CUDA_TENSOR_FLOAT(input);
  return ActQuantizedReLU::apply(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unpack_single_precision", &unpack_single_precision);
  m.def("minimax_quantize_single_precision", &minimax_quantize_single_precision);
  m.def("act_quantized_relu", &act_quantized_relu);
}
