/*
 * Cuda operators for quantization 
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
std::pair<Tensor, Tensor> pack_single_precision_cuda(
    Tensor data, Tensor min, Tensor max, int bits, bool stochastic, uint64_t seed);
Tensor unpack_single_precision_cuda(
    Tensor data, int bits, Tensor scale, Tensor min, int64_t N, int64_t num_groups, int64_t group_size);

Tensor act_quantize_dropout_mask_cuda(Tensor data);
Tensor act_dequantize_dropout_mask_cuda(Tensor mask, int N);

Tensor quantize_dropout_mask(Tensor data) {
  CHECK_CUDA_TENSOR_TYPE(data, torch::kUInt8);
  return act_quantize_dropout_mask_cuda(data);
}

Tensor dequantize_dropout_mask(Tensor mask, int N) {
  return act_dequantize_dropout_mask_cuda(mask, N);
}

// Pack/Unpack single precision
Tensor unpack_single_precision(Tensor data,
                               int bits,
                               Tensor scale,
                               Tensor min,
                               int64_t N,
                               int64_t num_groups,
                               int64_t group_size) {
  CHECK_CUDA_TENSOR_DIM_TYPE(data, 1, torch::kInt8);
  CHECK_CUDA_TENSOR_DIM_FLOAT(scale, 3);
  CHECK_CUDA_TENSOR_DIM_FLOAT(min, 3);

  return unpack_single_precision_cuda(data, bits, scale, min,
                                      N, num_groups, group_size);
}

std::pair<Tensor, Tensor> pack_single_precision(Tensor data,
                                                Tensor min,
                                                Tensor max,
                                                int bits,
                                                bool stochastic, int seed) {
  CHECK_CUDA_TENSOR_DIM_FLOAT(data, 3);
  CHECK_CUDA_TENSOR_DIM_FLOAT(min, 3);
  CHECK_CUDA_TENSOR_DIM_FLOAT(max, 3);

  return pack_single_precision_cuda(data, min, max, bits, stochastic, (uint64_t)seed);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unpack_single_precision", &unpack_single_precision);
  m.def("pack_single_precision", &pack_single_precision);
  m.def("act_quantize_dropout_mask", &quantize_dropout_mask);
  m.def("act_dequantize_dropout_mask", &dequantize_dropout_mask);
}
