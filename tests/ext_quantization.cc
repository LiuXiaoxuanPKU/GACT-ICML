/*
 * Cuda operators for quantization and mixed-precision packing
 */

#include <torch/extension.h>

std::pair<torch::Tensor, torch::Tensor> 
pack_mixed_precision_cuda(torch::Tensor data, torch::Tensor min,
                          torch::Tensor max, torch::Tensor bits);
torch::Tensor unpack_mixed_precision_cuda(torch::Tensor data, torch::Tensor bits,
                                          torch::Tensor scale, torch::Tensor min,
                                          int N, int D);

#define CHECK_CUDA_TENSOR(name, n_dim, type)                                      \
  TORCH_CHECK(name.device().is_cuda(), #name " must be a CUDA tensor!");          \
  TORCH_CHECK(name.is_contiguous(), #name " must be contiguous!");                \
  TORCH_CHECK(name.dim() == n_dim, "The dimension of " #name " is not correct!"); \
  TORCH_CHECK(name.dtype() == type, "The type of " #name " is not correct!");     \

std::pair<torch::Tensor, torch::Tensor> pack_mixed_precision(torch::Tensor data,
                                                             torch::Tensor min,
                                                             torch::Tensor max,
                                                             torch::Tensor bits) {
  CHECK_CUDA_TENSOR(data, 2, torch::kFloat32);
  CHECK_CUDA_TENSOR(min, 1, torch::kFloat32);
  CHECK_CUDA_TENSOR(max, 1, torch::kFloat32);
  CHECK_CUDA_TENSOR(bits, 1, torch::kInt32);

  return pack_mixed_precision_cuda(data, min, max, bits);
}

torch::Tensor unpack_mixed_precision(torch::Tensor data,
                                     torch::Tensor bits,
                                     torch::Tensor scale,
                                     torch::Tensor min,
                                     int N,
                                     int D) {
  CHECK_CUDA_TENSOR(data, 1, torch::kInt32);
  CHECK_CUDA_TENSOR(bits, 1, torch::kInt32);
  CHECK_CUDA_TENSOR(scale, 1, torch::kFloat32);
  CHECK_CUDA_TENSOR(min, 1, torch::kFloat32);

  return unpack_mixed_precision_cuda(data, bits, scale, min, N, D);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_mixed_precision", &pack_mixed_precision);
  m.def("unpack_mixed_precision", &unpack_mixed_precision);
}

