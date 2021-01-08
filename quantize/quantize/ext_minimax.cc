#include <torch/extension.h>

#include "ext_common.h"

torch::Tensor min_cuda(torch::Tensor x);
torch::Tensor max_cuda(torch::Tensor x);

torch::Tensor min(torch::Tensor x) {
  CHECK_CUDA_TENSOR_TYPE(x, torch::kFloat32);
 
  return min_cuda(x);
}

torch::Tensor max(torch::Tensor x) {
  CHECK_CUDA_TENSOR_TYPE(x, torch::kFloat32);

  return max_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("min", &min, "min (CUDA)");
  m.def("max", &max, "max (CUDA)");
}
