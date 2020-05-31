#include <torch/extension.h>

/*
enum Reduce { MIN, MAX };
template<Reduce r> torch::Tensor minimax_cuda(torch::Tensor x);
*/
torch::Tensor min_cuda(torch::Tensor x);
torch::Tensor max_cuda(torch::Tensor x);

torch::Tensor min(torch::Tensor x) {
  TORCH_CHECK(x.type().is_cuda(), "x must be a CUDA tensor!");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous!");
  TORCH_CHECK(x.dim() == 2, "x must be 2D!");
  // return minimax_cuda<MIN>(x);
  return min_cuda(x);
}

torch::Tensor max(torch::Tensor x) {
  TORCH_CHECK(x.type().is_cuda(), "x must be a CUDA tensor!");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous!");
  TORCH_CHECK(x.dim() == 2, "x must be 2D!");
  // return minimax_cuda<MAX>(x);
  return max_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("min", &min, "min (CUDA)");
  m.def("max", &max, "max (CUDA)");
}
