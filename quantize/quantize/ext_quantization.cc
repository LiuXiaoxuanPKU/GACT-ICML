/*
 * Cuda operators for quantization and mixed-precision packing
 */

#include <torch/extension.h>
#include <torch/torch.h>

using namespace torch::autograd;

// Declarations for functions in ext_quantization_cuda_kernel.cu
std::pair<torch::Tensor, torch::Tensor> pack_mixed_precision_cuda(
        torch::Tensor data, torch::Tensor min,
        torch::Tensor max, torch::Tensor bits, bool stochastic);
torch::Tensor unpack_mixed_precision_cuda(torch::Tensor data, torch::Tensor bits,
                                          torch::Tensor scale, torch::Tensor min,
                                          int N, int num_groups, int group_size);
std::pair<torch::Tensor, torch::Tensor> act_quantized_relu_forward_cuda(torch::Tensor data);
torch::Tensor act_quantized_relu_backward_cuda(torch::Tensor mask, torch::Tensor data);

// Helper for type check
#define CHECK_CUDA_TENSOR_DIM_TYPE(name, n_dim, type)                             \
  TORCH_CHECK(name.device().is_cuda(), #name " must be a CUDA tensor!");          \
  TORCH_CHECK(name.is_contiguous(), #name " must be contiguous!");                \
  TORCH_CHECK(name.dim() == n_dim, "The dimension of " #name " is not correct!"); \
  TORCH_CHECK(name.dtype() == type, "The type of " #name " is not correct!");     \

// Helper for type check
#define CHECK_CUDA_TENSOR_TYPE(name, type)                                        \
  TORCH_CHECK(name.device().is_cuda(), #name " must be a CUDA tensor!");          \
  TORCH_CHECK(name.is_contiguous(), #name " must be contiguous!");                \
  TORCH_CHECK(name.dtype() == type, "The type of " #name " is not correct!");     \


std::pair<torch::Tensor, torch::Tensor> pack_mixed_precision(torch::Tensor data,
                                                             torch::Tensor min,
                                                             torch::Tensor max,
                                                             torch::Tensor bits,
                                                             bool stochastic) {
  CHECK_CUDA_TENSOR_DIM_TYPE(data, 3, torch::kFloat32);
  CHECK_CUDA_TENSOR_DIM_TYPE(min, 3, torch::kFloat32);
  CHECK_CUDA_TENSOR_DIM_TYPE(max, 3, torch::kFloat32);
  CHECK_CUDA_TENSOR_DIM_TYPE(bits, 1, torch::kInt32);

  return pack_mixed_precision_cuda(data, min, max, bits, stochastic);
}

torch::Tensor unpack_mixed_precision(torch::Tensor data,
                                     torch::Tensor bits,
                                     torch::Tensor scale,
                                     torch::Tensor min,
                                     int N,
                                     int num_groups,
                                     int group_size) {
  CHECK_CUDA_TENSOR_DIM_TYPE(data, 1, torch::kInt32);
  CHECK_CUDA_TENSOR_DIM_TYPE(bits, 1, torch::kInt32);
  CHECK_CUDA_TENSOR_DIM_TYPE(scale, 3, torch::kFloat32);
  CHECK_CUDA_TENSOR_DIM_TYPE(min, 3, torch::kFloat32);

  return unpack_mixed_precision_cuda(data, bits, scale, min,
                                     N, num_groups, group_size);
}

// Activation quantized relu: use compressed bit stream to store activation
class ActQuantizedReLU : public Function<ActQuantizedReLU> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input) {
    torch::Tensor mask, output; 
    std::tie(output, mask) = act_quantized_relu_forward_cuda(input);
    ctx->save_for_backward({mask});
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    return {act_quantized_relu_backward_cuda(saved[0], grad_outputs[0])};
  }
};

torch::Tensor act_quantized_relu(torch::Tensor input) {
  CHECK_CUDA_TENSOR_TYPE(input, torch::kFloat32);
  return ActQuantizedReLU::apply(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_mixed_precision", &pack_mixed_precision);
  m.def("unpack_mixed_precision", &unpack_mixed_precision);
  m.def("act_quantized_relu", &act_quantized_relu);
}

