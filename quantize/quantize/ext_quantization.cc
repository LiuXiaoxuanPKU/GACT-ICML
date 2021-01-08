/*
 * Cuda operators for quantization and mixed-precision packing
 */

#include <torch/extension.h>
#include <torch/torch.h>

#include "ext_common.h"

using namespace torch::autograd;
using torch::Tensor;
using torch::IntArrayRef;

// Declarations for functions in ext_quantization_cuda_kernel.cu
std::pair<torch::Tensor, torch::Tensor> pack_mixed_precision_cuda(
          torch::Tensor data, torch::Tensor min,
          torch::Tensor max, torch::Tensor bits, bool stochastic);
torch::Tensor unpack_mixed_precision_cuda(torch::Tensor data, torch::Tensor bits,
                                          torch::Tensor scale, torch::Tensor min,
                                          int N, int num_groups, int group_size);

std::pair<torch::Tensor, torch::Tensor> act_quantized_relu_forward_cuda(torch::Tensor data);
torch::Tensor act_quantized_relu_backward_cuda(torch::Tensor grad_output, torch::Tensor mask);

std::pair<torch::Tensor, torch::Tensor> act_quantized_max_pool2d_forward_cuda(torch::Tensor input,
        IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
        bool ceil_mode, bool return_indices);
torch::Tensor act_quantized_max_pool2d_backward_cuda(torch::Tensor grad_output, torch::Tensor max_indices,
        IntArrayRef input_shape, 
        IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
        bool ceil_mode, bool return_indices);

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
    torch::Tensor output, mask;
    std::tie(output, mask) = act_quantized_relu_forward_cuda(input);
    ctx->save_for_backward({mask});
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    return {act_quantized_relu_backward_cuda(grad_outputs[0], saved[0])};
  }
};

torch::Tensor act_quantized_relu(torch::Tensor input) {
  CHECK_CUDA_TENSOR_TYPE(input, torch::kFloat32);
  return ActQuantizedReLU::apply(input);
}


// Activation quantized max_pool2d: use compressed bit stream to store activation
class ActQuantizedMaxPool2d : public Function<ActQuantizedMaxPool2d> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input, IntArrayRef kernel_size,
          IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, bool return_indices) {
    TORCH_CHECK(kernel_size.size() == 2);
    TORCH_CHECK(stride.size() == 2);
    TORCH_CHECK(padding.size() == 2);
    TORCH_CHECK(dilation.size() == 2);
    TORCH_CHECK(ceil_mode == false);
    TORCH_CHECK(return_indices == false);
    TORCH_CHECK(kernel_size[0] * kernel_size[1] < 16);

    torch::Tensor output, max_indices; 
    std::tie(output, max_indices) = act_quantized_max_pool2d_forward_cuda(input, kernel_size, stride, padding,
            dilation, ceil_mode, return_indices);
    ctx->save_for_backward({max_indices});
    ctx->saved_data["input_shape"] = input.sizes();
    ctx->saved_data["kernel_size"] = kernel_size;
    ctx->saved_data["stride"] = stride;
    ctx->saved_data["padding"] = padding;
    ctx->saved_data["dilation"] = dilation;
    ctx->saved_data["ceil_mode"] = ceil_mode;
    ctx->saved_data["return_indices"] = return_indices;
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    return {act_quantized_max_pool2d_backward_cuda(
                grad_outputs[0], saved[0],
                IntArrayRef(ctx->saved_data["input_shape"].toIntVector()),
                IntArrayRef(ctx->saved_data["kernel_size"].toIntVector()),
                IntArrayRef(ctx->saved_data["stride"].toIntVector()),
                IntArrayRef(ctx->saved_data["padding"].toIntVector()),
                IntArrayRef(ctx->saved_data["dilation"].toIntVector()),
                ctx->saved_data["ceil_mode"].toBool(),ctx->saved_data["return_indices"].toBool()),
            Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
  }
};

torch::Tensor act_quantized_max_pool2d(torch::Tensor input, IntArrayRef kernel_size,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, bool return_indices) {
  CHECK_CUDA_TENSOR_TYPE(input, torch::kFloat32);
  return ActQuantizedMaxPool2d::apply(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_mixed_precision", &pack_mixed_precision);
  m.def("unpack_mixed_precision", &unpack_mixed_precision);
  m.def("act_quantized_relu", &act_quantized_relu);
  m.def("act_quantized_max_pool2d", &act_quantized_max_pool2d);
}

