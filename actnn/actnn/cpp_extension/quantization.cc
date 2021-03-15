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
std::pair<Tensor, Tensor> pack_mixed_precision_cuda(
    Tensor data, Tensor min, Tensor max, Tensor bits, bool stochastic);
Tensor unpack_mixed_precision_cuda(
    Tensor data, Tensor bits, Tensor scale, Tensor min, int N, int num_groups, int group_size);
std::pair<Tensor, Tensor> pack_single_precision_cuda(
    Tensor data, Tensor min, Tensor max, int bits, bool stochastic);
Tensor unpack_single_precision_cuda(
    Tensor data, int bits, Tensor scale, Tensor min, int N, int num_groups, int group_size);

// ActQuantizedReLU
std::pair<Tensor, Tensor> act_quantized_relu_forward_cuda(Tensor data);
Tensor act_quantized_relu_backward_cuda(Tensor grad_output, Tensor mask);

// ActQuantizedMaxPool2d
std::pair<Tensor, Tensor> act_quantized_max_pool2d_forward_cuda(Tensor input,
        IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
        bool ceil_mode, bool return_indices);
Tensor act_quantized_max_pool2d_backward_cuda(Tensor grad_output, Tensor max_indices,
        IntArrayRef input_shape, 
        IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
        bool ceil_mode, bool return_indices);


// Pack/Unpack mixed precision
std::pair<Tensor, Tensor> pack_mixed_precision(Tensor data,
                                               Tensor min,
                                               Tensor max,
                                               Tensor bits,
                                               bool stochastic) {
  CHECK_CUDA_TENSOR_DIM_TYPE(data, 3, torch::kFloat32);
  CHECK_CUDA_TENSOR_DIM_TYPE(min, 3, torch::kFloat32);
  CHECK_CUDA_TENSOR_DIM_TYPE(max, 3, torch::kFloat32);
  CHECK_CUDA_TENSOR_DIM_TYPE(bits, 1, torch::kInt32);

  return pack_mixed_precision_cuda(data, min, max, bits, stochastic);
}

Tensor unpack_mixed_precision(Tensor data,
                              Tensor bits,
                              Tensor scale,
                              Tensor min,
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


// Pack/Unpack single precision
std::pair<Tensor, Tensor> pack_single_precision(Tensor data,
                                                Tensor min,
                                                Tensor max,
                                                int bits,
                                                bool stochastic) {
  CHECK_CUDA_TENSOR_DIM_TYPE(data, 3, torch::kFloat32);
  CHECK_CUDA_TENSOR_DIM_TYPE(min, 3, torch::kFloat32);
  CHECK_CUDA_TENSOR_DIM_TYPE(max, 3, torch::kFloat32);

  return pack_single_precision_cuda(data, min, max, bits, stochastic);
}

Tensor unpack_single_precision(Tensor data,
                               int bits,
                               Tensor scale,
                               Tensor min,
                               int N,
                               int num_groups,
                               int group_size) {
  CHECK_CUDA_TENSOR_DIM_TYPE(data, 1, torch::kInt8);
  CHECK_CUDA_TENSOR_DIM_TYPE(scale, 3, torch::kFloat32);
  CHECK_CUDA_TENSOR_DIM_TYPE(min, 3, torch::kFloat32);

  return unpack_single_precision_cuda(data, bits, scale, min,
                                      N, num_groups, group_size);
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
  CHECK_CUDA_TENSOR_TYPE(input, torch::kFloat32);
  return ActQuantizedReLU::apply(input);
}


// Activation quantized max_pool2d: use compressed bit stream to store activation
class ActQuantizedMaxPool2d : public Function<ActQuantizedMaxPool2d> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input, IntArrayRef kernel_size,
        IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, bool return_indices) {
    TORCH_CHECK(kernel_size.size() == 2);
    TORCH_CHECK(stride.size() == 2);
    TORCH_CHECK(padding.size() == 2);
    TORCH_CHECK(dilation.size() == 2);
    TORCH_CHECK(ceil_mode == false);
    TORCH_CHECK(return_indices == false);
    TORCH_CHECK(kernel_size[0] * kernel_size[1] < 16);

    Tensor output, max_indices;
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

Tensor act_quantized_max_pool2d(Tensor input, IntArrayRef kernel_size,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, bool return_indices) {
  CHECK_CUDA_TENSOR_TYPE(input, torch::kFloat32);
  return ActQuantizedMaxPool2d::apply(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_mixed_precision", &pack_mixed_precision);
  m.def("unpack_mixed_precision", &unpack_mixed_precision);
  m.def("pack_single_precision", &pack_single_precision);
  m.def("unpack_single_precision", &unpack_single_precision);
  m.def("act_quantized_relu", &act_quantized_relu);
  m.def("act_quantized_max_pool2d", &act_quantized_max_pool2d);
}
