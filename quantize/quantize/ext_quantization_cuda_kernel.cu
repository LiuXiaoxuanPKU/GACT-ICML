/*
 * Cuda kernels for quantization and mixed-precision packing
 */

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

using torch::IntArrayRef;
using torch::Tensor;

__global__ void compute_scale_kernel(const int32_t* __restrict__ bits,
                                     const float* __restrict__ min,
                                     const float* __restrict__ max,
                                     float* __restrict__ scale,
                                     int N,
                                     int num_groups) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < N * num_groups) {
    scale[id] = ((float)((1 << bits[id / num_groups]) - 1)) / (max[id] - min[id] + 2e-6);
  }
}

// Pack float32 data into int32 bit stream
__global__ void pack_mixed_precision_kernel(const int32_t* __restrict__ bits,
                                            const int32_t* __restrict__ prefix_sum,
                                            const float* __restrict__ data,
                                            const float* __restrict__ scale,
                                            const float* __restrict__ min,
                                            const float* __restrict__ noise,
                                            int32_t* __restrict__ packed,
                                            int N,
                                            int num_groups,
                                            int group_size) {
  extern __shared__ int packed_shared[];

  const int n = blockIdx.y;
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int id = (n * num_groups + group_id) * group_size + d;
  const int shared_len = group_size * bits[n] / 32;

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(packed_shared)[threadIdx.x] = make_int2(0, 0);
  }

  const int val = __float2int_rn(fmax((data[id] - min[n * num_groups + group_id]) * scale[n * num_groups + group_id] + noise[id] - 0.5, 0.0f));
  const int offset = d * bits[n];

  __syncthreads();
  for (int i = 0; i < bits[n]; i++) {
    atomicOr(packed_shared + (offset + i) % shared_len, (1 & (val >> i)) << ((offset + i) / shared_len));
  }
  __syncthreads();

  if (threadIdx.x * 2 < shared_len) {
    const int64_t global_offset = ((int64_t)(n == 0 ? 0 : prefix_sum[n-1]) * num_groups * group_size + bits[n] * group_id * group_size) / 32;
    reinterpret_cast<int2*>(packed)[global_offset/2 + threadIdx.x] = \
                             reinterpret_cast<int2*>(packed_shared)[threadIdx.x];
  }
}

// Pack float32 data into int32 bit stream
std::pair<torch::Tensor, torch::Tensor> pack_mixed_precision_cuda(torch::Tensor data,
                                                                  torch::Tensor min,
                                                                  torch::Tensor max,
                                                                  torch::Tensor bits,
                                                                  bool stochastic) {
  int N = data.size(0);
  int num_groups = data.size(1);
  int group_size = data.size(2);

  torch::Tensor prefix_sum = torch::cumsum(bits, 0, torch::kInt32);
  int64_t total_bits = ((int64_t) prefix_sum[-1].item<int32_t>()) * num_groups * group_size;

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
  torch::Tensor packed = torch::empty({(total_bits + 31) / 32,}, options);
  options = torch::TensorOptions().dtype(torch::kFloat).device(data.device());
  torch::Tensor scale = torch::empty({N, num_groups, 1}, options);

  int threads = 256;
  int blocks = (N * num_groups + threads - 1) / threads;

  compute_scale_kernel<<<blocks, threads>>>(
    bits.data_ptr<int32_t>(), min.data_ptr<float>(), max.data_ptr<float>(),
    scale.data_ptr<float>(), N, num_groups);

  torch::Tensor noise;
  if (stochastic) {
    noise = torch::rand({N, num_groups, group_size}, options);
  } else {
    noise = torch::full({N, num_groups, group_size}, 0.5, options);
  }

  int max_bit = torch::max(bits).item<int32_t>();
  dim3 block_dim(num_groups, N, 1);
  dim3 thread_dim(group_size, 1, 1);
  TORCH_CHECK(group_size % 32 == 0);

  pack_mixed_precision_kernel<<<block_dim, thread_dim, max_bit * group_size * sizeof(int) / 32>>>(
    bits.data_ptr<int32_t>(), prefix_sum.data_ptr<int32_t>(),
    data.data_ptr<float>(),
    scale.data_ptr<float>(), min.data_ptr<float>(),
    noise.data_ptr<float>(),
    packed.data_ptr<int32_t>(),
    N, num_groups, group_size);

  return std::make_pair(packed, scale);
}

// Unpack int32 bit stream to float32 data
__global__ void unpack_mixed_precision_kernel(const int32_t* __restrict__ bits,
                                              const int32_t* __restrict__ prefix_sum,
                                              const int32_t* __restrict__ data,
                                              const float* __restrict__ scale,
                                              const float* __restrict__ min,
                                              float* __restrict__ unpacked,
                                              int N,
                                              int num_groups,
                                              int group_size) {
  const int n = blockIdx.y;
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int id = (n * num_groups + group_id) * group_size + d;
  const int shared_len = group_size * bits[n] / 32;

  const int global_offset = ((n == 0 ? 0 : prefix_sum[n-1]) * num_groups * group_size + bits[n] * group_id * group_size) / 32;
  const int block_offset = d * bits[n];

  int val = 0;
  for (int i = 0; i < bits[n]; i++) {
    val |= (1 & (data[global_offset + (block_offset + i) % shared_len] >> ((block_offset + i) / shared_len))) << i;
  }

  unpacked[id] = ((float)val) / scale[n * num_groups + group_id] + min[n * num_groups + group_id];
}

// Unpack int32 bit stream to float32 data
torch::Tensor unpack_mixed_precision_cuda(torch::Tensor data,
                                          torch::Tensor bits,
                                          torch::Tensor scale,
                                          torch::Tensor min,
                                          int N,
                                          int num_groups,
                                          int group_size) {
  torch::Tensor prefix_sum = torch::cumsum(bits, 0, torch::kInt32);

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(data.device());
  torch::Tensor unpacked = torch::empty({N, num_groups, group_size}, options);

  dim3 block_dim(num_groups, N, 1);
  dim3 thread_dim(group_size, 1, 1);
  TORCH_CHECK(group_size % 32 == 0);

  unpack_mixed_precision_kernel<<<block_dim, thread_dim>>>(
    bits.data_ptr<int32_t>(), prefix_sum.data_ptr<int32_t>(),
    data.data_ptr<int32_t>(), 
    scale.data_ptr<float>(), min.data_ptr<float>(),
    unpacked.data_ptr<float>(),
    N, num_groups, group_size);

  return unpacked;
}


/****************************************/
/********** Act Quantized ReLU **********/
/****************************************/
#define ACT_QUANTIZED_RELU_NUM_THREADS 256
// Unpack int32 bit stream to float32 data
__global__ void act_quantized_relu_forward_kernel(const float* __restrict__ data,
                                                  int32_t* __restrict__ mask,
                                                  float* __restrict__ output,
                                                  int N,
                                                  int mask_len) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_offset = blockIdx.x * blockDim.x / 32;
  const int shared_len = ACT_QUANTIZED_RELU_NUM_THREADS / 32;
  __shared__ int mask_shared[ACT_QUANTIZED_RELU_NUM_THREADS / 32];

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(mask_shared)[threadIdx.x] = make_int2(0, 0);
  }

  if (id < N) {
    bool bit = data[id] > 0;
    output[id] = bit ? data[id] : 0;

    __syncthreads();
    atomicOr(mask_shared + threadIdx.x % shared_len, bit << (threadIdx.x / shared_len));
    __syncthreads();
  }

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(mask)[global_offset / 2 + threadIdx.x] = reinterpret_cast<int2*>(mask_shared)[threadIdx.x];
  }
}

std::pair<torch::Tensor, torch::Tensor> act_quantized_relu_forward_cuda(torch::Tensor data) {
  int n_elements = 1;
  for (size_t i = 0; i < data.dim(); ++i) {
    n_elements *= data.size(i);
  }

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
  int mask_len = (n_elements + 31) / 32;
  torch::Tensor mask = torch::empty({mask_len}, options);
  torch::Tensor output = torch::empty_like(data);

  int threads = ACT_QUANTIZED_RELU_NUM_THREADS;;
  int blocks = (n_elements + threads - 1) / threads;

  act_quantized_relu_forward_kernel<<<blocks, threads>>>(
    data.data_ptr<float>(), mask.data_ptr<int32_t>(), output.data_ptr<float>(),
    n_elements, mask_len);

  return std::make_pair(output, mask);
}

__global__ void act_quantized_relu_backward_kernel(const float* __restrict__ grad_output,
                                                   int32_t* __restrict__ mask,
                                                   float* __restrict__ grad_input,
                                                   int N,
                                                   int mask_len) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_offset = blockIdx.x * blockDim.x / 32;
  const int shared_len = ACT_QUANTIZED_RELU_NUM_THREADS / 32;

  if (id < N) {
    bool bit =  (mask[global_offset + threadIdx.x % shared_len] >> (threadIdx.x / shared_len)) & 1;
    grad_input[id] = bit ? grad_output[id] : 0.0;
  }
}


torch::Tensor act_quantized_relu_backward_cuda(torch::Tensor grad_output, torch::Tensor mask) {
  int n_elements = 1;
  for (size_t i = 0; i < grad_output.dim(); ++i) {
    n_elements *= grad_output.size(i);
  }

  int mask_len = (n_elements + 31) / 32;
  int threads = ACT_QUANTIZED_RELU_NUM_THREADS;
  int blocks = (n_elements + threads - 1) / threads;

  torch::Tensor grad_input = torch::empty_like(grad_output);

  act_quantized_relu_backward_kernel<<<blocks, threads>>>(
    grad_output.data_ptr<float>(), mask.data_ptr<int32_t>(), grad_input.data_ptr<float>(),
    n_elements, mask_len);

  return grad_input;
}


/****************************************/
/******** Act Quantized MaxPool2d *******/
/****************************************/
#define ACT_QUANTIZED_MAX_POOL2D_NUM_THREADS 256
__global__ void act_quantized_max_pool2d_forward_kernel(const float* __restrict__ input,
                                                        float* __restrict__ output,
                                                        int8_t* __restrict__ max_indices,
                                                        int n_elements,
                                                        int N, int C, int H, int W, int H_out, int W_out,
                                                        int KH, int KW, int SH, int SW, int PH, int PW,
                                                        int DH, int DW) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < n_elements) {
    int nc = id / (H_out * W_out);
    int h = id / W_out % H_out;
    int w = id % W_out;

    int h_base = h * SH - PH;
    int h_start = std::max(h_base, 0);
    int h_end = std::min(h_base + KH, H);
    int w_base = w * SW - PW;
    int w_start = std::max(w_base, 0);
    int w_end = std::min(w_base + KW, W);

    float v = -1e10;
    int8_t index;
    for (int i = h_start; i < h_end; i++) {
        for (int j = w_start; j < w_end; j++) {
            if (input[nc * (H * W) + i * W + j] > v) {
                v = input[nc * (H * W) + i * W + j];
                index = (i - h_base) * KW + j - w_base;
            }
        }
    }

    output[id] = v;
    max_indices[id] = index;
  }
}

std::pair<torch::Tensor, torch::Tensor> act_quantized_max_pool2d_forward_cuda(torch::Tensor input,
        IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
        bool ceil_mode, bool return_indices) {
  int N = input.size(0);
  int C = input.size(1);
  int H = input.size(2);
  int W = input.size(3);
  int H_out = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
  int W_out = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
  Tensor output = torch::empty({N, C, H_out, W_out}, options);
  options = torch::TensorOptions().dtype(torch::kInt8).device(input.device());
  Tensor max_indices = torch::empty({N, C, H_out, W_out}, options);

  int threads = ACT_QUANTIZED_MAX_POOL2D_NUM_THREADS;
  int n_elements = N * C * H_out * W_out;
  int blocks = (n_elements + threads - 1) / threads;

  act_quantized_max_pool2d_forward_kernel<<<blocks, threads>>>(
    input.data_ptr<float>(), output.data_ptr<float>(), max_indices.data_ptr<int8_t>(), n_elements,
    N, C, H, W, H_out, W_out, kernel_size[0], kernel_size[1], stride[0], stride[1],
    padding[0], padding[1], dilation[0], dilation[1]);

  return std::make_pair(output, max_indices);
}

__global__ void act_quantized_max_pool2d_backward_kernel(const float* __restrict__ grad_output,
                                                         int8_t* __restrict__ max_indices,
                                                         float* __restrict__ grad_input,
                                                         int n_elements,
                                                         int N, int C, int H, int W, int H_out, int W_out,
                                                         int KH, int KW, int SH, int SW, int PH, int PW,
                                                         int DH, int DW) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < n_elements) {
    int nc = id / (H_out * W_out);
    int h = id / W_out % H_out;
    int w = id % W_out;

    int h_base = h * SH - PH;
    int w_base = w * SW - PW;
    int8_t index = max_indices[id];
    int h_offset = index / KW;
    int w_offset = index % KW;

    atomicAdd(grad_input + (nc * H * W) + (h_base + h_offset) * W + (w_base + w_offset), grad_output[id]);
  }
}

torch::Tensor act_quantized_max_pool2d_backward_cuda(torch::Tensor grad_output, torch::Tensor max_indices,
        IntArrayRef input_shape, 
        IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
        bool ceil_mode, bool return_indices) {
  auto options = torch::TensorOptions().dtype(torch::kFloat).device(grad_output.device());
  Tensor grad_input =  torch::zeros(input_shape, options);

  int N = grad_output.size(0);
  int C = grad_output.size(1);
  int H_out = grad_output.size(2);
  int W_out = grad_output.size(3);
  int H = input_shape[2];
  int W = input_shape[3];

  int threads = ACT_QUANTIZED_MAX_POOL2D_NUM_THREADS;
  int n_elements = N * C * H_out * W_out;
  int blocks = (n_elements + threads - 1) / threads;

  act_quantized_max_pool2d_backward_kernel<<<blocks, threads>>>(
    grad_output.data_ptr<float>(), max_indices.data_ptr<int8_t>(), grad_input.data_ptr<float>(),
    n_elements,
    N, C, H, W, H_out, W_out, kernel_size[0], kernel_size[1], stride[0], stride[1],
    padding[0], padding[1], dilation[0], dilation[1]);

  return grad_input;
}
