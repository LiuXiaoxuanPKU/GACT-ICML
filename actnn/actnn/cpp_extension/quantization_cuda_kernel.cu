/*
 * Cuda kernels for quantization and mixed-precision packing
 */

#include <torch/extension.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

#define BLOCK_Y_DIM_MAX ((((int64_t)(1)) << 16) - 1)

using torch::IntArrayRef;
using torch::Tensor;
using torch::autograd::tensor_list;

/****************************************/
/***** Pack/Unpack Single Precision *****/
/****************************************/
template<typename scalar_t, bool boundary_check>
__global__ void fuse_single_precision_kernel (int32_t bits,
                                             const scalar_t* __restrict__ data,
                                             int8_t* __restrict__ packed,
                                             scalar_t* __restrict__ scale,
                                             scalar_t* __restrict__ min,
                                             std::pair<uint64_t, uint64_t> seeds,
                                             int64_t N,
                                             int64_t num_groups,
                                             int64_t group_size,
                                             int64_t block_idx_y_base) {
  const int64_t no = blockIdx.y + block_idx_y_base;
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int work_per_thread = 8 / bits;
  const int64_t global_thread_id = (no * num_groups + group_id) * group_size + d;

  __shared__ scalar_t min_red[8 * 256];
  __shared__ scalar_t tmp_scale[9];
  
  for (int ni = 0; ni < work_per_thread; ni++) {
    const int64_t n = no * work_per_thread + ni;

    if (boundary_check && n >= N) { break; }

    const int64_t id = (n * num_groups + group_id) * group_size + d;

    scalar_t cur_val = data[id];
    int idx = ni * 256 + d;
    min_red[idx] = cur_val;
  }

  __syncthreads();

  // for (int s = group_size / 2; s >= 32; s>>=1) {
  //   if (d < s) {
  //     for (int ni = 0; ni < work_per_thread; ni++) {
  //       int idx = ni * 256 + d;
  //       min_red[idx] = std::min(min_red[idx], min_red[idx + s]);
  //       max_red[idx] = std::max(max_red[idx], max_red[idx + s]);
  //     }
  //   }
  //   __syncthreads();
  // }


  for (int ni = 0; ni < work_per_thread; ni++) {
    scalar_t max_val = -1e30;
    scalar_t min_val = 1e30;

    for (int64_t k1_outer = 0; k1_outer < group_size / 32; ++k1_outer) {
      if (d + k1_outer * 32 < 256) {
        max_val = std::max(max_val, min_red[ni * 256 + d + k1_outer * 32]);
        min_val = std::min(min_val, min_red[ni * 256 + d + k1_outer * 32]);
      }
    }

    scalar_t max_val_t, min_val_t;
    unsigned int mask = __activemask();

    max_val_t = __shfl_down_sync(mask, (float)max_val, 16, 32);
    max_val = std::max(max_val, max_val_t);
    max_val_t = __shfl_down_sync(mask, (float)max_val, 8, 32);
    max_val = std::max(max_val, max_val_t);
    max_val_t = __shfl_down_sync(mask, (float)max_val, 4, 32);
    max_val = std::max(max_val, max_val_t);
    max_val_t = __shfl_down_sync(mask, (float)max_val, 2, 32);
    max_val = std::max(max_val, max_val_t);
    max_val_t = __shfl_down_sync(mask, (float)max_val, 1, 32);
    max_val = std::max(max_val, max_val_t);
    max_val = __shfl_sync(mask, (float)max_val, 0, 32);

    min_val_t = __shfl_down_sync(mask, (float)min_val, 16, 32);
    min_val = std::min(min_val, min_val_t);
    min_val_t = __shfl_down_sync(mask, (float)min_val, 8, 32);
    min_val = std::min(min_val, min_val_t);
    min_val_t = __shfl_down_sync(mask, (float)min_val, 4, 32);
    min_val = std::min(min_val, min_val_t);
    min_val_t = __shfl_down_sync(mask, (float)min_val, 2, 32);
    min_val = std::min(min_val, min_val_t);
    min_val_t = __shfl_down_sync(mask, (float)min_val, 1, 32);
    min_val = std::min(min_val, min_val_t);
    min_val = __shfl_sync(mask, (float)min_val, 0, 32);

    if (d == 0) {
      tmp_scale[ni] = scalar_t((1 << bits) - 1) / (max_val -  min_val + 2e-6);
      min_red[ni * 256] = min_val;
      const int64_t n = no * work_per_thread + ni;
      if (boundary_check && n >= N) { break; }
      min[n * num_groups + group_id] = min_val;
      scale[n * num_groups + group_id] = tmp_scale[ni];
    }
  }
  __syncthreads();
  
  // if (d == 0) {
  //   for (int ni = 0; ni < work_per_thread; ni++) {
  //     const int64_t n = no * work_per_thread + ni;
  //     if (boundary_check && n >= N) { break; }
  //     scalar_t group_min = min_red[ni * 256];
  //     scalar_t group_max= max_red[ni * 256];
  //     tmp_scale[ni] = scalar_t((1 << bits) - 1) /
  //                       (group_max -  group_min + 2e-6);
  //     min[n * num_groups + group_id] = group_min;
  //     scale[n * num_groups + group_id] = tmp_scale[ni];
  //   }
  // }
  // __syncthreads();

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, global_thread_id, seeds.second, &state);

  uint8_t local_packed = 0;
  for (int ni = 0; ni < work_per_thread; ni++) {
    const int64_t n = no * work_per_thread + ni;

    if (boundary_check && n >= N) { break; }
    const float noise = curand_uniform(&state);
    const int64_t id = (int64_t)(n * num_groups + group_id) * group_size + d;
    scalar_t quantized = fmax((data[id] - min_red[ni * 256]) * tmp_scale[ni] + noise - 0.5, 0.0f);
    const int32_t val = __float2int_rn(quantized);
    local_packed |= (val << (ni * bits));
  }
  packed[global_thread_id] = local_packed;
}

template<typename scalar_t, bool boundary_check>
__global__ void fuse_single_precision_half_kernel (int32_t bits,
                                             const scalar_t* __restrict__ data,
                                             int8_t* __restrict__ packed,
                                             scalar_t* __restrict__ scale,
                                             scalar_t* __restrict__ min,
                                             std::pair<uint64_t, uint64_t> seeds,
                                             int64_t N,
                                             int64_t num_groups,
                                             int64_t group_size,
                                             int64_t block_idx_y_base) {
  const int64_t no = blockIdx.y + block_idx_y_base;
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int work_per_thread = 8 / bits;
  const int64_t global_thread_id = (int64_t)(no * num_groups + group_id) * group_size + d;

  __shared__ half min_red[4 * 256];
  __shared__ half max_red[4 * 256];
  __shared__ half group_data[4 * 256];
  __shared__ half tmp_scale[5];
  
  for (int ni = 0; ni < work_per_thread; ni++) {
    const int64_t n = no * work_per_thread + ni;

    if (boundary_check && n >= N) { break; }

    const int64_t id = (int64_t)(n * num_groups + group_id) * group_size + d;

    half cur_val = __float2half(data[id]);
    int idx = ni * 256 + d;
    min_red[idx] = cur_val;
    max_red[idx] = cur_val;
    group_data[idx] = cur_val;
  }

  __syncthreads();

  for (int s = group_size / 2; s>0; s>>=1) {
    if (d < s) {
      for (int ni = 0; ni < work_per_thread; ni++) {
        int idx = ni * 256 + d;
        int ids = idx + s;
        min_red[idx] = __hle(min_red[idx], min_red[ids]) ? min_red[idx] : min_red[ids];
        max_red[idx] = __hgt(max_red[idx], max_red[ids]) ? max_red[idx] : max_red[ids];
      }
    }
    __syncthreads();
  }

  if (d == 0) {
    for (int ni = 0; ni < work_per_thread; ni++) {
      const int64_t n = no * work_per_thread + ni;
      if (boundary_check && n >= N) { break; }
      half group_min = min_red[ni * 256];
      half group_max= max_red[ni * 256];
      tmp_scale[ni] = __hdiv(
                        (__int2half_rn((1 << bits) - 1)),
                        (__hadd(__hsub(group_max, group_min), __float2half(5e-3))));
      min[n * num_groups + group_id] = __half2float(group_min);
      scale[n * num_groups + group_id] = __half2float(tmp_scale[ni]);
    }
  }
  __syncthreads();

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, global_thread_id, seeds.second, &state);

  uint8_t local_packed = 0;
  for (int ni = 0; ni < work_per_thread; ni++) {
    const int64_t n = no * work_per_thread + ni;

    if (boundary_check && n >= N) { break; }

    const float noise = curand_uniform(&state);
    half quantized = __hsub(
                      __hadd(
                        __hmul(
                          __hsub(group_data[ni * 256 + d],
                                  min_red[ni * 256]), 
                          tmp_scale[ni]),
                        __float2half(noise)),
                      __float2half (0.5));
    quantized = __hgt(quantized, __float2half (0.0)) ? quantized : __float2half (0.0);
    const int32_t val = __half2int_rn(quantized);
    local_packed |= (val << (ni * bits));
  }
  packed[global_thread_id] = local_packed;
}

tensor_list minimax_quantize_single_precision_cuda(Tensor data, int bits) {
  int64_t N = data.size(0);
  int64_t num_groups = data.size(1);
  int group_size = data.size(2);

  const int work_per_thread = 8 / bits;
  TORCH_CHECK(8 % bits == 0);

  int64_t N_round = N + (work_per_thread - N % work_per_thread) % work_per_thread;
  int64_t total_bits = ((int64_t)bits) * (N_round * num_groups * group_size);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(data.device());
  Tensor packed = torch::zeros({(total_bits + 8) / 8,}, options);

  auto options_minimax = torch::TensorOptions().dtype(data.scalar_type()).device(data.device());
  Tensor min = torch::empty({N, num_groups, 1}, options_minimax);
  Tensor scale = torch::empty({N, num_groups, 1}, options_minimax);

  // Random number generator
  auto gen = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    int threads = 256;
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(threads * work_per_thread);
  };

  // Pack
  int64_t logical_block_y_dim = (N + work_per_thread - 1) / work_per_thread;
  for (int64_t block_idx_y_base = 0; block_idx_y_base < logical_block_y_dim; block_idx_y_base += BLOCK_Y_DIM_MAX) {
    dim3 block_dim(num_groups, std::min(logical_block_y_dim - block_idx_y_base, BLOCK_Y_DIM_MAX), 1);
    dim3 thread_dim(group_size, 1, 1);

    if (N % work_per_thread == 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "minimax_quantize_single_precision", ([&] {
      fuse_single_precision_kernel<scalar_t, false><<<block_dim, thread_dim>>>(
        bits,
        data.data_ptr<scalar_t>(),
        packed.data_ptr<int8_t>(),
        scale.data_ptr<scalar_t>(),
        min.data_ptr<scalar_t>(),
        rng_engine_inputs,
        N, num_groups, group_size, block_idx_y_base
      );
    }));
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "minimax_quantize_single_precision", ([&] {
        fuse_single_precision_kernel<scalar_t, true><<<block_dim, thread_dim>>>(
          bits,
          data.data_ptr<scalar_t>(), packed.data_ptr<int8_t>(),
          scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
          rng_engine_inputs,
          N, num_groups, group_size, block_idx_y_base);
      }));
    }
  }

  tensor_list ret;
  ret.push_back(packed);
  ret.push_back(scale);
  ret.push_back(min);
  return ret;
}

// Unpack int32 bit stream to float16/32 data
template<typename scalar_t, bool boundary_check>
__global__ void unpack_single_precision_kernel(int32_t bits,
                                               const int8_t* __restrict__ data,
                                               const scalar_t* __restrict__ scale,
                                               const scalar_t* __restrict__ min,
                                               scalar_t* __restrict__ unpacked,
                                               int64_t N,
                                               int64_t num_groups,
                                               int group_size,
                                               int64_t block_idx_y_base) {
  const int64_t no = blockIdx.y + block_idx_y_base;
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int64_t global_thread_id = (no * num_groups + group_id) * group_size + d;

  int work_per_thread = 8 / bits;

  uint8_t local_packed = data[global_thread_id];
  int mask = ((1 << bits) - 1);
  for (int ni = 0; ni < work_per_thread; ni++) {
    const int64_t n = no * work_per_thread + ni;

    if (boundary_check && n >= N) { break; }

    const int val = (local_packed >> (ni * bits)) & mask;
    const int64_t id = (int64_t)(n * num_groups + group_id) * group_size + d;
    unpacked[id] = ((scalar_t)val) / scale[n * num_groups + group_id] + min[n * num_groups + group_id];
    // unpacked[id] = ((scalar_t)val) * scale[n * num_groups + group_id] + min[n * num_groups + group_id];
  }
}

// Unpack int32 bit stream to float16/32 data
Tensor unpack_single_precision_cuda(Tensor data,
                                    int bits,
                                    Tensor scale,
                                    Tensor min,
                                    int64_t N,
                                    int64_t num_groups,
                                    int group_size) {
  auto options = torch::TensorOptions().dtype(scale.dtype()).device(data.device());
  Tensor unpacked = torch::empty({N, num_groups, group_size}, options);

  int work_per_thread = 8 / bits;
  TORCH_CHECK(8 % bits == 0);

  int64_t logical_block_y_dim = (N + work_per_thread - 1) / work_per_thread;
  for (int64_t block_idx_y_base = 0; block_idx_y_base < logical_block_y_dim; block_idx_y_base += BLOCK_Y_DIM_MAX) {
    dim3 block_dim(num_groups, std::min(logical_block_y_dim - block_idx_y_base, BLOCK_Y_DIM_MAX), 1);
    dim3 thread_dim(group_size, 1, 1);

    if (N % work_per_thread == 0) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(scale.scalar_type(), "unpack_single_precision", ([&] {
        unpack_single_precision_kernel<scalar_t, false><<<block_dim, thread_dim>>>(
          bits,
          data.data_ptr<int8_t>(),
          scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
          unpacked.data_ptr<scalar_t>(),
          N, num_groups, group_size, block_idx_y_base);
      }));
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(scale.scalar_type(), "unpack_single_precision", ([&] {
        unpack_single_precision_kernel<scalar_t, true><<<block_dim, thread_dim>>>(
          bits,
          data.data_ptr<int8_t>(),
          scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
          unpacked.data_ptr<scalar_t>(),
          N, num_groups, group_size, block_idx_y_base);
      }));
    }
  }
  return unpacked;
}

/****************************************/
/******** Act Quantized Dropout Mask ****/
/****************************************/
#define ACT_QUANTIZED_DROPOUT_MASK_THREADS 512
// Unpack int32 bit stream to float16/32 data
template <typename scalar_t>
__global__ void act_quantize_dropout_mask_kernel(const scalar_t* __restrict__ data,
                                                  int32_t* __restrict__ mask,
                                                  int N,
                                                  int mask_len) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_offset = blockIdx.x * blockDim.x / (sizeof(int32_t) * 8);
  const int shared_len = ACT_QUANTIZED_DROPOUT_MASK_THREADS / (sizeof(int32_t) * 8);
  __shared__ int mask_shared[ACT_QUANTIZED_DROPOUT_MASK_THREADS / (sizeof(int32_t) * 8)];

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(mask_shared)[threadIdx.x] = make_int2(0, 0);
  }

  if (id < N) {
    bool bit = data[id];
    __syncthreads();
    atomicOr(mask_shared + threadIdx.x % shared_len, bit << (threadIdx.x / shared_len));
    __syncthreads();
  }

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(mask)[global_offset / 2 + threadIdx.x] = reinterpret_cast<int2*>(mask_shared)[threadIdx.x];
  }
}

Tensor act_quantize_dropout_mask_cuda(Tensor data) {
  int n_elements = 1;
  for (size_t i = 0; i < data.dim(); ++i) {
    n_elements *= data.size(i);
  }

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
  int mask_len = (n_elements + sizeof(int32_t) * 8 - 1) / (sizeof(int32_t) * 8);
  Tensor mask = torch::empty({mask_len}, options);

  int threads = ACT_QUANTIZED_DROPOUT_MASK_THREADS;
  int blocks = (n_elements + threads - 1) / threads;

  AT_DISPATCH_INTEGRAL_TYPES(data.scalar_type(), "act_quantize_dropout_mask", ([&] {
    act_quantize_dropout_mask_kernel<scalar_t><<<blocks, threads>>>(
      data.data_ptr<scalar_t>(), mask.data_ptr<int32_t>(),
      n_elements, mask_len);
  }));

  return mask;
}

template <typename scalar_t>
__global__ void act_dequantize_dropout_mask_kernel(int32_t* __restrict__ mask,
                                                   scalar_t* __restrict__ output,
                                                   int N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_offset = blockIdx.x * blockDim.x / (sizeof(int32_t) * 8);
  const int shared_len = ACT_QUANTIZED_DROPOUT_MASK_THREADS / (sizeof(int32_t) * 8);

  if (id < N) {
    scalar_t bit =  (mask[global_offset + threadIdx.x % shared_len] >> (threadIdx.x / shared_len)) & 1;
    output[id] = bit;
  }
}


Tensor act_dequantize_dropout_mask_cuda(Tensor mask, int N) {
  int threads = ACT_QUANTIZED_DROPOUT_MASK_THREADS;
  int blocks = (N + threads - 1) / threads;
  auto options = torch::TensorOptions().dtype(torch::kUInt8).device(mask.device());
  Tensor output = torch::zeros({N,}, options);

  AT_DISPATCH_INTEGRAL_TYPES(torch::kUInt8, "act_dequantize_dropout_mask", ([&] {
      act_dequantize_dropout_mask_kernel<scalar_t><<<blocks, threads>>>(
        mask.data_ptr<int32_t>(), output.data_ptr<scalar_t>(), N);
  }));

  return output;
}


/****************************************/
/********** Act Quantized ReLU **********/
/****************************************/
#define ACT_QUANTIZED_RELU_NUM_THREADS 512
// Unpack int32 bit stream to float16/32 data
template <typename scalar_t>
__global__ void act_quantized_relu_forward_kernel(const scalar_t* __restrict__ data,
                                                  int32_t* __restrict__ mask,
                                                  scalar_t* __restrict__ output,
                                                  int N,
                                                  int mask_len) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_offset = blockIdx.x * blockDim.x / (sizeof(int32_t) * 8);
  const int shared_len = ACT_QUANTIZED_RELU_NUM_THREADS / (sizeof(int32_t) * 8);
  __shared__ int mask_shared[ACT_QUANTIZED_RELU_NUM_THREADS / (sizeof(int32_t) * 8)];

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(mask_shared)[threadIdx.x] = make_int2(0, 0);
  }

  if (id < N) {
    bool bit = data[id] > 0;
    if (bit) {
      output[id] = data[id];
    } else {
      output[id] = 0.0;
    }

    __syncthreads();
    atomicOr(mask_shared + threadIdx.x % shared_len, bit << (threadIdx.x / shared_len));
    __syncthreads();
  }

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(mask)[global_offset / 2 + threadIdx.x] = reinterpret_cast<int2*>(mask_shared)[threadIdx.x];
  }
}

std::pair<Tensor, Tensor> act_quantized_relu_forward_cuda(Tensor data) {
  int n_elements = 1;
  for (size_t i = 0; i < data.dim(); ++i) {
    n_elements *= data.size(i);
  }

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
  int mask_len = (n_elements + sizeof(int32_t) * 8 - 1) / (sizeof(int32_t) * 8);
  Tensor mask = torch::empty({mask_len}, options);
  Tensor output = torch::empty_like(data);

  int threads = ACT_QUANTIZED_RELU_NUM_THREADS;
  int blocks = (n_elements + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "act_quantized_relu_forward", ([&] {
    act_quantized_relu_forward_kernel<scalar_t><<<blocks, threads>>>(
      data.data_ptr<scalar_t>(), mask.data_ptr<int32_t>(), output.data_ptr<scalar_t>(),
      n_elements, mask_len);
  }));

  return std::make_pair(output, mask);
}

template <typename scalar_t>
__global__ void act_quantized_relu_backward_kernel(const scalar_t* __restrict__ grad_output,
                                                   int32_t* __restrict__ mask,
                                                   scalar_t* __restrict__ grad_input,
                                                   int N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_offset = blockIdx.x * blockDim.x / (sizeof(int32_t) * 8);
  const int shared_len = ACT_QUANTIZED_RELU_NUM_THREADS / (sizeof(int32_t) * 8);

  if (id < N) {
    bool bit =  (mask[global_offset + threadIdx.x % shared_len] >> (threadIdx.x / shared_len)) & 1;
    if (bit) {
      grad_input[id] = grad_output[id];
    } else {
      grad_input[id] = 0.0;
    }
  }
}


Tensor act_quantized_relu_backward_cuda(Tensor grad_output, Tensor mask) {
  int n_elements = 1;
  for (size_t i = 0; i < grad_output.dim(); ++i) {
    n_elements *= grad_output.size(i);
  }

  int threads = ACT_QUANTIZED_RELU_NUM_THREADS;
  int blocks = (n_elements + threads - 1) / threads;

  Tensor grad_input = torch::empty_like(grad_output);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "act_quantized_relu_backward", ([&] {
      act_quantized_relu_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.data_ptr<scalar_t>(), mask.data_ptr<int32_t>(), grad_input.data_ptr<scalar_t>(),
        n_elements);
  }));

  return grad_input;
}