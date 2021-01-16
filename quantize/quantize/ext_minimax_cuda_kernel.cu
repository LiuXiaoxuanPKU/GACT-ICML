#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

using torch::Tensor;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif

__global__ void minimax_cuda_kernel(const float* __restrict__ data,
                                    float* __restrict__ min,
                                    float* __restrict__ max,
                                    int N,
                                    int D) {
  float max_val, min_val;
  max_val = -1e30;
  min_val = 1e30;

  for (int k1_outer = 0; k1_outer < D / 32; ++k1_outer) {
    max_val = std::max(max_val, data[blockIdx.x * D + k1_outer * 32 + threadIdx.x]);
    min_val = std::min(min_val, data[blockIdx.x * D + k1_outer * 32 + threadIdx.x]);
  }

  unsigned int mask;
  float max_val_t, min_val_t;
  mask = __activemask();

  max_val_t = __shfl_down_sync(mask, max_val, 16, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 8, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 4, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 2, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 1, 32);
  max_val = std::max(max_val, max_val_t);
  max_val = __shfl_sync(mask, max_val, 0, 32);
  max[blockIdx.x] = max_val;

  min_val_t = __shfl_down_sync(mask, min_val, 16, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 8, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 4, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 2, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 1, 32);
  min_val = std::min(min_val, min_val_t);
  min_val = __shfl_sync(mask, min_val, 0, 32);
  min[blockIdx.x] = min_val;
}


std::pair<Tensor, Tensor> minimax_cuda(torch::Tensor data) {
  int N = data.size(0);
  int D = data.size(1);

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(data.device());
  Tensor min = torch::empty({N,}, options);
  Tensor max = torch::empty({N,}, options);

  int blocks = N;
  int threads = 32;

  minimax_cuda_kernel<<<blocks, threads>>>(
    data.data_ptr<float>(), min.data_ptr<float>(), max.data_ptr<float>(),
    N, D);

  return std::make_pair(min, max);
}
