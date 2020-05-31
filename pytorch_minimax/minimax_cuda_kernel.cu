#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <torch/extension.h>
#include "minimax.h"

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <class T, Reduce r> __device__ __forceinline__ T id() {
  switch (r) {
  case MIN:
    return std::numeric_limits<T>::max();
  case MAX:
    return std::numeric_limits<T>::lowest();
  }
}

template <class T, Reduce r> __device__ __forceinline__ T op(T lhs, T rhs) {
  switch (r) {
  case MIN:
    return lhs < rhs ? lhs : rhs;
  case MAX:
    return lhs < rhs ? rhs : lhs;
  default:
    return 0;
  }
}

template <class T, unsigned int blockSize, bool nIsPow2, Reduce r>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int j = blockIdx.x;
  unsigned int i = blockIdx.y * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.y;

  T mySum = id<T, r>();

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    // mySum += g_idata[i];
    mySum = op<T, r>(mySum, g_idata[j * n + i]);

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    // if (nIsPow2 || i + blockSize < n) mySum += g_idata[i + blockSize];
    if (nIsPow2 || i + blockSize < n) mySum = op<T, r>(mySum, g_idata[j * n + i + blockSize]);

    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    // sdata[tid] = mySum = mySum + sdata[tid + 256];
    sdata[tid] = mySum = op<T, r>(mySum, sdata[tid + 256]);
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    // sdata[tid] = mySum = mySum + sdata[tid + 128];
    sdata[tid] = mySum = op<T, r>(mySum, sdata[tid + 128]);
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    // sdata[tid] = mySum = mySum + sdata[tid + 64];
    sdata[tid] = mySum = op<T, r>(mySum, sdata[tid + 64]);
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    // if (blockSize >= 64) mySum += sdata[tid + 32];
    if (blockSize >= 64) mySum = op<T, r>(mySum, sdata[tid + 32]);
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      // mySum += tile32.shfl_down(mySum, offset);
      mySum = op<T, r>(mySum, tile32.shfl_down(mySum, offset));
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[j * gridDim.y + blockIdx.y] = mySum;
}

#define N_THREADS 512
/*
template<Reduce r> torch::Tensor minimax_cuda(torch::Tensor x) {
  int n = x.size(0);
  int d = x.size(1);
  int e = d / N_THREADS + (d % N_THREADS > 0);

  dim3 d_grid(n, (d + N_THREADS - 1) / N_THREADS);
	int sizeof_dtype;
  switch (x.type().scalarType()) {
  case torch::ScalarType::Double:
    sizeof_dtype = 64;
		break;
  case torch::ScalarType::Float:
    sizeof_dtype = 32;
		break;
	}
  int sm_size = (N_THREADS <= 32) ? 2 * N_THREADS * sizeof_dtype : N_THREADS * sizeof_dtype;

  torch::Tensor y = torch::empty({n, e}, x.options());
	AT_DISPATCH_FLOATING_TYPES(x.type(), "min_cuda_forward", [&] {
  	reduce6<scalar_t, N_THREADS, false, r><<<d_grid, N_THREADS, sm_size>>>(x.data<scalar_t>(), y.data<scalar_t>(), d);
	});
  return std::get<0>(torch::min(y, 1));
}
*/

torch::Tensor min_cuda(torch::Tensor x) {
  int n = x.size(0);
  int d = x.size(1);
  int e = d / N_THREADS + (d % N_THREADS > 0);

  dim3 d_grid(n, (d + N_THREADS - 1) / N_THREADS);
	int sizeof_dtype;
  switch (x.type().scalarType()) {
  case torch::ScalarType::Double:
    sizeof_dtype = 64;
		break;
  case torch::ScalarType::Float:
    sizeof_dtype = 32;
		break;
	}
  int sm_size = (N_THREADS <= 32) ? 2 * N_THREADS * sizeof_dtype : N_THREADS * sizeof_dtype;

  torch::Tensor y = torch::empty({n, e}, x.options());
	AT_DISPATCH_FLOATING_TYPES(x.type(), "min_cuda_forward", [&] {
  	reduce6<scalar_t, N_THREADS, false, MIN><<<d_grid, N_THREADS, sm_size>>>(x.data<scalar_t>(), y.data<scalar_t>(), d);
	});
  return std::get<0>(torch::min(y, 1));
}

torch::Tensor max_cuda(torch::Tensor x) {
  int n = x.size(0);
  int d = x.size(1);
  int e = d / N_THREADS + (d % N_THREADS > 0);

  dim3 d_grid(n, (d + N_THREADS - 1) / N_THREADS);
	int sizeof_dtype;
  switch (x.type().scalarType()) {
  case torch::ScalarType::Double:
    sizeof_dtype = 64;
		break;
  case torch::ScalarType::Float:
    sizeof_dtype = 32;
		break;
	}
  int sm_size = (N_THREADS <= 32) ? 2 * N_THREADS * sizeof_dtype : N_THREADS * sizeof_dtype;

  torch::Tensor y = torch::empty({n, e}, x.options());
	AT_DISPATCH_FLOATING_TYPES(x.type(), "min_cuda_forward", [&] {
  	reduce6<scalar_t, N_THREADS, false, MAX><<<d_grid, N_THREADS, sm_size>>>(x.data<scalar_t>(), y.data<scalar_t>(), d);
	});
  return std::get<0>(torch::max(y, 1));
}
