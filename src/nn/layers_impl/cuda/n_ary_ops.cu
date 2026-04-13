/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>

#include "nn/layers_impl/cuda/n_ary_ops.hpp"
#include "type/cuda/vectorized_types.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cuda {
namespace nary {

static size_t compute_total_elements(const Vec<size_t>& shape) {
  size_t total = 1;
  for (auto dim : shape) total *= dim;
  return total;
}

template <typename T, typename Functor>
__global__ void nary_forward_vec_kernel(const T** inputs, T* output, size_t n_inputs,
                                        size_t n_vectors) {
  using VecT = typename VectoredTraits<T>::type;
  static constexpr int VecSize = VectoredTraits<T>::size;
  Functor op;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vectors) return;

  const VecT* in0 = reinterpret_cast<const VecT*>(inputs[0]);
  VecT acc = in0[idx];

  for (size_t i = 1; i < n_inputs; ++i) {
    const VecT* ini = reinterpret_cast<const VecT*>(inputs[i]);
    VecT val = ini[idx];
    T* acc_ptr = reinterpret_cast<T*>(&acc);
    const T* val_ptr = reinterpret_cast<const T*>(&val);
    for (int k = 0; k < VecSize; ++k) {
      acc_ptr[k] = op(acc_ptr[k], val_ptr[k]);
    }
  }

  VecT* out_vec = reinterpret_cast<VecT*>(output);
  out_vec[idx] = acc;
}

template <typename T, typename Functor>
__global__ void nary_forward_scalar_tail_kernel(const T** inputs, T* output, size_t n_inputs,
                                                size_t offset, size_t tail_elements) {
  Functor op;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tail_elements) return;
  size_t pos = offset + idx;
  T acc = inputs[0][pos];
  for (size_t i = 1; i < n_inputs; ++i) acc = op(acc, inputs[i][pos]);
  output[pos] = acc;
}

template <typename T>
__global__ void nary_backward_add_vec_kernel(const T* grad_output, T** grad_inputs, size_t n_inputs,
                                             size_t n_vectors) {
  using VecT = typename VectoredTraits<T>::type;
  static constexpr int VecSize = VectoredTraits<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vectors) return;

  const VecT* go_vec = reinterpret_cast<const VecT*>(grad_output);
  VecT g = go_vec[idx];

  for (size_t i = 0; i < n_inputs; ++i) {
    VecT* gi_vec = reinterpret_cast<VecT*>(grad_inputs[i]);
    VecT cur = gi_vec[idx];
    T* cur_ptr = reinterpret_cast<T*>(&cur);
    const T* g_ptr = reinterpret_cast<const T*>(&g);
    for (int k = 0; k < VecSize; ++k) cur_ptr[k] += g_ptr[k];
    gi_vec[idx] = cur;
  }
}

template <typename T>
__global__ void nary_backward_add_scalar_tail_kernel(const T* grad_output, T** grad_inputs,
                                                     size_t n_inputs, size_t offset,
                                                     size_t tail_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tail_elements) return;
  size_t pos = offset + idx;
  T g = grad_output[pos];
  for (size_t i = 0; i < n_inputs; ++i) grad_inputs[i][pos] += g;
}

template <typename T>
__global__ void nary_backward_sub_vec_kernel(const T* grad_output, T** grad_inputs, size_t n_inputs,
                                             size_t n_vectors) {
  using VecT = typename VectoredTraits<T>::type;
  static constexpr int VecSize = VectoredTraits<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vectors) return;

  const VecT* go_vec = reinterpret_cast<const VecT*>(grad_output);
  VecT g = go_vec[idx];

  {
    VecT* gi_vec = reinterpret_cast<VecT*>(grad_inputs[0]);
    VecT cur = gi_vec[idx];
    T* cur_ptr = reinterpret_cast<T*>(&cur);
    const T* g_ptr = reinterpret_cast<const T*>(&g);
    for (int k = 0; k < VecSize; ++k) cur_ptr[k] += g_ptr[k];
    gi_vec[idx] = cur;
  }

  for (size_t i = 1; i < n_inputs; ++i) {
    VecT* gi_vec = reinterpret_cast<VecT*>(grad_inputs[i]);
    VecT cur = gi_vec[idx];
    T* cur_ptr = reinterpret_cast<T*>(&cur);
    const T* g_ptr = reinterpret_cast<const T*>(&g);
    for (int k = 0; k < VecSize; ++k) cur_ptr[k] -= g_ptr[k];
    gi_vec[idx] = cur;
  }
}

template <typename T>
__global__ void nary_backward_sub_scalar_tail_kernel(const T* grad_output, T** grad_inputs,
                                                     size_t n_inputs, size_t offset,
                                                     size_t tail_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tail_elements) return;
  size_t pos = offset + idx;
  T g = grad_output[pos];
  grad_inputs[0][pos] += g;
  for (size_t i = 1; i < n_inputs; ++i) grad_inputs[i][pos] -= g;
}

template <typename T>
__global__ void nary_backward_mul_kernel(const T* grad_output, T** grad_inputs,
                                         const T** fwd_inputs, size_t n_inputs,
                                         size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;
  T grad = grad_output[idx];
  for (size_t i = 0; i < n_inputs; ++i) {
    T contrib = grad;
    for (size_t j = 0; j < n_inputs; ++j) {
      if (i != j) contrib *= fwd_inputs[j][idx];
    }
    grad_inputs[i][idx] += contrib;
  }
}

template <typename T>
__global__ void nary_backward_div_kernel(const T* grad_output, T** grad_inputs,
                                         const T** fwd_inputs, size_t n_inputs,
                                         size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;
  T grad = grad_output[idx];

  T denom_prod = static_cast<T>(1.0);
  for (size_t i = 1; i < n_inputs; ++i) denom_prod *= fwd_inputs[i][idx];

  grad_inputs[0][idx] += grad / denom_prod;
  for (size_t i = 1; i < n_inputs; ++i) {
    grad_inputs[i][idx] += -grad * fwd_inputs[0][idx] / (denom_prod * fwd_inputs[i][idx]);
  }
}

template <typename T>
static const T** upload_const_ptrs(const Vec<const T*>& host_ptrs, void* workspace,
                                   size_t byte_offset) {
  size_t n = host_ptrs.size();
  char* dst = static_cast<char*>(workspace) + byte_offset;
  cudaMemcpy(dst, host_ptrs.data(), n * sizeof(const T*), cudaMemcpyHostToDevice);
  return reinterpret_cast<const T**>(dst);
}

template <typename T>
static T** upload_ptrs(const Vec<T*>& host_ptrs, void* workspace, size_t byte_offset) {
  size_t n = host_ptrs.size();
  char* dst = static_cast<char*>(workspace) + byte_offset;
  cudaMemcpy(dst, host_ptrs.data(), n * sizeof(T*), cudaMemcpyHostToDevice);
  return reinterpret_cast<T**>(dst);
}

template <typename T>
void run_forward(const Vec<const T*>& inputs, T* output, const Vec<size_t>& shape,
                 const NAryOp& op_type, void* workspace, cudaStream_t stream) {
  if (inputs.empty()) {
    throw std::runtime_error("run_forward (CUDA): requires at least one input");
  }

  size_t total_elements = compute_total_elements(shape);
  size_t n_inputs = inputs.size();

  const T** d_inputs = upload_const_ptrs(inputs, workspace, 0);

  constexpr int VecSize = VectoredTraits<T>::size;
  size_t n_vectors = total_elements / VecSize;
  size_t tail = total_elements % VecSize;

  constexpr size_t block_size = 256;

  auto launch_forward = [&](auto functor_tag) {
    using Functor = decltype(functor_tag);
    if (n_vectors > 0) {
      size_t grid = (n_vectors + block_size - 1) / block_size;
      nary_forward_vec_kernel<T, Functor>
          <<<grid, block_size, 0, stream>>>(d_inputs, output, n_inputs, n_vectors);
    }
    if (tail > 0) {
      size_t offset = n_vectors * VecSize;
      size_t grid = (tail + block_size - 1) / block_size;
      nary_forward_scalar_tail_kernel<T, Functor>
          <<<grid, block_size, 0, stream>>>(d_inputs, output, n_inputs, offset, tail);
    }
  };

  switch (op_type) {
    case NAryOp::ADD:
      launch_forward(functors::Add<T>{});
      break;
    case NAryOp::SUB:
      launch_forward(functors::Sub<T>{});
      break;
    case NAryOp::MUL:
      launch_forward(functors::Mul<T>{});
      break;
    case NAryOp::DIV:
      launch_forward(functors::Div<T>{});
      break;
    default:
      throw std::runtime_error("run_forward (CUDA): unknown operation type");
  }
}

template <typename T>
void run_backward(const T* grad_output, Vec<T*>& grad_inputs, const Vec<const T*>& fwd_inputs,
                  const Vec<size_t>& shape, const NAryOp& op_type, void* workspace,
                  cudaStream_t stream) {
  if (fwd_inputs.empty() || grad_inputs.empty()) {
    throw std::runtime_error("run_backward (CUDA): requires at least one input");
  }

  size_t total_elements = compute_total_elements(shape);
  size_t n_inputs = fwd_inputs.size();

  size_t ptr_block = n_inputs * sizeof(void*);
  const T** d_fwd_inputs = upload_const_ptrs(fwd_inputs, workspace, 0);
  T** d_grad_inputs = upload_ptrs(grad_inputs, workspace, ptr_block);

  constexpr int VecSize = VectoredTraits<T>::size;
  size_t n_vectors = total_elements / VecSize;
  size_t tail = total_elements % VecSize;
  size_t tail_offset = n_vectors * VecSize;

  constexpr size_t block_size = 256;

  switch (op_type) {
    case NAryOp::ADD: {
      if (n_vectors > 0) {
        size_t grid = (n_vectors + block_size - 1) / block_size;
        nary_backward_add_vec_kernel<T>
            <<<grid, block_size, 0, stream>>>(grad_output, d_grad_inputs, n_inputs, n_vectors);
      }
      if (tail > 0) {
        size_t grid = (tail + block_size - 1) / block_size;
        nary_backward_add_scalar_tail_kernel<T><<<grid, block_size, 0, stream>>>(
            grad_output, d_grad_inputs, n_inputs, tail_offset, tail);
      }
      break;
    }
    case NAryOp::SUB: {
      if (n_vectors > 0) {
        size_t grid = (n_vectors + block_size - 1) / block_size;
        nary_backward_sub_vec_kernel<T>
            <<<grid, block_size, 0, stream>>>(grad_output, d_grad_inputs, n_inputs, n_vectors);
      }
      if (tail > 0) {
        size_t grid = (tail + block_size - 1) / block_size;
        nary_backward_sub_scalar_tail_kernel<T><<<grid, block_size, 0, stream>>>(
            grad_output, d_grad_inputs, n_inputs, tail_offset, tail);
      }
      break;
    }
    case NAryOp::MUL: {
      size_t grid = (total_elements + block_size - 1) / block_size;
      nary_backward_mul_kernel<T><<<grid, block_size, 0, stream>>>(
          grad_output, d_grad_inputs, d_fwd_inputs, n_inputs, total_elements);
      break;
    }
    case NAryOp::DIV: {
      size_t grid = (total_elements + block_size - 1) / block_size;
      nary_backward_div_kernel<T><<<grid, block_size, 0, stream>>>(
          grad_output, d_grad_inputs, d_fwd_inputs, n_inputs, total_elements);
      break;
    }
    default:
      throw std::runtime_error("run_backward (CUDA): unknown operation type");
  }
}

#define INSTANTIATE_NARY_OPS(T)                                                                    \
  template void run_forward<T>(const Vec<const T*>&, T*, const Vec<size_t>&, const NAryOp&, void*, \
                               cudaStream_t);                                                      \
  template void run_backward<T>(const T*, Vec<T*>&, const Vec<const T*>&, const Vec<size_t>&,      \
                                const NAryOp&, void*, cudaStream_t);

INSTANTIATE_NARY_OPS(fp16)
INSTANTIATE_NARY_OPS(bf16)
INSTANTIATE_NARY_OPS(float)
INSTANTIATE_NARY_OPS(double)

#undef INSTANTIATE_NARY_OPS

}  // namespace nary
}  // namespace cuda
}  // namespace tnn

#endif
