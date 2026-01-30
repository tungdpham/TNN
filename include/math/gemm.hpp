/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

/**
 * This file provides a wrapper header for gemm functions
 */
#include "cpu/gemm.hpp"
#include "cuda/gemm.hpp"
#include "device/dptr.hpp"
#include "device/task.hpp"

namespace tnn {

template <typename IO_T, typename Param_T = IO_T, typename Compute_T = IO_T>
void gemm(const dptr &A, const dptr &B, const dptr &C, const size_t M, const size_t N,
          const size_t K, const bool trans_A, const bool trans_B, const IO_T alpha, const IO_T beta,
          const size_t lda, const size_t ldb, const size_t ldc) {
  if (A.device_type() != B.device_type() || A.device_type() != C.device_type()) {
    throw std::runtime_error("All device pointers must be on the same device type for gemm.");
  }
  if (A.device_type() == DeviceType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "gemm mixed dtype dispatch not implemented for CPU (io/param/compute must match).");
    }
    create_cpu_task("default", cpu::gemm<IO_T>, A.get<IO_T>(), B.get<Param_T>(), C.get<IO_T>(), M,
                    N, K, trans_A, trans_B, alpha, beta, lda, ldb, ldc);
  }
#ifdef USE_CUDA
  else if (A.device_type() == DeviceType::GPU) {
    create_cuda_task("default", cuda::gemm_ex<IO_T, Param_T, Compute_T>, A.get<IO_T>(),
                     B.get<Param_T>(), C.get<IO_T>(), M, N, K, trans_A, trans_B, alpha, beta, lda,
                     ldb, ldc);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for gemm.");
  }
}
} // namespace tnn