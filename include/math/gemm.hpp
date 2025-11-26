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
#include "device/device_ptr.hpp"

namespace tnn {

template <typename T>
void gemm(const device_ptr<T[]> &A, const device_ptr<T[]> &B, const device_ptr<T[]> &C,
          const size_t M, const size_t N, const size_t K, const bool trans_A, const bool trans_B,
          const T alpha, const T beta) {
  if (A.device_type() != B.device_type() || A.device_type() != C.device_type()) {
    throw std::runtime_error("All device pointers must be on the same device type for gemm.");
  }
  if (A.device_type() == DeviceType::CPU) {
    cpu::gemm<T>(A.get(), B.get(), C.get(), M, N, K, trans_A, trans_B, alpha, beta);
  }
#ifdef USE_CUDA
  else if (A.device_type() == DeviceType::GPU) {
    cuda::gemm<T>(A.get(), B.get(), C.get(), M, N, K, trans_A, trans_B, alpha, beta);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for gemm.");
  }
}
} // namespace tnn