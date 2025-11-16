#include "cuda/error_handler.hpp"
#include "math/cuda/gemm.hpp"

#include <cublas_v2.h>

namespace tnn {
namespace cuda {

// Helper function to get cuBLAS handle (this should be managed globally)
cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  if (!handle) {
    cublasCreate(&handle);
  }
  return handle;
}

// Specialization for float
template <>
void gemm<float>(const float *A, const float *B, float *C, const size_t M, const size_t N,
                 const size_t K, const float alpha, const bool trans_A, const bool trans_B) {
  cublasHandle_t handle = get_cublas_handle();
  const float beta = 0.0;

  cublasOperation_t opA = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Note: cuBLAS uses column-major, so we need to swap A and B
  cublasSgemm(handle, opB, opA, N, M, K, &alpha, B, trans_B ? K : N, A, trans_A ? M : K, &beta, C,
              N);

  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

// Specialization for double
template <>
void gemm<double>(const double *A, const double *B, double *C, const size_t M, const size_t N,
                  const size_t K, const double alpha, const bool trans_A, const bool trans_B) {
  cublasHandle_t handle = get_cublas_handle();
  const double beta = 0.0;

  cublasOperation_t opA = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Note: cuBLAS uses column-major, so we need to swap A and B
  cublasDgemm(handle, opB, opA, N, M, K, &alpha, B, trans_B ? K : N, A, trans_A ? M : K, &beta, C,
              N);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

} // namespace cuda
} // namespace tnn
