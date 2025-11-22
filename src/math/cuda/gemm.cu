#include "cuda/error_handler.hpp"
#include "math/cuda/gemm.hpp"

#include <cublas_v2.h>

namespace tnn {
namespace cuda {

cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  if (!handle) {
    cublasCreate(&handle);
  }
  return handle;
}

template <>
void gemm<float>(const float *A, const float *B, float *C, const size_t M, const size_t N,
                 const size_t K, const bool trans_A, const bool trans_B, const float alpha,
                 const float beta, cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();

  cublasSetStream(handle, stream);

  cublasOperation_t opA = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasSgemm(handle, opB, opA, N, M, K, &alpha, B, trans_B ? K : N, A, trans_A ? M : K, &beta, C,
              N);

  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <>
void gemm<double>(const double *A, const double *B, double *C, const size_t M, const size_t N,
                  const size_t K, const bool trans_A, const bool trans_B, const double alpha,
                  const double beta, cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();

  cublasSetStream(handle, stream);

  cublasOperation_t opA = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasDgemm(handle, opB, opA, N, M, K, &alpha, B, trans_B ? K : N, A, trans_A ? M : K, &beta, C,
              N);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

} // namespace cuda
} // namespace tnn
