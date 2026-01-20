#include "cuda/error_handler.hpp"
#include "math/cuda/gemm.hpp"

#include "type/type.hpp"
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
void gemm<fp16>(const fp16 *A, const fp16 *B, fp16 *C, const size_t M, const size_t N,
                const size_t K, const bool trans_A, const bool trans_B, const fp16 alpha,
                const fp16 beta, cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();
  cublasSetStream(handle, stream);
  cublasOperation_t opA = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasHgemm(handle, opB, opA, N, M, K, &alpha, B, trans_B ? K : N, A, trans_A ? M : K, &beta, C,
              N);
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

template <>
void gemm_strided_batched<float>(const float *A, const float *B, float *C, const size_t M,
                                 const size_t N, const size_t K, const bool trans_A,
                                 const bool trans_B, const float alpha, const float beta,
                                 const size_t batch_count, const size_t stride_A,
                                 const size_t stride_B, const size_t stride_C,
                                 cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();

  cublasSetStream(handle, stream);

  cublasOperation_t opA = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasSgemmStridedBatched(handle, opB, opA, N, M, K, &alpha, B, trans_B ? K : N, stride_B, A,
                            trans_A ? M : K, stride_A, &beta, C, N, stride_C, batch_count);

  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <>
void gemm_strided_batched_ex<float>(const float *A, const float *B, float *C, const size_t M,
                                    const size_t N, const size_t K, const bool trans_A,
                                    const bool trans_B, const float alpha, const float beta,
                                    const size_t batch_count, const size_t stride_A,
                                    const size_t stride_B, const size_t stride_C, const size_t lda,
                                    const size_t ldb, const size_t ldc, cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();

  cublasSetStream(handle, stream);

  cublasOperation_t opA = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;

  size_t lda_val = lda ? lda : (trans_A ? M : K);
  size_t ldb_val = ldb ? ldb : (trans_B ? K : N);
  size_t ldc_val = ldc ? ldc : N;

  cublasSgemmStridedBatched(handle, opB, opA, N, M, K, &alpha, B, ldb_val, stride_B, A, lda_val,
                            stride_A, &beta, C, ldc_val, stride_C, batch_count);

  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <>
void gemm_strided_batched<double>(const double *A, const double *B, double *C, const size_t M,
                                  const size_t N, const size_t K, const bool trans_A,
                                  const bool trans_B, const double alpha, const double beta,
                                  const size_t batch_count, const size_t stride_A,
                                  const size_t stride_B, const size_t stride_C,
                                  cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();

  cublasSetStream(handle, stream);

  cublasOperation_t opA = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasDgemmStridedBatched(handle, opB, opA, N, M, K, &alpha, B, trans_B ? K : N, stride_B, A,
                            trans_A ? M : K, stride_A, &beta, C, N, stride_C, batch_count);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <>
void gemm_strided_batched_ex<double>(const double *A, const double *B, double *C, const size_t M,
                                     const size_t N, const size_t K, const bool trans_A,
                                     const bool trans_B, const double alpha, const double beta,
                                     const size_t batch_count, const size_t stride_A,
                                     const size_t stride_B, const size_t stride_C, const size_t lda,
                                     const size_t ldb, const size_t ldc, cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();

  cublasSetStream(handle, stream);

  cublasOperation_t opA = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;

  size_t lda_val = lda ? lda : (trans_A ? M : K);
  size_t ldb_val = ldb ? ldb : (trans_B ? K : N);
  size_t ldc_val = ldc ? ldc : N;

  cublasDgemmStridedBatched(handle, opB, opA, N, M, K, &alpha, B, ldb_val, stride_B, A, lda_val,
                            stride_A, &beta, C, ldc_val, stride_C, batch_count);
  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <>
void gemm_strided_batched_ex<fp16>(const fp16 *A, const fp16 *B, fp16 *C, const size_t M,
                                   const size_t N, const size_t K, const bool trans_A,
                                   const bool trans_B, const fp16 alpha, const fp16 beta,
                                   const size_t batch_count, const size_t stride_A,
                                   const size_t stride_B, const size_t stride_C, const size_t lda,
                                   const size_t ldb, const size_t ldc, cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();

  cublasSetStream(handle, stream);

  cublasOperation_t opA = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;

  size_t lda_val = lda ? lda : (trans_A ? M : K);
  size_t ldb_val = ldb ? ldb : (trans_B ? K : N);
  size_t ldc_val = ldc ? ldc : N;

  cublasHgemmStridedBatched(handle, opB, opA, N, M, K, &alpha, B, ldb_val, stride_B, A, lda_val,
                            stride_A, &beta, C, ldc_val, stride_C, batch_count);

  cuda::checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

} // namespace cuda
} // namespace tnn
