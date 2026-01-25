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

template <typename T> struct CudaType;
template <> struct CudaType<fp16> {
  static constexpr cudaDataType_t type = CUDA_R_16F;
};
template <> struct CudaType<float> {
  static constexpr cudaDataType_t type = CUDA_R_32F;
};
template <> struct CudaType<double> {
  static constexpr cudaDataType_t type = CUDA_R_64F;
};

template <typename T> struct CublasComputeType;
template <> struct CublasComputeType<fp16> {
  static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_16F;
};
template <> struct CublasComputeType<float> {
  static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32F;
};
template <> struct CublasComputeType<double> {
  static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_64F;
};

template <typename A_T, typename B_T, typename C_T, typename Compute_T>
void gemm_ex(const A_T *A, const B_T *B, C_T *C, const size_t M, const size_t N, const size_t K,
             const bool transA, const bool transB, const Compute_T alpha, const Compute_T beta,
             const size_t lda, const size_t ldb, const size_t ldc, cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();
  cublasSetStream(handle, stream);

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasGemmEx(handle, opB, opA, N, M, K, &alpha, B, CudaType<B_T>::type, ldb, A,
               CudaType<A_T>::type, lda, &beta, C, CudaType<C_T>::type, ldc,
               CublasComputeType<Compute_T>::type, CUBLAS_GEMM_DEFAULT);
  tnn::cuda::checkCudaError(cudaGetLastError(), "copy", __FILE__, __LINE__);
}

template <typename A_T, typename B_T, typename C_T, typename Compute_T>
void gemm_strided_batched_ex(const A_T *A, const B_T *B, C_T *C, const size_t M, const size_t N,
                             const size_t K, const bool transA, const bool transB,
                             const Compute_T alpha, const Compute_T beta, const size_t lda,
                             const size_t ldb, const size_t ldc, const size_t strideA,
                             const size_t strideB, const size_t strideC, const size_t batch_count,
                             cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();
  cublasSetStream(handle, stream);

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasGemmStridedBatchedEx(handle, opB, opA, N, M, K, &alpha, B, CudaType<B_T>::type, ldb,
                             strideB, A, CudaType<A_T>::type, lda, strideA, &beta, C,
                             CudaType<C_T>::type, ldc, strideC, batch_count,
                             CublasComputeType<Compute_T>::type, CUBLAS_GEMM_DEFAULT);
  tnn::cuda::checkCudaError(cudaGetLastError(), "copy", __FILE__, __LINE__);
}

#define INSTANTIATE_CUBLAS_GEMM(A_T, B_T, C_T, Compute_T)                                          \
  template void gemm_ex<A_T, B_T, C_T, Compute_T>(                                                 \
      const A_T *A, const B_T *B, C_T *C, const size_t M, const size_t N, const size_t K,          \
      const bool transA, const bool transB, const Compute_T alpha, const Compute_T beta,           \
      const size_t lda, const size_t ldb, const size_t ldc, cudaStream_t stream);                  \
  template void gemm_strided_batched_ex<A_T, B_T, C_T, Compute_T>(                                 \
      const A_T *A, const B_T *B, C_T *C, const size_t M, const size_t N, const size_t K,          \
      const bool transA, const bool transB, const Compute_T alpha, const Compute_T beta,           \
      const size_t lda, const size_t ldb, const size_t ldc, const size_t strideA,                  \
      const size_t strideB, const size_t strideC, const size_t batch_count, cudaStream_t stream);

#define INSTANTIATE_CUBLAS_GEMM_COMPUTE(A_T, B_T, C_T, COMPUTE_T)                                  \
  INSTANTIATE_CUBLAS_GEMM(A_T, B_T, C_T, COMPUTE_T)

#define INSTANTIATE_CUBLAS_GEMM_C(A_T, B_T, C_T)                                                   \
  INSTANTIATE_CUBLAS_GEMM_COMPUTE(A_T, B_T, C_T, fp16)                                             \
  INSTANTIATE_CUBLAS_GEMM_COMPUTE(A_T, B_T, C_T, float)                                            \
  INSTANTIATE_CUBLAS_GEMM_COMPUTE(A_T, B_T, C_T, double)

#define INSTANTIATE_CUBLAS_GEMM_B(A_T, B_T)                                                        \
  INSTANTIATE_CUBLAS_GEMM_C(A_T, B_T, fp16)                                                        \
  INSTANTIATE_CUBLAS_GEMM_C(A_T, B_T, float)                                                       \
  INSTANTIATE_CUBLAS_GEMM_C(A_T, B_T, double)

#define INSTANTIATE_CUBLAS_GEMM_A(A_T)                                                             \
  INSTANTIATE_CUBLAS_GEMM_B(A_T, fp16)                                                             \
  INSTANTIATE_CUBLAS_GEMM_B(A_T, float)                                                            \
  INSTANTIATE_CUBLAS_GEMM_B(A_T, double)

INSTANTIATE_CUBLAS_GEMM_A(fp16)
INSTANTIATE_CUBLAS_GEMM_A(float)
INSTANTIATE_CUBLAS_GEMM_A(double)
#undef INSTANTIATE_CUBLAS_GEMM_A
#undef INSTANTIATE_CUBLAS_GEMM_B
#undef INSTANTIATE_CUBLAS_GEMM_C
#undef INSTANTIATE_CUBLAS_GEMM_COMPUTE
#undef INSTANTIATE_CUBLAS_GEMM

} // namespace cuda
} // namespace tnn
