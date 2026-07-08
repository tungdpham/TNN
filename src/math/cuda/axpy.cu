#include <cublas_v2.h>

#include <stdexcept>

#include "math/cuda/axpy.hpp"

namespace tunx {
namespace cuda {

extern cublasHandle_t get_cublas_handle();

static cudaDataType_t get_cuda_data_type(DType_t dtype) {
  switch (dtype) {
    case DType_t::FP16:
      return CUDA_R_16F;
    case DType_t::BF16:
      return CUDA_R_16BF;
    case DType_t::FP32:
      return CUDA_R_32F;
    case DType_t::FP64:
      return CUDA_R_64F;
    default:
      throw std::runtime_error("Unsupported DType for axpy");
  }
}

static cudaDataType_t get_compute_type(DType_t dtype) {
  if (dtype == DType_t::FP16 || dtype == DType_t::BF16 || dtype == DType_t::FP32) {
    return CUDA_R_32F;
  }
  return get_cuda_data_type(dtype);
}

void axpy(const void* X, void* Y, size_t num_elements, DType_t dtype, cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();
  cublasSetStream(handle, stream);

  cudaDataType_t data_type = get_cuda_data_type(dtype);
  cudaDataType_t compute_type = get_compute_type(dtype);

  float alpha_f32 = 1.0f;
  double alpha_f64 = 1.0;
  const void* alpha_ptr =
      (dtype == DType_t::FP64) ? (const void*)&alpha_f64 : (const void*)&alpha_f32;

  cublasStatus_t status = cublasAxpyEx(handle, num_elements, alpha_ptr, data_type, X, data_type, 1,
                                       Y, data_type, 1, compute_type);

  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cublasAxpyEx failed with status: " + std::to_string(status));
  }
}

}  // namespace cuda
}  // namespace tunx
