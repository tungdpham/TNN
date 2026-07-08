#pragma once

#ifdef USE_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "type/type.hpp"

namespace tunx {
namespace cuda {

void synchronize();

template <typename T>
__device__ __forceinline__ void gpu_atomic_add(T* address, T val) {
  atomicAdd(address, val);
}

template <>
__device__ __forceinline__ void gpu_atomic_add<int8_t>(int8_t* address, int8_t val) {
  unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 3));
  unsigned int shift = ((size_t)address & 3) * 8;
  unsigned int old = *address_as_ui, assumed;
  do {
    assumed = old;
    int8_t old_val = (char)((assumed >> shift) & 0xFF);
    int8_t new_val = old_val + val;
    unsigned int new_ui =
        (assumed & ~(0xFFu << shift)) | (((unsigned int)new_val & 0xFFu) << shift);
    old = atomicCAS(address_as_ui, assumed, new_ui);
  } while (assumed != old);
}

template <typename T>
__device__ __forceinline__ T lowest_value();

template <>
__device__ __forceinline__ float lowest_value<float>() {
  return -1.0e20f;
}

template <>
__device__ __forceinline__ double lowest_value<double>() {
  return -1.0e300;
}

template <>
__device__ __forceinline__ int8 lowest_value<int8>() {
  return -128;
}

template <>
__device__ __forceinline__ int lowest_value<int>() {
  return -2147483648;
}

template <typename T>
__device__ __forceinline__ T device_exp(T value);

template <>
__device__ __forceinline__ float device_exp<float>(float value) {
  return expf(value);
}

template <>
__device__ __forceinline__ double device_exp<double>(double value) {
  return exp(value);
}

template <>
__device__ __forceinline__ int8 device_exp<int8>(int8 value) {
  return static_cast<int8>(expf(static_cast<float>(value)));
}

template <>
__device__ __forceinline__ int device_exp<int>(int value) {
  return static_cast<int>(expf(static_cast<float>(value)));
}

}  // namespace cuda
}  // namespace tunx

#endif
