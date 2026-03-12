#pragma once

#ifdef USE_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "type/type.hpp"

namespace tnn {

template <typename T>
struct VectoredTraits;

template <>
struct VectoredTraits<uint8_t> {
  using type = uchar4;
  static constexpr int size = 4;
};

template <>
struct VectoredTraits<char> {
  using type = char4;
  static constexpr int size = 4;
};

template <>
struct VectoredTraits<int> {
  using type = int4;
  static constexpr int size = 4;
};

template <>
struct VectoredTraits<fp16> {
  using type = half2;
  static constexpr int size = 2;
};

template <>
struct VectoredTraits<bf16> {
  using type = __nv_bfloat162;
  static constexpr int size = 2;
};

template <>
struct VectoredTraits<float> {
  using type = float4;
  static constexpr int size = 4;
};

template <>
struct VectoredTraits<double> {
  using type = double2;
  static constexpr int size = 2;
};

struct UInt64Vec {
  unsigned long x;
};

template <>
struct VectoredTraits<unsigned long> {
  using type = UInt64Vec;
  static constexpr int size = 1;
};

namespace functors {

template <typename T>
struct Add {
  __device__ T operator()(T a, T b) const { return a + b; }
};
template <typename T>
struct Sub {
  __device__ T operator()(T a, T b) const { return a - b; }
};
template <typename T>
struct Mul {
  __device__ T operator()(T a, T b) const { return a * b; }
};
template <typename T>
struct Div {
  __device__ T operator()(T a, T b) const { return a / b; }
};
template <typename T>
struct Min {
  __device__ T operator()(T a, T b) const { return (a < b) ? a : b; }
};
template <typename T>
struct Max {
  __device__ T operator()(T a, T b) const { return (a > b) ? a : b; }
};

template <typename T>
struct Equal {
  __device__ T operator()(T a, T b) const { return (a == b) ? (T)1 : (T)0; }
};
template <typename T>
struct Greater {
  __device__ T operator()(T a, T b) const { return (a > b) ? (T)1 : (T)0; }
};

template <typename T>
struct FMAdd;
template <>
struct FMAdd<fp16> {
  __device__ fp16 operator()(fp16 a, fp16 b, fp16 c) const { return __hfma(a, b, c); }
};
template <>
struct FMAdd<bf16> {
  __device__ bf16 operator()(bf16 a, bf16 b, bf16 c) const { return __hfma(a, b, c); }
};
template <>
struct FMAdd<float> {
  __device__ float operator()(float a, float b, float c) const { return fmaf(a, b, c); }
};
template <>
struct FMAdd<double> {
  __device__ double operator()(double a, double b, double c) const { return fma(a, b, c); }
};

template <typename T>
struct FMSub;
template <>
struct FMSub<fp16> {
  __device__ fp16 operator()(fp16 a, fp16 b, fp16 c) const { return __hfma(a, b, __hneg(c)); }
};
template <>
struct FMSub<bf16> {
  __device__ bf16 operator()(bf16 a, bf16 b, bf16 c) const { return __hfma(a, b, __hneg(c)); }
};
template <>
struct FMSub<float> {
  __device__ float operator()(float a, float b, float c) const { return fmaf(a, b, -c); }
};
template <>
struct FMSub<double> {
  __device__ double operator()(double a, double b, double c) const { return fma(a, b, -c); }
};

template <typename T>
struct FNMAdd;
template <>
struct FNMAdd<fp16> {
  __device__ fp16 operator()(fp16 a, fp16 b, fp16 c) const { return __hfma(__hneg(a), b, c); }
};
template <>
struct FNMAdd<bf16> {
  __device__ bf16 operator()(bf16 a, bf16 b, bf16 c) const { return __hfma(__hneg(a), b, c); }
};
template <>
struct FNMAdd<float> {
  __device__ float operator()(float a, float b, float c) const { return fmaf(-a, b, c); }
};
template <>
struct FNMAdd<double> {
  __device__ double operator()(double a, double b, double c) const { return fma(-a, b, c); }
};

template <typename T>
struct Sqrt {
  __device__ T operator()(T a) const { return sqrtf((float)a); }
};
template <>
struct Sqrt<fp16> {
  __device__ fp16 operator()(fp16 a) const { return hsqrt(a); }
};
template <>
struct Sqrt<bf16> {
  __device__ bf16 operator()(bf16 a) const { return hsqrt(a); }
};
template <>
struct Sqrt<float> {
  __device__ float operator()(float a) const { return sqrtf(a); }
};
template <>
struct Sqrt<double> {
  __device__ double operator()(double a) const { return sqrt(a); }
};
template <>
struct Sqrt<unsigned long> {
  __device__ unsigned long operator()(unsigned long a) const { return sqrtf((float)a); }
};

template <typename T>
struct Rsqrt {
  __device__ T operator()(T a) const { return (T)1 / sqrt(a); }
};
template <>
struct Rsqrt<float> {
  __device__ float operator()(float a) const { return rsqrtf(a); }
};
template <>
struct Rsqrt<bf16> {
  __device__ bf16 operator()(bf16 a) const { return hrsqrt(a); }
};
template <typename T>
struct Rcp {
  __device__ T operator()(T a) const { return (T)1 / a; }
};

template <typename T>
struct Abs {
  __device__ T operator()(T a) const { return abs(a); }
};
template <>
struct Abs<fp16> {
  __device__ fp16 operator()(fp16 a) const { return __habs(a); }
};
template <>
struct Abs<bf16> {
  __device__ bf16 operator()(bf16 a) const { return __habs(a); }
};
template <>
struct Abs<float> {
  __device__ float operator()(float a) const { return fabsf(a); }
};
template <>
struct Abs<double> {
  __device__ double operator()(double a) const { return fabs(a); }
};
template <>
struct Abs<unsigned long> {
  __device__ unsigned long operator()(unsigned long a) const { return a; }
};

template <typename T>
struct AddScalar {
  T s;
  __device__ T operator()(T a) const { return a + s; }
};
template <typename T>
struct SubScalar {
  T s;
  __device__ T operator()(T a) const { return a - s; }
};
template <typename T>
struct MulScalar {
  T s;
  __device__ T operator()(T a) const { return a * s; }
};
template <typename T>
struct DivScalar {
  T s;
  __device__ T operator()(T a) const { return a / s; }
};
template <typename T>
struct ScalarMax {
  T s;
  __device__ T operator()(T a) const { return (a > s) ? a : s; }
};
template <typename T>
struct Clamp {
  T min_v, max_v;
  __device__ T operator()(T a) const { return (a < min_v) ? min_v : ((a > max_v) ? max_v : a); }
};

template <typename T>
struct SubMulScalar {
  T sub, mul;
  __device__ T operator()(T a) const { return (a - sub) * mul; }
};
template <typename T>
struct MulAddScalar {
  T mul, add;
  __device__ T operator()(T a) const { return a * mul + add; }
};
template <typename T>
struct Axpy {
  T alpha;
  __device__ T operator()(T x, T y) const { return alpha * x + y; }
};

}  // namespace functors
}  // namespace tnn

#endif