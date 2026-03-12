#pragma once

#include <cstdint>
#include <stdexcept>
#ifdef USE_CUDA
#include <cuda_bf16.h>  // IWYU pragma: export
#include <cuda_fp16.h>  // IWYU pragma: export
#else
#include "type/bf16.hpp"  // IWYU pragma: export
#include "type/fp16.hpp"  // IWYU pragma: export
#endif
#include <vector>

namespace tnn {
template <typename T>
using Vec = std::vector<T>;
using std::string;

#if defined(USE_CUDA)
typedef __half fp16;
typedef __nv_bfloat16 bf16;
#endif
typedef float fp32;
typedef double fp64;
typedef unsigned char uchar;

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<bf16> {
  static constexpr const char *name = "bf16";
  static const float epsilon;
  using ComputePrecision = fp32;
  using HigherPrecision = fp32;
};

template <>
struct TypeTraits<fp16> {
  static constexpr const char *name = "fp16";
  static const float epsilon;
  using ComputePrecision = fp32;
  using HigherPrecision = fp32;
};

template <>
struct TypeTraits<fp32> {
  static constexpr const char *name = "float32";
  static const float epsilon;
  using ComputePrecision = fp32;
  using HigherPrecision = fp64;
};

template <>
struct TypeTraits<fp64> {
  static constexpr const char *name = "float64";
  static const float epsilon;
  using ComputePrecision = fp64;
  using HigherPrecision = fp64;
};

enum class DType_t : uint32_t {
  BYTE,
  UINT8_T,
  BOOL,
  FP16,
  BF16,
  FP32,
  FP64,
  INT32_T,
  INT64_T,
  SIZE_T,
  UNKNOWN
};

template <typename T>
constexpr DType_t dtype_of() {
  return DType_t::UNKNOWN;
}
template <>
constexpr DType_t dtype_of<uchar>() {
  return DType_t::BYTE;
}
template <>
constexpr DType_t dtype_of<bool>() {
  return DType_t::BOOL;
}
template <>
constexpr DType_t dtype_of<bf16>() {
  return DType_t::BF16;
}
template <>
constexpr DType_t dtype_of<fp16>() {
  return DType_t::FP16;
}
template <>
constexpr DType_t dtype_of<float>() {
  return DType_t::FP32;
}
template <>
constexpr DType_t dtype_of<double>() {
  return DType_t::FP64;
}
template <>
constexpr DType_t dtype_of<int32_t>() {
  return DType_t::INT32_T;
}
template <>
constexpr DType_t dtype_of<int64_t>() {
  return DType_t::INT64_T;
}
template <>
constexpr DType_t dtype_of<size_t>() {
  return DType_t::SIZE_T;
}

enum class SBool : uint8_t { FALSE = 0, TRUE = 1 };

inline float dtype_eps(DType_t dtype) {
  switch (dtype) {
    case DType_t::FP16:
      return TypeTraits<fp16>::epsilon;
    case DType_t::BF16:
      return TypeTraits<bf16>::epsilon;
    case DType_t::FP32:
      return TypeTraits<fp32>::epsilon;
    case DType_t::FP64:
      return TypeTraits<fp64>::epsilon;
    default:
      throw std::runtime_error("Unknown data type for dtype_eps");
  }
}

inline size_t get_dtype_size(DType_t dtype) {
  switch (dtype) {
    case DType_t::BYTE:
      return sizeof(uchar);
    case DType_t::UINT8_T:
      return sizeof(uint8_t);
    case DType_t::BOOL:
      return sizeof(bool);
    case DType_t::FP16:
      return sizeof(fp16);
    case DType_t::BF16:
      return sizeof(bf16);
    case DType_t::FP32:
      return sizeof(fp32);
    case DType_t::FP64:
      return sizeof(fp64);
    case DType_t::INT32_T:
      return sizeof(int32_t);
    case DType_t::INT64_T:
      return sizeof(int64_t);
    case DType_t::SIZE_T:
      return sizeof(size_t);
    default:
      throw std::runtime_error("Unknown data type for get_dtype_size");
  }
}

inline std::string dtype_to_string(DType_t dtype) {
  switch (dtype) {
    case DType_t::BYTE:
      return "BYTE";
    case DType_t::UINT8_T:
      return "UINT8_T";
    case DType_t::BOOL:
      return "BOOL";
    case DType_t::FP16:
      return "FP16";
    case DType_t::BF16:
      return "BF16";
    case DType_t::FP32:
      return "FP32";
    case DType_t::FP64:
      return "FP64";
    case DType_t::INT32_T:
      return "INT32";
    case DType_t::INT64_T:
      return "INT64";
    case DType_t::SIZE_T:
      return "SIZE_T";
    default:
      return "UNKNOWN";
  }
}

inline DType_t string_to_dtype(const std::string &dtype_str) {
  if (dtype_str == "BYTE" || dtype_str == "UINT8") {
    return DType_t::UINT8_T;
  } else if (dtype_str == "BOOL") {
    return DType_t::BOOL;
  } else if (dtype_str == "FP16") {
    return DType_t::FP16;
  } else if (dtype_str == "BF16") {
    return DType_t::BF16;
  } else if (dtype_str == "FP32" || dtype_str == "FLOAT") {
    return DType_t::FP32;
  } else if (dtype_str == "FP64" || dtype_str == "DOUBLE") {
    return DType_t::FP64;
  } else if (dtype_str == "INT32") {
    return DType_t::INT32_T;
  } else if (dtype_str == "INT64") {
    return DType_t::INT64_T;
  } else if (dtype_str == "SIZE_T") {
    return DType_t::SIZE_T;
  } else {
    throw std::runtime_error("Unknown data type string: " + dtype_str);
  }
}

#define DISPATCH_DTYPE(dtype_value, type_alias, ...)         \
  switch (dtype_value) {                                     \
    case DType_t::FP16: {                                    \
      using type_alias = fp16;                               \
      __VA_ARGS__;                                           \
      break;                                                 \
    }                                                        \
    case DType_t::BF16: {                                    \
      using type_alias = bf16;                               \
      __VA_ARGS__;                                           \
      break;                                                 \
    }                                                        \
    case DType_t::FP32: {                                    \
      using type_alias = float;                              \
      __VA_ARGS__;                                           \
      break;                                                 \
    }                                                        \
    case DType_t::FP64: {                                    \
      using type_alias = double;                             \
      __VA_ARGS__;                                           \
      break;                                                 \
    }                                                        \
    default:                                                 \
      throw std::runtime_error("Unknown dtype in dispatch"); \
  }

#define DISPATCH_ON_ANY_DTYPE(dtype_value, type_alias, ...)  \
  switch (dtype_value) {                                     \
    case DType_t::UINT8_T: {                                 \
      using type_alias = uint8_t;                            \
      __VA_ARGS__;                                           \
      break;                                                 \
    }                                                        \
    case DType_t::FP16: {                                    \
      using type_alias = fp16;                               \
      __VA_ARGS__;                                           \
      break;                                                 \
    }                                                        \
    case DType_t::BF16: {                                    \
      using type_alias = bf16;                               \
      __VA_ARGS__;                                           \
      break;                                                 \
    }                                                        \
    case DType_t::FP32: {                                    \
      using type_alias = float;                              \
      __VA_ARGS__;                                           \
      break;                                                 \
    }                                                        \
    case DType_t::FP64: {                                    \
      using type_alias = double;                             \
      __VA_ARGS__;                                           \
      break;                                                 \
    }                                                        \
    default:                                                 \
      throw std::runtime_error("Unknown dtype in dispatch"); \
  }

}  // namespace tnn