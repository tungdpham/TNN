#pragma once

#include "ops/ops.hpp"
#include "tensor.hpp"

namespace synet {
inline Tensor &operator+=(Tensor &lhs, const Tensor &rhs) {
  if (lhs.shape() != rhs.shape())
    throw std::invalid_argument("Tensor shapes must match for addition");
  if (lhs.dtype() != rhs.dtype()) throw std::runtime_error("DType mismatch in Tensor addition");

  DISPATCH_ANY_DTYPE(lhs.dtype(), T,
                     ops::add<T>(lhs.data_ptr(), rhs.data_ptr(), lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor &operator-=(Tensor &lhs, const Tensor &rhs) {
  if (lhs.shape() != rhs.shape())
    throw std::invalid_argument("Tensor shapes must match for subtraction");
  if (lhs.dtype() != rhs.dtype()) throw std::runtime_error("DType mismatch in Tensor subtraction");

  DISPATCH_ANY_DTYPE(lhs.dtype(), T,
                     ops::sub<T>(lhs.data_ptr(), rhs.data_ptr(), lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor &operator*=(Tensor &lhs, const Tensor &rhs) {
  if (lhs.shape() != rhs.shape())
    throw std::invalid_argument("Tensor shapes must match for multiplication");
  if (lhs.dtype() != rhs.dtype())
    throw std::runtime_error("DType mismatch in Tensor multiplication");

  DISPATCH_ANY_DTYPE(lhs.dtype(), T,
                     ops::mul<T>(lhs.data_ptr(), rhs.data_ptr(), lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor &operator/=(Tensor &lhs, const Tensor &rhs) {
  if (lhs.shape() != rhs.shape())
    throw std::invalid_argument("Tensor shapes must match for division");
  if (lhs.dtype() != rhs.dtype()) throw std::runtime_error("DType mismatch in Tensor division");

  DISPATCH_ANY_DTYPE(lhs.dtype(), T,
                     ops::div<T>(lhs.data_ptr(), rhs.data_ptr(), lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor operator+(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs;
  result += rhs;
  return result;
}

inline Tensor operator-(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs;
  result -= rhs;
  return result;
}

inline Tensor operator*(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs;
  result *= rhs;
  return result;
}

inline Tensor operator/(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs;
  result /= rhs;
  return result;
}

inline Tensor &operator+=(Tensor &lhs, double scalar) {
  DISPATCH_ANY_DTYPE(
      lhs.dtype(), T,
      ops::add_scalar<T>(lhs.data_ptr(), static_cast<T>(scalar), lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor &operator-=(Tensor &lhs, double scalar) {
  DISPATCH_ANY_DTYPE(
      lhs.dtype(), T,
      ops::sub_scalar<T>(lhs.data_ptr(), static_cast<T>(scalar), lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor &operator*=(Tensor &lhs, double scalar) {
  DISPATCH_ANY_DTYPE(
      lhs.dtype(), T,
      ops::mul_scalar<T>(lhs.data_ptr(), static_cast<T>(scalar), lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor &operator/=(Tensor &lhs, double scalar) {
  if (scalar == 0.0) throw std::invalid_argument("Division by zero");
  DISPATCH_ANY_DTYPE(
      lhs.dtype(), T,
      ops::div_scalar<T>(lhs.data_ptr(), static_cast<T>(scalar), lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor operator+(const Tensor &lhs, double scalar) {
  Tensor result = lhs;
  result += scalar;
  return result;
}

inline Tensor operator-(const Tensor &lhs, double scalar) {
  Tensor result = lhs;
  result -= scalar;
  return result;
}

inline Tensor operator*(const Tensor &lhs, double scalar) {
  Tensor result = lhs;
  result *= scalar;
  return result;
}

inline Tensor operator/(const Tensor &lhs, double scalar) {
  Tensor result = lhs;
  result /= scalar;
  return result;
}

inline Tensor operator+(double scalar, const Tensor &rhs) {
  Tensor result = rhs;
  result += scalar;
  return result;
}

inline Tensor operator*(double scalar, const Tensor &rhs) {
  Tensor result = rhs;
  result *= scalar;
  return result;
}
}  // namespace synet