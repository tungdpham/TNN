/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "tensor/itensor.hpp"

namespace tnn {

class Tensor : public std::shared_ptr<ITensor> {
public:
  using std::shared_ptr<ITensor>::shared_ptr;

  Tensor() : std::shared_ptr<ITensor>() {}

  Tensor(const std::shared_ptr<ITensor> &ptr) : std::shared_ptr<ITensor>(ptr) {}

  Tensor(std::shared_ptr<ITensor> &&ptr) : std::shared_ptr<ITensor>(std::move(ptr)) {}
};

inline Tensor operator+(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs->clone();
  result->add(rhs);
  return result;
}

inline Tensor operator-(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs->clone();
  result->sub(rhs);
  return result;
}

inline Tensor operator*(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs->clone();
  result->mul(rhs);
  return result;
}

inline Tensor operator/(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs->clone();
  result->div(rhs);
  return result;
}

inline Tensor operator+(const Tensor &lhs, double scalar) {
  Tensor result = lhs->clone();
  result->add_scalar(scalar);
  return result;
}

inline Tensor operator-(const Tensor &lhs, double scalar) {
  Tensor result = lhs->clone();
  result->sub_scalar(scalar);
  return result;
}

inline Tensor operator*(const Tensor &lhs, double scalar) {
  Tensor result = lhs->clone();
  result->mul_scalar(scalar);
  return result;
}

inline Tensor operator/(const Tensor &lhs, double scalar) {
  Tensor result = lhs->clone();
  result->div_scalar(scalar);
  return result;
}

inline Tensor operator+(double scalar, const Tensor &rhs) {
  Tensor result = rhs->clone();
  result->add_scalar(scalar);
  return result;
}

inline Tensor operator*(double scalar, const Tensor &rhs) {
  Tensor result = rhs->clone();
  result->mul_scalar(scalar);
  return result;
}

}  // namespace tnn

#include "tensor/tensor_factory.hpp"
