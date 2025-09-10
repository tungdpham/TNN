/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

enum Layout { NCHW, NHWC, NCDHW, NDHWC };

template <typename T, Layout L> struct TensorView;

template <typename T> struct TensorView<T, NCHW> {
  static constexpr size_t dims = 4;

  inline static void compute_strides(size_t *strides, const size_t *shape) {
    strides[0] = shape[1] * shape[2] * shape[3];
    strides[1] = shape[2] * shape[3];
    strides[2] = shape[3];
    strides[3] = 1;
  }
};

template <typename T> struct TensorView<T, NHWC> {
  static constexpr size_t dims = 4;

  inline static void compute_strides(size_t *strides, const size_t *shape) {
    strides[0] = shape[1] * shape[2] * shape[3];
    strides[1] = 1;
    strides[2] = shape[1] * shape[3];
    strides[3] = shape[1];
  }
};

template <typename T> struct TensorView<T, NCDHW> {
  static constexpr size_t dims = 5;

  inline static void compute_strides(size_t *strides, const size_t *shape) {
    strides[0] = shape[1] * shape[2] * shape[3] * shape[4];
    strides[1] = shape[2] * shape[3] * shape[4];
    strides[2] = shape[3] * shape[4];
    strides[3] = shape[4];
    strides[4] = 1;
  }
};

template <typename T> struct TensorView<T, NDHWC> {
  static constexpr size_t dims = 5;

  inline static void compute_strides(size_t *strides, const size_t *shape) {
    strides[0] = shape[1] * shape[2] * shape[3] * shape[4];
    strides[1] = 1;
    strides[2] = shape[1] * shape[3] * shape[4];
    strides[3] = shape[1] * shape[4];
    strides[4] = shape[1];
  }
};
