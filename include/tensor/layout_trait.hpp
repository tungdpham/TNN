/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <stdexcept>

namespace tnn {
enum Layout { NCHW, NHWC, NCDHW, NDHWC };

template <Layout L> struct LayoutTrait {
  inline void assign_shape(size_t *shape, size_t n, size_t c, size_t h, size_t w) {
    // Default implementation for unsupported layouts
    throw std::invalid_argument("Unsupported layout for assign_shape");
  }

  inline void compute_strides(size_t *strides, const size_t *shape) {
    // Default implementation for unsupported layouts
    throw std::invalid_argument("Unsupported layout for compute_strides");
  }
};

template <> struct LayoutTrait<NCHW> {
  static constexpr size_t dims = 4;
  size_t shape[dims];
  size_t strides[dims];

  inline void assign_shape(size_t n, size_t c, size_t h, size_t w) {
    shape[0] = n;
    shape[1] = c;
    shape[2] = h;
    shape[3] = w;
    compute_strides();
  }

  inline void compute_strides() {
    strides[0] = shape[1] * shape[2] * shape[3];
    strides[1] = shape[2] * shape[3];
    strides[2] = shape[3];
    strides[3] = 1;
  }

  inline size_t batch_size() const { return shape[0]; }
  inline size_t channels() const { return shape[1]; }
  inline size_t height() const { return shape[2]; }
  inline size_t width() const { return shape[3]; }
};

template <> struct LayoutTrait<NHWC> {
  static constexpr size_t dims = 4;
  size_t shape[dims];
  size_t strides[dims];

  inline void assign_shape(size_t n, size_t c, size_t h, size_t w) {
    shape[0] = n;
    shape[1] = h;
    shape[2] = w;
    shape[3] = c;
    compute_strides();
  }

  inline void compute_strides() {
    strides[0] = shape[1] * shape[2] * shape[3];
    strides[1] = shape[2] * shape[3];
    strides[2] = shape[3];
    strides[3] = 1;
  }

  inline size_t batch_size() const { return shape[0]; }
  inline size_t channels() const { return shape[3]; }
  inline size_t height() const { return shape[1]; }
  inline size_t width() const { return shape[2]; }
};

template <> struct LayoutTrait<NCDHW> {
  static constexpr size_t dims = 5;
  size_t shape[dims];
  size_t strides[dims];

  inline void assign_shape(size_t n, size_t c, size_t d, size_t h, size_t w) {
    shape[0] = n;
    shape[1] = c;
    shape[2] = d;
    shape[3] = h;
    shape[4] = w;
    compute_strides();
  }

  inline void compute_strides() {
    strides[0] = shape[1] * shape[2] * shape[3] * shape[4];
    strides[1] = shape[2] * shape[3] * shape[4];
    strides[2] = shape[3] * shape[4];
    strides[3] = shape[4];
    strides[4] = 1;
  }

  inline size_t batch_size() const { return shape[0]; }
  inline size_t channels() const { return shape[1]; }
  inline size_t depth() const { return shape[2]; }
  inline size_t height() const { return shape[3]; }
  inline size_t width() const { return shape[4]; }
};

template <> struct LayoutTrait<NDHWC> {
  static constexpr size_t dims = 5;
  size_t shape[dims];
  size_t strides[dims];

  inline void assign_shape(size_t n, size_t c, size_t d, size_t h, size_t w) {
    shape[0] = n;
    shape[1] = d;
    shape[2] = h;
    shape[3] = w;
    shape[4] = c;
    compute_strides();
  }

  inline void compute_strides() {
    strides[0] = shape[1] * shape[2] * shape[3] * shape[4];
    strides[1] = shape[2] * shape[3] * shape[4];
    strides[2] = shape[3] * shape[4];
    strides[3] = shape[4];
    strides[4] = 1;
  }

  inline size_t batch_size() const { return shape[0]; }
  inline size_t channels() const { return shape[4]; }
  inline size_t depth() const { return shape[1]; }
  inline size_t height() const { return shape[2]; }
  inline size_t width() const { return shape[3]; }
};

} // namespace tnn