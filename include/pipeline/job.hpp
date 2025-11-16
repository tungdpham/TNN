/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "tensor/tensor.hpp"

namespace tnn {
enum JobType { FORWARD, BACKWARD };

template <typename T = float> struct Job {
  Tensor<T> data;
  size_t micro_batch_id;

  Job() = default;

  Job(const Tensor<T> &d, size_t mb_id) : data(d.clone()), micro_batch_id(mb_id) {}

  Job(const Job &other) : data(other.data.clone()), micro_batch_id(other.micro_batch_id) {}

  Job(Job &&other) noexcept : data(std::move(other.data)), micro_batch_id(other.micro_batch_id) {}

  Job &operator=(const Job &other) {
    if (this != &other) {
      data = other.data.clone();
      micro_batch_id = other.micro_batch_id;
    }
    return *this;
  }

  Job &operator=(Job &&other) noexcept {
    if (this != &other) {
      data = std::move(other.data);
      micro_batch_id = other.micro_batch_id;
    }
    return *this;
  }
};
} // namespace tnn