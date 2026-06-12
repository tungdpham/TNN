/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/tensor_bundle.hpp"

namespace synet {
struct Job {
  TensorBundle data;
  size_t mb_id;

  Job()
      : data(),
        mb_id(0) {}

  Job(TensorBundle d, size_t mb_id)
      : data(std::move(d)),
        mb_id(mb_id) {}

  Job(const Job &other) = default;

  Job(Job &&other) noexcept
      : data(std::move(other.data)),
        mb_id(other.mb_id) {}

  Job &operator=(const Job &other) = default;

  Job &operator=(Job &&other) noexcept {
    if (this != &other) {
      data = std::move(other.data);
      mb_id = other.mb_id;
    }
    return *this;
  }
};

template <typename Archiver>
void archive(Archiver &archiver, const Job &job) {
  archiver(job.mb_id, job.data);
}

template <typename Archiver>
void archive(Archiver &archiver, Job &job) {
  archiver(job.mb_id, job.data);
}

}  // namespace synet