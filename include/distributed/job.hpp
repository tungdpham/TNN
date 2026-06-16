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
  size_t pid;

  Job()
      : data(),
        pid(0) {}

  Job(TensorBundle d, size_t pid)
      : data(std::move(d)),
        pid(pid) {}

  Job(const Job &other) = default;

  Job(Job &&other) noexcept
      : data(std::move(other.data)),
        pid(other.pid) {}

  Job &operator=(const Job &other) = default;

  Job &operator=(Job &&other) noexcept {
    if (this != &other) {
      data = std::move(other.data);
      pid = other.pid;
    }
    return *this;
  }
};

template <typename Archiver>
void archive(Archiver &archiver, const Job &job) {
  archiver(job.pid, job.data);
}

template <typename Archiver>
void archive(Archiver &archiver, Job &job) {
  archiver(job.pid, job.data);
}

}  // namespace synet