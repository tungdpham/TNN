/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <atomic>
#include <iostream>
#include <system_error>
#include <tuple>
#include <utility>

#include "device/device_manager.hpp"
#include "flow.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tnn {

using ErrorStatus = std::error_code;

class TaskHandler;

class Task {
protected:
  std::atomic<bool> is_ready_ = false;
  ErrorStatus status_{};

public:
  virtual ~Task() {}

  bool ready() const { return is_ready_.load(std::memory_order_acquire); }

  virtual ErrorStatus sync() = 0;

protected:
  void set_ready_state(ErrorStatus status = {}) {
    status_ = status;
    is_ready_.store(true, std::memory_order_release);
  }
};

class CPUTask : public Task {
private:
  [[maybe_unused]] CPUFlow *flow_;

public:
  template <typename Func, typename... Args>
  explicit CPUTask(CPUFlow *flow, Func &&func, Args &&...args)
      : flow_(flow) {
    auto bound_work = [f = std::forward<Func>(func),
                       args_tuple = std::tuple<Args...>(std::forward<Args>(args)...)]() mutable {
      std::apply(f, std::move(args_tuple));
    };

    bound_work();
    set_ready_state(ErrorStatus{});
  }

  ~CPUTask() override = default;

  ErrorStatus sync() override { return this->status_; }

  CPUTask(const CPUTask &) = delete;
  CPUTask &operator=(const CPUTask &) = delete;
  CPUTask(CPUTask &&) = delete;
  CPUTask &operator=(CPUTask &&) = delete;
};

#ifdef USE_CUDA

inline tnn::ErrorStatus cuda_error_to_status(cudaError_t err) {
  if (err == cudaSuccess) {
    return tnn::ErrorStatus{};
  }
  std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
  return std::make_error_code(std::errc::resource_unavailable_try_again);
}

class CUDATask : public Task {
private:
  CUDAFlow *flow_;

public:
  template <typename Func, typename... Args>
  explicit CUDATask(CUDAFlow *flow, Func &&func, Args &&...args)
      : flow_(flow) {
    cudaStream_t stream = flow_->get_stream();

    auto launch_func = [f = std::forward<Func>(func),
                        args_tuple = std::tuple<Args...>(std::forward<Args>(args)...),
                        stream]() mutable {
      std::apply(f, std::tuple_cat(std::move(args_tuple), std::make_tuple(stream)));
    };

    launch_func();
  }

  ErrorStatus sync() override {
    if (ready()) {
      return status_;
    }

    cudaError_t err = cudaStreamSynchronize(flow_->get_stream());
    ErrorStatus status = cuda_error_to_status(err);
    set_ready_state(status);

    return status;
  }

  ~CUDATask() override {}

  CUDATask(const CUDATask &) = delete;
  CUDATask &operator=(const CUDATask &) = delete;
  CUDATask(CUDATask &&) = delete;
  CUDATask &operator=(CUDATask &&) = delete;
};
#endif

template <typename Func, typename... Args>
std::unique_ptr<Task> create_cpu_task(flowHandle_t handle, Func &&func, Args &&...args) {
  auto &CPUDevice = getHost();
  CPUFlow *flow = dynamic_cast<CPUFlow *>(CPUDevice.getFlow(handle));
  if (!flow) {
    throw std::runtime_error("Failed to get CPU flow with ID: " + std::to_string(handle.id));
  }
  return std::make_unique<CPUTask>(flow, std::forward<Func>(func), std::forward<Args>(args)...);
}

#ifdef USE_CUDA
// bundle the function and inject a stream based on the handle
template <typename Func, typename... Args>
std::unique_ptr<Task> create_cuda_task(flowHandle_t handle, Func &&func, Args &&...args) {
  auto &GPUDevice = getGPU();
  CUDAFlow *flow = dynamic_cast<CUDAFlow *>(GPUDevice.getFlow(handle));
  if (!flow) {
    throw std::runtime_error("Failed to get CUDA flow with ID: " + std::to_string(handle.id));
  }
  return std::make_unique<CUDATask>(flow, std::forward<Func>(func), std::forward<Args>(args)...);
}
#endif

inline void task_sync_all(const std::initializer_list<Task *> &tasks) {
  for (const auto &task : tasks) {
    if (task) {
      auto errorStatus = task->sync();
      if (errorStatus != ErrorStatus{}) {
        throw std::runtime_error("Task synchronization failed with error: " +
                                 errorStatus.message());
      }
    }
  }
}

}  // namespace tnn