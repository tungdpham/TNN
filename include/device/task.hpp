#pragma once

#include "device/device.hpp"

#include <atomic>
#include <functional>
#include <iostream>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>

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
  virtual ~Task() = default;

  bool ready() const { return is_ready_.load(std::memory_order_acquire); }

  virtual void execute() = 0;
  virtual ErrorStatus sync() = 0;

protected:
  void set_ready_state(ErrorStatus status = {}) {
    status_ = status;
    is_ready_.store(true, std::memory_order_release);
  }
};

class CPUTask : public Task {
  friend class TaskHandler;

private:
  std::function<void()> work_function_;
  const Device *device_;

public:
  template <typename Func, typename... Args>
  explicit CPUTask(const Device *device, Func &&func, Args &&...args) : device_(device) {
    auto bound_work = [f = std::forward<Func>(func),
                       args_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
      std::apply(f, std::move(args_tuple));
    };

    // Store the work function
    work_function_ = std::move(bound_work);
    execute();
  }

  ~CPUTask() override = default;

  void execute() override {
    work_function_();
    set_ready_state(ErrorStatus{});
  }

  ErrorStatus sync() override {
    while (!ready()) {
      std::cout << status_ << " " << is_ready_.load(std::memory_order_acquire) << std::endl;
      std::this_thread::yield();
    }
    return this->status_;
  }

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
  friend class TaskHandler;

private:
  cudaStream_t stream_ = nullptr;
  cudaEvent_t event_ = nullptr;
  const Device *device_;
  std::function<void()> launch_function_;

public:
  template <typename Func, typename... Args>
  explicit CUDATask(const Device *device, Func &&func, Args &&...args) : device_(device) {
    cudaStreamCreate(&stream_);
    cudaEventCreate(&event_);

    // Automatically append cudaStream_t as the last parameter
    // This eliminates the need for lambda wrapping in user code
    auto launch_func = [f = std::forward<Func>(func),
                        args_tuple = std::make_tuple(std::forward<Args>(args)...),
                        stream = stream_]() mutable {
      // Concatenate user args with stream parameter and apply to function
      std::apply(f, std::tuple_cat(std::move(args_tuple), std::make_tuple(stream)));
    };

    launch_function_ = std::move(launch_func);
    execute();
  }

  void execute() override {
    launch_function_();
    cudaEventRecord(event_, stream_);
  }

  ErrorStatus sync() override {
    if (ready()) {
      return status_;
    }

    cudaError_t err = cudaEventSynchronize(event_);
    ErrorStatus status = cuda_error_to_status(err);
    set_ready_state(status);

    return status;
  }

  ~CUDATask() override {
    if (stream_)
      cudaStreamDestroy(stream_);
    if (event_)
      cudaEventDestroy(event_);
  }

  CUDATask(const CUDATask &) = delete;
  CUDATask &operator=(const CUDATask &) = delete;
  CUDATask(CUDATask &&) = delete;
  CUDATask &operator=(CUDATask &&) = delete;
};
#endif
} // namespace tnn