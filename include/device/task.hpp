#pragma once

#include "device/device_manager.hpp"
#include "flow.hpp"

#include <atomic>
#include <functional>
#include <iostream>
#include <mutex>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

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
private:
  std::function<void()> work_function_;
  CPUFlow *flow_;

public:
  template <typename Func, typename... Args>
  explicit CPUTask(CPUFlow *flow, Func &&func, Args &&...args) : flow_(flow) {
    auto bound_work = [f = std::forward<Func>(func),
                       args_tuple = std::tuple<Args...>(std::forward<Args>(args)...)]() mutable {
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
private:
  CUDAFlow *flow_;
  cudaEvent_t event_ = nullptr;
  std::function<void()> launch_function_;

  static inline std::vector<cudaEvent_t> event_pool_;
  static inline std::mutex pool_mutex_;

  static cudaEvent_t get_event() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (event_pool_.empty()) {
      cudaEvent_t e;
      cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
      return e;
    }
    cudaEvent_t e = event_pool_.back();
    event_pool_.pop_back();
    return e;
  }

  static void release_event(cudaEvent_t e) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    event_pool_.push_back(e);
  }

public:
  template <typename Func, typename... Args>
  explicit CUDATask(CUDAFlow *flow, Func &&func, Args &&...args) : flow_(flow) {
    // event_ = get_event();

    cudaStream_t stream = flow_->get_stream();

    // Automatically append cudaStream_t as the last parameter
    auto launch_func = [f = std::forward<Func>(func),
                        args_tuple = std::tuple<Args...>(std::forward<Args>(args)...),
                        stream]() mutable {
      std::apply(f, std::tuple_cat(std::move(args_tuple), std::make_tuple(stream)));
    };

    launch_function_ = std::move(launch_func);
    execute();
  }

  void execute() override {
    launch_function_();
    // cudaEventRecord(event_, flow_->get_stream());
  }

  ErrorStatus sync() override {
    if (ready()) {
      return status_;
    }

    // cudaError_t err = cudaStreamWaitEvent(flow_->get_stream(), event_, 0);
    cudaError_t err = cudaStreamSynchronize(flow_->get_stream());
    ErrorStatus status = cuda_error_to_status(err);
    set_ready_state(status);

    return status;
  }

  ~CUDATask() override {
    auto err = sync();
    if (err != ErrorStatus{}) {
      std::cerr << "Error in CUDATask sync in destructor: " << err.message() << std::endl;
      std::cerr << "You might want to capture task and call sync() explicitly to handle errors."
                << std::endl;
    }

    if (event_) {
      release_event(event_);
      event_ = nullptr;
    }
  }

  CUDATask(const CUDATask &) = delete;
  CUDATask &operator=(const CUDATask &) = delete;
  CUDATask(CUDATask &&) = delete;
  CUDATask &operator=(CUDATask &&) = delete;
};
#endif

template <typename Func, typename... Args>
std::unique_ptr<Task> create_cpu_task(std::string flow_id, Func &&func, Args &&...args) {
  auto CPUDevice = &getCPU();
  CPUFlow *flow = dynamic_cast<CPUFlow *>(CPUDevice->getFlow(flow_id));
  if (!flow) {
    throw std::runtime_error("Failed to get CPU flow with ID: " + flow_id);
  }
  return std::make_unique<CPUTask>(flow, std::forward<Func>(func), std::forward<Args>(args)...);
}

#ifdef USE_CUDA
template <typename Func, typename... Args>
std::unique_ptr<Task> create_gpu_task(std::string flow_id, Func &&func, Args &&...args) {
  auto GPUDevice = &getGPU();
  CUDAFlow *flow = dynamic_cast<CUDAFlow *>(GPUDevice->getFlow(flow_id));
  if (!flow) {
    throw std::runtime_error("Failed to get CUDA flow with ID: " + flow_id);
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

} // namespace tnn