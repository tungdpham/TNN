#pragma once

#include "device/device_manager.hpp"
#include "job.hpp"
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>

namespace tnn {

template <typename T> class JobPool;

template <typename T> class JobDeleter {
public:
  explicit JobDeleter(JobPool<T> *pool = nullptr) : pool_(pool) {}

  void operator()(Job<T> *ptr) const;

private:
  JobPool<T> *pool_;
};

template <typename T> using PooledJob = std::unique_ptr<Job<T>, JobDeleter<T>>;

template <typename T = float> class JobPool {
public:
  static constexpr size_t DEFAULT_JOB_CAPACITY = 1024;
  static constexpr size_t MAX_POOL_SIZE = 128;

  PooledJob<T> get_job(size_t min_capacity = DEFAULT_JOB_CAPACITY) {
    JobDeleter<T> deleter(this);

    if (is_shutting_down_.load(std::memory_order_relaxed)) {
      return PooledJob<T>(new Job<T>(), deleter);
    }

    {
      std::lock_guard<std::mutex> lock(pool_mutex_);

      auto it = pool_.begin();
      while (it != pool_.end()) {
        Job<T> *raw_job = *it;

        // Reuse jobs within 2x of requested size to reduce allocations
        if (raw_job->data.capacity() >= min_capacity) {
          pool_.erase(it);
          raw_job->micro_batch_id = 0;
          return PooledJob<T>(raw_job, deleter);
        }
        ++it;
      }
    }

    return PooledJob<T>(new Job<T>(Tensor<float>({min_capacity, 1, 1, 1}, &getCPU()), 0), deleter);
  }

  static JobPool &instance() {
    static JobPool pool;
    return pool;
  }

  ~JobPool() {
    is_shutting_down_.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (Job<T> *job : pool_) {
      delete job;
    }
    pool_.clear();
  }

  void return_job_internal(Job<T> *job) {
    if (job == nullptr) {
      return;
    }

    if (is_shutting_down_.load(std::memory_order_relaxed)) {
      delete job;
      return;
    }

    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (pool_.size() < MAX_POOL_SIZE) {
      pool_.push_back(job);
    } else {
      std::cerr << "JobPool is full. Discarding job with capacity: " << job->data.capacity()
                << std::endl;
      delete job;
    }
  }

  bool is_shutting_down() const { return is_shutting_down_.load(std::memory_order_relaxed); }

  size_t pool_size() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return pool_.size();
  }

private:
  std::atomic<bool> is_shutting_down_{false};
  std::deque<Job<T> *> pool_;
  mutable std::mutex pool_mutex_;
};

template <typename T> inline void JobDeleter<T>::operator()(Job<T> *ptr) const {
  if (pool_) {
    pool_->return_job_internal(ptr);
  } else {
    delete ptr;
  }
}

} // namespace tnn
