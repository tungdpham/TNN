/*
 * Batch-level asynchronous prefetcher for SYNET data loaders.
 *
 * This wraps an existing BaseDataLoader and overlaps get_batch(batch N+1)
 * with model compute/communication for batch N. It does not change dataset
 * semantics; it only moves get_batch calls to one background producer thread.
 */
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>

#include "data_loading/data_loader.hpp"
#include "tensor/tensor.hpp"

namespace synet {

class BatchPrefetcher {
public:
  struct Batch {
    Tensor data;
    Tensor labels;
    bool valid = false;
  };

  BatchPrefetcher(BaseDataLoader &loader, size_t batch_size, size_t prefetch_depth = 2)
      : loader_(loader),
        batch_size_(batch_size),
        prefetch_depth_(prefetch_depth > 0 ? prefetch_depth : 1) {}

  BatchPrefetcher(const BatchPrefetcher &) = delete;
  BatchPrefetcher &operator=(const BatchPrefetcher &) = delete;

  ~BatchPrefetcher() { stop(); }

  void start() {
    stop();
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_requested_ = false;
      finished_ = false;
      error_ = nullptr;
      queue_.clear();
    }
    worker_ = std::thread(&BatchPrefetcher::producer_loop, this);
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_requested_ = true;
    }
    cv_not_empty_.notify_all();
    cv_not_full_.notify_all();

    if (worker_.joinable()) {
      worker_.join();
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.clear();
      finished_ = true;
    }
  }

  bool next(Tensor &batch_data, Tensor &batch_labels) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_not_empty_.wait(lock,
                       [&] { return stop_requested_ || finished_ || !queue_.empty() || error_; });

    if (error_) {
      std::rethrow_exception(error_);
    }

    if (queue_.empty()) {
      return false;
    }

    Batch batch = std::move(queue_.front());
    queue_.pop_front();

    lock.unlock();
    cv_not_full_.notify_one();

    batch_data = std::move(batch.data);
    batch_labels = std::move(batch.labels);
    return batch.valid;
  }

private:
  void producer_loop() {
    try {
      while (true) {
        {
          std::unique_lock<std::mutex> lock(mu_);
          cv_not_full_.wait(lock,
                            [&] { return stop_requested_ || queue_.size() < prefetch_depth_; });

          if (stop_requested_) {
            finished_ = true;
            cv_not_empty_.notify_all();
            return;
          }
        }

        Batch batch;
        batch.valid = loader_.get_batch(batch_size_, batch.data, batch.labels);

        if (batch.valid) {
          batch.data = batch.data.clone();
          batch.labels = batch.labels.clone();
        }

        {
          std::lock_guard<std::mutex> lock(mu_);
          if (!batch.valid) {
            finished_ = true;
            cv_not_empty_.notify_all();
            return;
          }
          queue_.push_back(std::move(batch));
        }

        cv_not_empty_.notify_one();
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        error_ = std::current_exception();
        finished_ = true;
      }
      cv_not_empty_.notify_all();
    }
  }

  BaseDataLoader &loader_;
  size_t batch_size_;
  size_t prefetch_depth_;

  std::mutex mu_;
  std::condition_variable cv_not_empty_;
  std::condition_variable cv_not_full_;
  std::deque<Batch> queue_;
  std::thread worker_;

  bool stop_requested_ = false;
  bool finished_ = false;
  std::exception_ptr error_ = nullptr;
};

}  // namespace synet
