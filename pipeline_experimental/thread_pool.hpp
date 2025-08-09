#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

// For C++17 and later, we'll use std::apply
#if __cplusplus >= 201703L
#include <type_traits>
#endif

class ThreadPool {
public:
  ThreadPool(size_t threads);
  ~ThreadPool();

  // Enqueue a function with arguments and return a future for the result
  // WARNING: Avoid calling enqueue() from within a task that waits for another
  // task's completion, as this can lead to deadlocks if all worker threads
  // become blocked waiting for subtasks.
  template <class F, class... Args>
#if __cplusplus >= 201703L
  auto enqueue(F &&f,
               Args &&...args) -> std::future<std::invoke_result_t<F, Args...>>;
#else
  auto enqueue(F &&f, Args &&...args)
      -> std::future<decltype(std::declval<F>()(std::declval<Args>()...))>;
#endif

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;

  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
  for (size_t i = 0; i < threads; ++i) {
    workers.emplace_back([this] {
      for (;;) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->queue_mutex);
          this->condition.wait(
              lock, [this] { return this->stop || !this->tasks.empty(); });
          if (this->stop && this->tasks.empty())
            return;
          task = std::move(this->tasks.front());
          this->tasks.pop();
        }
        task();
      }
    });
  }
}

template <class F, class... Args>
#if __cplusplus >= 201703L
auto ThreadPool::enqueue(F &&f, Args &&...args)
    -> std::future<std::invoke_result_t<F, Args...>> {
  using return_type = std::invoke_result_t<F, Args...>;

  // Modern C++ lambda capture approach
  auto task = std::make_shared<std::packaged_task<return_type()>>(
      [f = std::forward<F>(f),
       args = std::make_tuple(std::forward<Args>(args)...)]() mutable {
        return std::apply(f, std::move(args));
      });
#else
auto ThreadPool::enqueue(F &&f, Args &&...args)
    -> std::future<decltype(std::declval<F>()(std::declval<Args>()...))> {
  using return_type = decltype(std::declval<F>()(std::declval<Args>()...));

  // Fallback to std::bind for pre-C++17
  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));
#endif

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    if (stop)
      throw std::runtime_error("enqueue on stopped ThreadPool");
    tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
  return res;
}

inline ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread &worker : workers) {
    worker.join();
  }
}
