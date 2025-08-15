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
#include <iostream>
// For C++17, we'll use std::apply and std::invoke_result_t
#include <type_traits>
#include <iostream>

// Platform-specific headers for CPU affinity
#ifdef __linux__
#include <pthread.h> // For pthread_setaffinity_np
#include <sched.h>   // For CPU_SET, sched_getaffinity, etc.
#endif

class ThreadPool {
public:
    ThreadPool(size_t threads);
    ~ThreadPool();

    template <class F, class... Args>
    auto enqueue(F &&f, Args &&...args) -> std::future<std::invoke_result_t<F, Args...>>;

    inline size_t size() const { return workers.size(); }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    size_t num_cores;
};

inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
    // Determine the number of available CPU cores
    num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) {
        num_cores = 1; // Fallback in case the call fails
    }
    if (threads > num_cores) {
        // Log or handle the case where more threads are requested than available cores
        // For this example, we'll just cap the number of threads.
        threads = num_cores;
    }
    std::cout << "Creating ThreadPool with " << threads << " threads" << std::endl;

    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this, i] {
#ifdef __linux__
            // Set CPU affinity for this worker thread
            size_t cpu_count = sched_getaffinity(0, sizeof(cpu_set_t), nullptr);
            size_t core_id = cpu_count - (i % cpu_count) - 1; // Wrap around if more threads than cores
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(core_id, &cpuset); // Pin to a core, wrapping around if needed
            std::cout << "Setting CPU affinity for thread " << i << " to core " << core_id << std::endl;

            if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
                // Error setting affinity, perhaps log this
                std::cerr << "Warning: Could not set CPU affinity for thread " << i << std::endl;
            }
#endif
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
auto ThreadPool::enqueue(F &&f, Args &&...args)
    -> std::future<std::invoke_result_t<F, Args...>> {
    using return_type = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        [f = std::forward<F>(f), args = std::make_tuple(std::forward<Args>(args)...)]() mutable {
            return std::apply(f, std::move(args));
        });

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

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        std::queue<std::function<void()>> empty;
        tasks.swap(empty);
    }
}