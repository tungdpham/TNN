#pragma once

#include <future>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <map>
#include "../tensor/tensor.hpp"
#include "thread_pool.hpp"

namespace pipeline {

// A simple abstract class for communication between pipeline stages.
// This could be implemented with MPI for multi-node or a simple
// queue for intra-process communication.
template <typename T>
class Communicator {
public:
    virtual ~Communicator() = default;
    virtual void send(std::future<Tensor<T>> future, int source_stage_idx, int micro_batch_id) = 0;
    virtual Tensor<T> receive(int source_stage_idx, int micro_batch_id) = 0;
    virtual void send_grad(std::future<Tensor<T>> future, int dest_stage_idx, int micro_batch_id) = 0;
    virtual Tensor<T> receive_grad(int dest_stage_idx, int micro_batch_id) = 0;
};

// A simple in-process communicator for demonstration purposes.
// It uses a blocking queue to transfer tensors between threads.
template <typename T>
class InProcessCommunicator : public Communicator<T> {
private:
    using Queue = std::queue<Tensor<T>>;
    using QueueMap = std::map<int, Queue>;

    std::mutex forward_mutex_;
    std::map<int, QueueMap> forward_queues_; // Key: micro_batch_id, Value: map of stage_idx to queue
    std::condition_variable forward_cv_;

    std::mutex backward_mutex_;
    std::map<int, QueueMap> backward_queues_; // Key: micro_batch_id, Value: map of stage_idx to queue
    std::condition_variable backward_cv_;
    
    ThreadPool thread_pool_; // Reuse threads instead of creating new ones

public:
    InProcessCommunicator(size_t num_threads = 4) : thread_pool_(num_threads) {}
    void send(std::future<Tensor<T>> future, int source_stage_idx, int micro_batch_id) override {
        thread_pool_.enqueue([this, f = std::move(future), source_stage_idx, micro_batch_id]() mutable {
            Tensor<T> tensor = f.get();
            {
                std::lock_guard<std::mutex> lock(forward_mutex_);
                forward_queues_[micro_batch_id][source_stage_idx].push(std::move(tensor));
            }
            forward_cv_.notify_all();
        });
    }

    Tensor<T> receive(int source_stage_idx, int micro_batch_id) override {
        std::unique_lock<std::mutex> lock(forward_mutex_);
        forward_cv_.wait(lock, [&] {
            return forward_queues_.count(micro_batch_id) &&
                   forward_queues_[micro_batch_id].count(source_stage_idx) &&
                   !forward_queues_[micro_batch_id][source_stage_idx].empty();
        });
        Tensor<T> tensor = std::move(forward_queues_[micro_batch_id][source_stage_idx].front());
        forward_queues_[micro_batch_id][source_stage_idx].pop();
        return tensor;
    }

    void send_grad(std::future<Tensor<T>> future, int dest_stage_idx, int micro_batch_id) override {
        thread_pool_.enqueue([this, f = std::move(future), dest_stage_idx, micro_batch_id]() mutable {
            Tensor<T> tensor = f.get();
            {
                std::lock_guard<std::mutex> lock(backward_mutex_);
                backward_queues_[micro_batch_id][dest_stage_idx].push(std::move(tensor));
            }
            backward_cv_.notify_all();
        });
    }

    Tensor<T> receive_grad(int dest_stage_idx, int micro_batch_id) override {
        std::unique_lock<std::mutex> lock(backward_mutex_);
        backward_cv_.wait(lock, [&] {
            return backward_queues_.count(micro_batch_id) &&
                   backward_queues_[micro_batch_id].count(dest_stage_idx) &&
                   !backward_queues_[micro_batch_id][dest_stage_idx].empty();
        });
        Tensor<T> tensor = std::move(backward_queues_[micro_batch_id][dest_stage_idx].front());
        backward_queues_[micro_batch_id][dest_stage_idx].pop();
        return tensor;
    }
};

} // namespace pipeline
