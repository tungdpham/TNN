#pragma once

#include <memory>
#include "../tensor/tensor.hpp"

namespace pipeline {

// A simple abstract class for communication between pipeline stages.
// This could be implemented with MPI for multi-node or a simple
// queue for intra-process communication.
template <typename T>
class Communicator {
public:
    virtual ~Communicator() = default;
    virtual void send(const Tensor<T>& tensor, int destination_rank) = 0;
    virtual Tensor<T> receive(int source_rank) = 0;
};

// A simple in-process communicator for demonstration purposes.
// It uses a blocking queue to transfer tensors between threads.
template <typename T>
class InProcessCommunicator : public Communicator<T> {
public:
    void send(const Tensor<T>& tensor, int destination_rank) override {
        // In a real scenario, this would involve serialization and
        // sending data over a network or through CUDA IPC.
        // For this example, we'll just use a shared queue.
        std::lock_guard<std::mutex> lock(queue_mutex_);
        message_queue_.push(tensor);
        cv_.notify_one();
    }

    Tensor<T> receive(int source_rank) override {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        cv_.wait(lock, [this]{ return !message_queue_.empty(); });
        Tensor<T> tensor = message_queue_.front();
        message_queue_.pop();
        return tensor;
    }

private:
    std::queue<Tensor<T>> message_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
};

} // namespace pipeline
