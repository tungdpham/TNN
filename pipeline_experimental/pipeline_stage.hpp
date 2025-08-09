#include "../nn/sequential.hpp"
#include "pipeline_communicator.hpp"
#include "task.hpp"
#include "thread_pool.hpp"
#include <atomic>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <chrono>

namespace tpipeline {

template <typename T = float> class PipelineStage {
public:
  explicit PipelineStage(std::unique_ptr<tnn::Sequential<T>> model,
                        std::unique_ptr<PipelineCommunicator<T>> communicator,
                        const std::string &name = "")
      : model_(std::move(model)), communicator_(std::move(communicator)), name_(name), 
        should_stop_(false), thread_pool_(3) {} // 3 threads: receive, process, send

  virtual ~PipelineStage() {
    stop();
  }

  // Initialize and start the stage
  virtual void start() {
    should_stop_ = false;
    
    // Start the three parallel operations
    receive_future_ = thread_pool_.enqueue([this]() { receive_loop(); });
    process_future_ = thread_pool_.enqueue([this]() { process_loop(); });
    send_future_ = thread_pool_.enqueue([this]() { send_loop(); });
  }

  // Stop the stage
  virtual void stop() {
    should_stop_ = true;
    
    // Wait for all threads to complete
    if (receive_future_.valid()) receive_future_.wait();
    if (process_future_.valid()) process_future_.wait();
    if (send_future_.valid()) send_future_.wait();
  }

  // Get the name of the stage
  std::string name() const { return name_; }

  // Get the communicator (useful for coordinator to send tasks)
  PipelineCommunicator<T>* get_communicator() { return communicator_.get(); }

protected:
  // Continuous loop for receiving input tasks
  void receive_loop() {
    while (!should_stop_) {
      communicator_->receive_input_task();
      std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Small delay to prevent busy waiting
    }
  }

  // Continuous loop for processing tasks
  void process_loop() {
    while (!should_stop_) {
      if (communicator_->has_input_task() && !is_processing_) {
        is_processing_ = true;
        try {
          tpipeline::Task<T> task = communicator_->dequeue_input_task();
          process_task(task);
        } catch (const std::exception& e) {
          // Handle empty queue gracefully
        }
        is_processing_ = false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Continuous loop for sending output tasks
  void send_loop() {
    while (!should_stop_) {
      communicator_->send_output_task();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Process a single task (to be implemented by derived classes)
  virtual void process_task(const tpipeline::Task<T>& task) = 0;

protected:
  std::unique_ptr<tnn::Sequential<T>> model_;
  std::unique_ptr<PipelineCommunicator<T>> communicator_;
  std::string name_;
  
  std::atomic<bool> should_stop_;
  std::atomic<bool> is_processing_{false};
  
  ThreadPool thread_pool_;
  std::future<void> receive_future_;
  std::future<void> process_future_;
  std::future<void> send_future_;
};

template <typename T = float>
class InProcessPipelineStage : public PipelineStage<T> {
public:
  explicit InProcessPipelineStage(
      std::unique_ptr<tnn::Sequential<T>> model,
      std::unique_ptr<PipelineCommunicator<T>> communicator,
      const std::string &name = "")
      : PipelineStage<T>(std::move(model), std::move(communicator), name) {}

protected:
  void process_task(const tpipeline::Task<T>& task) override {
    // Forward or backward pass based on task type
    tpipeline::Task<T> output_task = task; // Copy task structure
    
    if (task.type == tpipeline::TaskType::Forward) {
      // Forward pass
      output_task.data = this->model_->forward(task.data);
    } else if (task.type == tpipeline::TaskType::Backward) {
      // Backward pass
      output_task.data = this->model_->backward(task.data);
    }
    
    // Enqueue the result for sending
    this->communicator_->enqueue_output_task(output_task);
  }
};

} // namespace tpipeline