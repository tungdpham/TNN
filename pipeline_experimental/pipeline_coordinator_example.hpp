#pragma once

#include "pipeline_stage.hpp"
#include "pipeline_communicator.hpp"
#include "task.hpp"
#include "../tensor/tensor.hpp"
#include <vector>
#include <memory>

namespace tpipeline {

template <typename T = float>
class PipelineCoordinator {
public:
    PipelineCoordinator() = default;
    
    // Add a stage to the pipeline
    void add_stage(std::unique_ptr<PipelineStage<T>> stage) {
        stages_.push_back(std::move(stage));
    }
    
    // Start all stages
    void start() {
        for (auto& stage : stages_) {
            stage->start();
        }
    }
    
    // Stop all stages
    void stop() {
        for (auto& stage : stages_) {
            stage->stop();
        }
    }
    
    // Process a batch by splitting it into microbatches and sending to first stage
    void process_batch(const Tensor<T>& batch, int num_microbatches = 4) {
        if (stages_.empty()) return;
        
        // Split batch into microbatches
        std::vector<Tensor<T>> microbatches = split_into_microbatches(batch, num_microbatches);
        
        // Send each microbatch to the first stage
        auto* first_stage_comm = stages_[0]->get_communicator();
        for (int i = 0; i < microbatches.size(); ++i) {
            tpipeline::Task<T> task(tpipeline::TaskType::Forward, microbatches[i], i);
            first_stage_comm->enqueue_task(task);
        }
    }
    
private:
    std::vector<std::unique_ptr<PipelineStage<T>>> stages_;
    
    // Helper function to split batch into microbatches
    std::vector<Tensor<T>> split_into_microbatches(const Tensor<T>& batch, int num_microbatches) {
        std::vector<Tensor<T>> microbatches;
        // Implementation depends on your Tensor class
        // This is a placeholder - you'll need to implement based on your Tensor API
        for (int i = 0; i < num_microbatches; ++i) {
            microbatches.push_back(batch); // Simplified - should actually split the batch
        }
        return microbatches;
    }
};

// Example factory function to create a pipeline
template <typename T = float>
std::unique_ptr<PipelineCoordinator<T>> create_example_pipeline() {
    auto coordinator = std::make_unique<PipelineCoordinator<T>>();
    
    // Create communicators for in-process pipeline (simplified)
    // In a real implementation, you'd properly chain these communicators
    
    // Stage 1
    auto stage1_model = std::make_unique<tnn::Sequential<T>>();
    auto stage1_comm = std::make_unique<InProcessPipelineCommunicator<T>>(nullptr, nullptr);
    auto stage1 = std::make_unique<InProcessPipelineStage<T>>(
        std::move(stage1_model), std::move(stage1_comm), "Stage1");
    
    // Stage 2
    auto stage2_model = std::make_unique<tnn::Sequential<T>>();
    auto stage2_comm = std::make_unique<InProcessPipelineCommunicator<T>>(nullptr, nullptr);
    auto stage2 = std::make_unique<InProcessPipelineStage<T>>(
        std::move(stage2_model), std::move(stage2_comm), "Stage2");
    
    coordinator->add_stage(std::move(stage1));
    coordinator->add_stage(std::move(stage2));
    
    return coordinator;
}

} // namespace tpipeline
