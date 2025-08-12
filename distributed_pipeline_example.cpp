#include "pipeline/distributed_coordinator.hpp"
#include "nn/sequential.hpp"
#include "nn/layers.hpp"
#include "tensor/tensor.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace tnn;
using namespace tpipeline;

namespace mnist_constants {
    constexpr float LR_INITIAL = 0.001f;
    constexpr float EPSILON = 1e-15f;
}

// Create a simple CNN model for demonstration
Sequential<float> create_demo_model() {
    SequentialBuilder<float> builder("distributed_cnn");
    
    // Simple CNN for CIFAR-10 like data (32x32x3)
    builder.conv2d(3, 32, 3, 3, 1, 1, 1, 1, "relu")    // 32x32x32
           .maxpool2d(2, 2, 2, 2)                        // 16x16x32
           .conv2d(32, 64, 3, 3, 1, 1, 1, 1, "relu")     // 16x16x64
           .maxpool2d(2, 2, 2, 2)                        // 8x8x64
           .flatten()                                     // 4096
           .dense(4096, 512, "relu")                      // 512
           .dense(512, 10, "none");                       // 10 (classes)
    
    Sequential<float> model = builder.build();
    auto optimizer = std::make_unique<tnn::Adam<float>>(
        mnist_constants::LR_INITIAL, 0.9f, 0.999f, 1e-8f);
    model.set_optimizer(std::move(optimizer));
    
    // Set loss function for the model
    auto loss_function = tnn::LossFactory<float>::create_crossentropy(mnist_constants::EPSILON);
    model.set_loss(std::move(loss_function));
    return model;
}

int main() {
    try {
        std::cout << "=== Distributed Pipeline Example ===" << std::endl;
        
        // Create the model
        auto model = create_demo_model();
        model.print_summary({32, 32, 32, 3});
        
        // Define remote endpoints where stages will be deployed
        std::vector<DistributedPipelineCoordinator<float>::RemoteEndpoint> endpoints = {
            {"localhost", 8001, "stage_0"},  // First stage
            {"localhost", 8002, "stage_1"},  // Second stage  
            {"localhost", 8003, "stage_2"},  // Third stage
            {"localhost", 8004, "stage_3"}   // Fourth stage
        };
        
        std::cout << "\nConfigured " << endpoints.size() << " remote endpoints:" << std::endl;
        for (const auto& ep : endpoints) {
            std::cout << "  " << ep.stage_id << " -> " << ep.host << ":" << ep.port << std::endl;
        }
        
        // Create distributed coordinator
        std::cout << "\nCreating distributed coordinator..." << std::endl;
        DistributedPipelineCoordinator<float> coordinator(
            std::move(model), endpoints, 4, "localhost", 8000);
        
        // Deploy stages to remote machines
        std::cout << "\nDeploying stages to remote endpoints..." << std::endl;
        std::cout << "NOTE: Make sure network workers are running on the specified ports:" << std::endl;
        for (const auto& ep : endpoints) {
            std::cout << "  ./network_worker " << ep.port << " &" << std::endl;
        }
        std::cout << std::endl;
        
        // Give user time to start workers
        std::cout << "Waiting 5 seconds for workers to start..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        if (!coordinator.deploy_stages()) {
            std::cerr << "Failed to deploy stages. Make sure workers are running." << std::endl;
            return 1;
        }
        
        // Start the pipeline
        std::cout << "\nStarting distributed pipeline..." << std::endl;
        coordinator.start();
        
        // Create dummy input data (batch of 4 images, 32x32x3)
        std::cout << "\nCreating dummy input batch..." << std::endl;
        Tensor<float> input_batch({4, 3, 32, 32});
        input_batch.fill_random_uniform(1.0f);
        
        std::cout << "Input batch shape: ";
        for (size_t dim : input_batch.shape()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        // Perform forward pass
        std::cout << "\nPerforming forward pass..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        coordinator.forward(input_batch);
        coordinator.join(true); // Wait for forward pass completion
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Forward pass completed in " << duration.count() << " ms" << std::endl;
        
        // Get output from the final stage
        std::cout << "\nRetrieving results..." << std::endl;
        auto task_messages = coordinator.get_task_messages();
        
        std::cout << "Received " << task_messages.size() << " task messages" << std::endl;
        for (const auto& msg : task_messages) {
            if (msg.task.has_value()) {
                const auto& task = msg.task.value();
                std::cout << "  Task " << task.micro_batch_id << ": output shape ";
                for (size_t dim : task.data.shape()) {
                    std::cout << dim << " ";
                }
                std::cout << std::endl;
            }
        }
        
        // Create dummy gradients for backward pass
        std::cout << "\nCreating dummy gradients for backward pass..." << std::endl;
        std::vector<Tensor<float>> gradients;
        for (int i = 0; i < 4; ++i) {
            Tensor<float> grad({1, 3, 32, 32}); // Gradient w.r.t. 10 output classes
            grad.fill_random_uniform(0.1f);
            gradients.push_back(grad);
        }
        
        // Perform backward pass
        std::cout << "\nPerforming backward pass..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        coordinator.backward(gradients);
        coordinator.join(false); // Wait for backward pass completion
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Backward pass completed in " << duration.count() << " ms" << std::endl;
        
        // Update parameters
        std::cout << "\nUpdating parameters across all stages..." << std::endl;
        coordinator.update_parameters();
        
        // Stop the pipeline
        std::cout << "\nStopping distributed pipeline..." << std::endl;
        coordinator.stop();
        
        std::cout << "\n=== Distributed Pipeline Example Completed ===" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
