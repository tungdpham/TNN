#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include "layers/layers.hpp"
#include "tensor/tensor.hpp"

int main() {
    std::cout << "Testing SIMD-optimized Conv2D layer..." << std::endl;
    
    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    // Create a simple conv layer
    const size_t in_channels = 16;
    const size_t out_channels = 32;
    const size_t kernel_h = 3;
    const size_t kernel_w = 3;
    const size_t batch_size = 4;
    const size_t input_h = 32;
    const size_t input_w = 32;
    
    // Create conv layer
    auto conv_layer = Layers::conv2d<float>(in_channels, out_channels, 
                                           kernel_h, kernel_w, 1, 1, 1, 1);
    
    // Create input tensor
    Tensor<float> input(batch_size, in_channels, input_h, input_w);
    
    // Fill with random data
    for (size_t i = 0; i < input.size(); ++i) {
        input.data()[i] = dis(gen);
    }
    
    std::cout << "Input shape: [" << batch_size << ", " << in_channels 
              << ", " << input_h << ", " << input_w << "]" << std::endl;
    
    // Time the forward pass
    auto start = std::chrono::high_resolution_clock::now();
    
    const int num_iterations = 100;
    Tensor<float> output;
    
    for (int i = 0; i < num_iterations; ++i) {
        output = conv_layer->forward(input, 0);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Output shape: [" << output.batch_size() << ", " 
              << output.channels() << ", " << output.height() 
              << ", " << output.width() << "]" << std::endl;
    
    std::cout << "Average time per forward pass: " 
              << duration.count() / num_iterations << " microseconds" << std::endl;
    
    // Check for reasonable output values
    float min_val = output.data()[0];
    float max_val = output.data()[0];
    float sum = 0.0f;
    
    for (size_t i = 0; i < output.size(); ++i) {
        float val = output.data()[i];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
    }
    
    float mean = sum / output.size();
    
    std::cout << "Output statistics:" << std::endl;
    std::cout << "  Min: " << min_val << std::endl;
    std::cout << "  Max: " << max_val << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    
    // Check if output is reasonable (not NaN or infinite)
    bool valid = true;
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output.data()[i])) {
            valid = false;
            break;
        }
    }
    
    if (valid) {
        std::cout << "✓ SIMD optimization test PASSED - output values are valid!" << std::endl;
    } else {
        std::cout << "✗ SIMD optimization test FAILED - invalid output values detected!" << std::endl;
        return 1;
    }
    
    return 0;
}
