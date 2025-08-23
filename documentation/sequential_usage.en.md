# Sequential Model Usage Guide

This guide covers how to use the `Sequential` class and `SequentialBuilder` to build and train neural network models.

## Table of Contents
- [Basic Sequential Class Usage](#basic-sequential-class-usage)
- [SequentialBuilder for Automatic Shape Inference](#sequentialbuilder-for-automatic-shape-inference)
- [Training Configuration](#training-configuration)
- [Model Training](#model-training)
- [Model Persistence](#model-persistence)
- [Performance Monitoring](#performance-monitoring-optional)

## Basic Sequential Class Usage

### Creating a Model
```cpp
// Create an empty sequential model
tnn::Sequential<float> model("my_model_name");

// Add layers manually
model.add(std::make_unique<DenseLayer<float>>(784, 128, "relu"));
model.add(std::make_unique<DenseLayer<float>>(128, 10, "linear"));
```

### Layer Management
```cpp
// Insert layer at specific position
model.insert(1, std::make_unique<DropoutLayer<float>>(0.5));

// Remove layer by index
model.remove(2);

// Access layers
Layer<float>& layer = model[0];  // or model.at(0)

// Get model size
size_t num_layers = model.size();
```

## SequentialBuilder for Automatic Shape Inference

The `SequentialBuilder` provides a fluent interface with automatic shape inference, eliminating the need to manually calculate input dimensions for each layer.

### Basic Usage Pattern
```cpp
auto model = tnn::SequentialBuilder<float>("model_name")
    .input({channels, height, width})  // Set input shape first
    .conv2d(out_channels, kernel_h, kernel_w, ...)  // Auto-infers input channels
    .maxpool2d(pool_h, pool_w, ...)
    .flatten()
    .dense(output_features, ...)  // Auto-infers input features
    .build();
```

### Layer Methods

#### Input Layer (Required First)
```cpp
.input({1, 28, 28})  // Shape without batch dimension: [channels, height, width]
```
**Parameters:**
- `shape`: Input dimensions excluding batch size

#### Convolutional Layers
```cpp
// Automatic input channel inference
.conv2d(out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, activation, use_bias, name)

// Manual specification (backward compatibility)
.conv2d(in_channels, out_channels, kernel_h, kernel_w, ...)
```
**Parameters:**
- `out_channels`: Number of output feature maps
- `kernel_h/w`: Kernel dimensions
- `stride_h/w`: Stride values (default: 1)
- `pad_h/w`: Padding values (default: 0)
- `activation`: Activation function ("relu", "elu", "linear", etc.)
- `use_bias`: Whether to use bias terms (default: true)
- `name`: Layer name (optional, auto-generated if empty)

#### Pooling Layers
```cpp
.maxpool2d(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, name)
```
**Parameters:**
- `pool_h/w`: Pooling window size
- `stride_h/w`: Stride values (default: same as pool size)
- `pad_h/w`: Padding values (default: 0)

#### Dense Layers
```cpp
// Automatic input feature inference
.dense(output_features, activation, use_bias, name)

// Manual specification
.dense(input_features, output_features, activation, use_bias, name)
```
**Parameters:**
- `output_features`: Number of output neurons
- `activation`: Activation function
- `use_bias`: Whether to use bias (default: true)

#### Utility Layers
```cpp
.flatten(name)                    // Flattens multi-dimensional input
.dropout(dropout_rate, name)      // Dropout regularization
.activation(activation_name, name) // Standalone activation layer
.batchnorm(epsilon, momentum, affine, name)  // Batch normalization
```

## Training Configuration

### Setting Optimizer
```cpp
// Create and set optimizer
auto optimizer = std::make_unique<tnn::Adam<float>>(
    learning_rate,    // e.g., 0.001f
    beta1,           // e.g., 0.9f (momentum term)
    beta2,           // e.g., 0.999f (RMSprop term)
    epsilon          // e.g., 1e-8f (numerical stability)
);
model.set_optimizer(std::move(optimizer));
```

### Setting Loss Function
```cpp
// Create and set loss function
auto loss = tnn::LossFactory<float>::create_crossentropy(epsilon); // e.g., 1e-15f
model.set_loss_function(std::move(loss));
```

### Training Mode Control
```cpp
model.train();  // Enable training mode (affects dropout, batchnorm)
model.eval();   // Enable evaluation mode
```

## Model Training

### Training Loop Example
```cpp
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    model.train();
    
    while (data_loader.get_next_batch(batch_data, batch_labels)) {
        // Forward pass
        Tensor<float> predictions = model.forward(batch_data);
        
        // Apply softmax for classification
        utils::apply_softmax<float>(predictions);
        
        // Compute loss (for monitoring)
        float loss = model.loss_function()->compute_loss(predictions, batch_labels);
        
        // Backward pass
        Tensor<float> loss_gradient = 
            model.loss_function()->compute_gradient(predictions, batch_labels);
        model.backward(loss_gradient);
        
        // Update parameters
        model.update_parameters();
    }
    
    // Validation phase
    model.eval();
    // ... validation code ...
}
```

## Model Persistence

### Saving Models
```cpp
// Save both architecture and weights
model.save_to_file("path/to/model");  // Creates .json and .bin files

// Save only configuration
model.save_config("config.json");
```

### Loading Models
```cpp
// Load complete model
auto loaded_model = tnn::Sequential<float>::from_file("path/to/model");

// Load from configuration
auto config_model = tnn::Sequential<float>::load_from_config_file("config.json");
```

## Performance Monitoring (Optional)

### Profiling
```cpp
// Enable performance profiling
model.enable_profiling(true);

// After training, view accumulated timing results
model.print_profiling_summary();

// Clear profiling data
model.clear_profiling_data();
```

### Model Information
```cpp
// Print model architecture summary
model.print_summary({batch_size, channels, height, width});

// Get parameter count
size_t total_params = model.parameter_count();

// Print model configuration
model.print_config();
```

### Learning Rate Scheduling
```cpp
// Get current learning rate
float current_lr = model.optimizer()->get_learning_rate();

// Update learning rate
model.optimizer()->set_learning_rate(new_lr);
```

## Complete Example

```cpp
#include "nn/sequential.hpp"
#include "nn/optimizers.hpp"
#include "nn/loss.hpp"

// Build model with automatic shape inference
auto model = tnn::SequentialBuilder<float>("mnist_classifier")
    .input({1, 28, 28})                           // 1 channel, 28x28 image
    .conv2d(32, 5, 5, 1, 1, 0, 0, "relu")        // 32 filters, 5x5 kernel, 1x1 stride, 0x0 padding
    .maxpool2d(2, 2, 2, 2, 0, 0)                 // 2x2 pool, 2x2 stride, 0x0 padding
    .conv2d(64, 5, 5, 1, 1, 0, 0, "relu")        // 64 filters, 5x5 kernel, 1x1 stride, 0x0 padding
    .maxpool2d(2, 2, 2, 2)                       // 2x2 pooling, 2x2 stride, 0x0 padding
    .flatten()                                    // Flatten for dense layers
    .dense(128, "relu")                           // 128 output features
    .dropout(0.5)                                 // 50% dropout 
    .dense(10, "linear")                          // 10 output features, linear activation
    .build();

// Configure training
auto optimizer = std::make_unique<tnn::Adam<float>>(0.001f, 0.9f, 0.999f, 1e-8f);
model.set_optimizer(std::move(optimizer));

auto loss = tnn::LossFactory<float>::create_crossentropy(1e-15f);
model.set_loss_function(std::move(loss));

// Enable profiling (optional)
model.enable_profiling(true);

// Training loop (simplified)
for (int epoch = 0; epoch < epochs; ++epoch) {
    // ... training code as shown above ...
}

// Save trained model
model.save_to_file("trained_model");
```

## Key Benefits

1. **Automatic Shape Inference**: No need to manually calculate dimensions between layers
2. **Fluent Interface**: Chain layer additions with method chaining
3. **Type Safety**: Template-based design ensures type consistency
4. **Flexible Configuration**: Support for various optimizers and loss functions
5. **Performance Monitoring**: Built-in profiling and debugging capabilities
6. **Model Persistence**: Easy saving and loading of complete models

This design makes building and training neural networks more intuitive while maintaining the flexibility needed for complex architectures.