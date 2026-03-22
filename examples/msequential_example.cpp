/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

/**
 * MSequential Block Example
 *
 * This example demonstrates the usage of the MSequential block,
 * which implements the Multi-Input Single-Output (MISO) architecture
 * from Section 3.1.2 of the paper.
 */

#include <iostream>
#include <memory>
#include <vector>

#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/blocks_impl/msequential.hpp"
#include "nn/blocks_impl/sequential.hpp"
#include "nn/layer.hpp"
#include "nn/layer_builder.hpp"
#include "nn/layers.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

/**
 * Simple Mock Join Layer for demonstration purposes.
 * This layer takes multiple inputs and sums them element-wise.
 * In practice, you would use a proper Concat or other join layer.
 */
class MockJoinLayer : public Layer {
private:
  size_t num_inputs_;
  std::unordered_map<size_t, Vec<Vec<size_t>>> input_shapes_cache_;

public:
  explicit MockJoinLayer(size_t num_inputs, const std::string &name = "mock_join")
      : num_inputs_(num_inputs) {
    this->name_ = name;
  }

  static constexpr const char *TYPE_NAME = "mock_join";

  std::string type() const override { return TYPE_NAME; }

  void forward(const Vec<ConstTensor> &inputs, const Vec<Tensor> &outputs,
               size_t mb_id = 0) override {
    if (inputs.size() != num_inputs_) {
      throw std::runtime_error("MockJoinLayer: Expected " + std::to_string(num_inputs_) +
                               " inputs, got " + std::to_string(inputs.size()));
    }

    // Cache input shapes for backward
    Vec<Vec<size_t>> shapes;
    for (const auto &input : inputs) {
      shapes.push_back(input->shape());
    }
    input_shapes_cache_[mb_id] = shapes;

    // Simple element-wise sum (all inputs must have same shape)
    const Vec<size_t> &output_shape = inputs[0]->shape();
    outputs[0]->ensure(output_shape);

    // Sum all inputs
    outputs[0]->fill(0.0f);
    for (const auto &input : inputs) {
      DISPATCH_DTYPE(this->io_dtype_, T, {
        T *out_data = outputs[0]->data_as<T>();
        const T *in_data = input->data_as<T>();
        for (size_t i = 0; i < outputs[0]->size(); ++i) {
          out_data[i] += in_data[i];
        }
      });
    }
  }

  void backward(const Vec<ConstTensor> &grad_outputs, const Vec<Tensor> &grad_inputs,
                size_t mb_id = 0) override {
    // For element-wise sum, gradient distributes equally to all inputs
    for (auto &grad_input : grad_inputs) {
      grad_outputs[0]->copy_to(grad_input);
    }
  }

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override {
    if (input_shapes.empty()) {
      throw std::runtime_error("MockJoinLayer: At least one input shape required");
    }
    // Output shape is same as input shape (element-wise operation)
    return {input_shapes[0]};
  }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = name_;
    config.type = TYPE_NAME;
    config.set("num_inputs", num_inputs_);
    return config;
  }

  static std::unique_ptr<MockJoinLayer> create_from_config(const LayerConfig &config) {
    size_t num_inputs = config.get<size_t>("num_inputs", 2);
    return std::make_unique<MockJoinLayer>(num_inputs, config.name);
  }
};

void example_msequential_basic() {
  std::cout << "MSequential Basic Example\n";

  // Initialize device
  initializeDefaultDevices();
  const Device &device = getHost();
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);

  // Create two branches with simple dense layers
  Vec<std::unique_ptr<Sequential>> sequences;

  // Branch 1: Dense -> ReLU
  {
    auto layers = LayerBuilder({10})
                      .dense(8, true, "branch1_dense")
                      .activation("relu", "branch1_relu")
                      .build();
    sequences.push_back(std::make_unique<Sequential>(std::move(layers), "branch_1"));
  }

  // Branch 2: Dense -> ReLU
  {
    auto layers = LayerBuilder({10})
                      .dense(8, true, "branch2_dense")
                      .activation("relu", "branch2_relu")
                      .build();
    sequences.push_back(std::make_unique<Sequential>(std::move(layers), "branch_2"));
  }

  // Create join layer (element-wise sum)
  auto join_layer = std::make_unique<MockJoinLayer>(2, "sum_join");

  // Create MSequential
  auto mseq = std::make_unique<MSequential>(std::move(sequences), std::move(join_layer),
                                            "two_branch_model");

  // Initialize
  mseq->set_allocator(allocator);
  mseq->set_flow_handle(defaultFlowHandle);
  mseq->set_io_dtype(DType_t::FP32);
  mseq->init();

  // Print summary
  Vec<Vec<size_t>> input_shapes = {{2, 10}, {2, 10}};  // batch_size=2, features=10
  mseq->print_summary(input_shapes);

  // Create input tensors
  Tensor input1 = make_tensor<float>({2, 10}, device);
  Tensor input2 = make_tensor<float>({2, 10}, device);
  input1->fill(1.0f);
  input2->fill(2.0f);

  // Create output tensor
  Vec<size_t> output_shape = mseq->output_shapes(input_shapes)[0];
  Tensor output = make_tensor<float>(output_shape, device);

  // Forward pass
  std::cout << "Forward Pass";
  Vec<ConstTensor> inputs = {input1, input2};
  Vec<Tensor> outputs = {output};
  mseq->forward(inputs, outputs, 0);

  std::cout << "Output shape: (";
  for (size_t i = 0; i < output->shape().size(); ++i) {
    if (i > 0) std::cout << ",";
    std::cout << output->shape()[i];
  }
  std::cout << ")\n";

  // Print some output values
  const float *out_data = output->data_as<float>();
  std::cout << "First 5 output values: ";
  for (size_t i = 0; i < std::min<size_t>(5, output->size()); ++i) {
    std::cout << out_data[i] << " ";
  }
  std::cout << "\n";

  // Backward pass
  std::cout << "Backward Pass";
  Tensor grad_output = make_tensor<float>(output->shape(), device);
  grad_output->fill(1.0f);

  Tensor grad_input1 = make_tensor<float>(input1->shape(), device);
  Tensor grad_input2 = make_tensor<float>(input2->shape(), device);

  Vec<ConstTensor> grad_outs = {grad_output};
  Vec<Tensor> grad_ins = {grad_input1, grad_input2};
  mseq->backward(grad_outs, grad_ins, 0);

  std::cout << "Backward pass completed successfully.\n";

  std::cout << "Example Complete";
}

void example_msequential_heterogeneous() {
  std::cout << "MSequential Heterogeneous Branches Example\n";

  initializeDefaultDevices();
  const Device &device = getHost();
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);

  // Create three branches with different depths
  Vec<std::unique_ptr<Sequential>> sequences;

  // Branch 1: Shallow network (low M_b - O)
  {
    auto layers = LayerBuilder({100}).dense(50, true, "branch1_dense").build();
    sequences.push_back(std::make_unique<Sequential>(std::move(layers), "shallow_branch"));
  }

  // Branch 2: Medium network
  {
    auto layers = LayerBuilder({100})
                      .dense(200, true, "branch2_dense1")
                      .activation("relu")
                      .dense(50, true, "branch2_dense2")
                      .build();
    sequences.push_back(std::make_unique<Sequential>(std::move(layers), "medium_branch"));
  }

  // Branch 3: Deep network (high M_b - O)
  {
    auto layers = LayerBuilder({100})
                      .dense(300, true, "branch3_dense1")
                      .activation("relu")
                      .dense(200, true, "branch3_dense2")
                      .activation("relu")
                      .dense(50, true, "branch3_dense3")
                      .build();
    sequences.push_back(std::make_unique<Sequential>(std::move(layers), "deep_branch"));
  }

  // Join layer
  auto join_layer = std::make_unique<MockJoinLayer>(3, "sum_join");

  // Create MSequential
  auto mseq = std::make_unique<MSequential>(std::move(sequences), std::move(join_layer),
                                            "heterogeneous_model");

  mseq->set_allocator(allocator);
  mseq->set_flow_handle(defaultFlowHandle);
  mseq->set_io_dtype(DType_t::FP32);
  mseq->init();

  // Print summary - note the optimal execution order
  Vec<Vec<size_t>> input_shapes = {{4, 100}, {4, 100}, {4, 100}};
  mseq->print_summary(input_shapes);
}

int main(int argc, char **argv) {
  try {
    // Run examples
    example_msequential_basic();
    example_msequential_heterogeneous();

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
