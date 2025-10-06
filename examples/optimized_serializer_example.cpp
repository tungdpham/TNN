/*
 * Example demonstrating the optimized binary serializer
 */

#include "pipeline/message.hpp"
#include "pipeline/optimized_binary_serializer.hpp"
#include "tensor/tensor.hpp"
#include <chrono>
#include <iostream>

using namespace tpipeline;

int main() {
  // Create a test message with a tensor task
  std::vector<size_t> shape = {3, 224, 224}; // Example image tensor
  Tensor<float> test_tensor(shape);

  // Fill with some test data
  float *data = test_tensor.data();
  for (size_t i = 0; i < test_tensor.size(); ++i) {
    data[i] = static_cast<float>(i % 256) / 255.0f;
  }

  Task<float> task(TaskType::FORWARD, test_tensor, 42);
  Message<float> original_message(CommandType::FORWARD_TASK, task);
  original_message.sender_id = "worker_1";
  original_message.recipient_id = "coordinator";
  original_message.sequence_number = 123;

  // Test serialization performance
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<uint8_t> serialized_data = BinarySerializer::serialize_message(original_message);

  auto serialize_end = std::chrono::high_resolution_clock::now();

  // Test deserialization
  Message<float> deserialized_message =
      BinarySerializer::deserialize_message<float>(serialized_data);

  auto end = std::chrono::high_resolution_clock::now();

  // Print results
  auto serialize_time =
      std::chrono::duration_cast<std::chrono::microseconds>(serialize_end - start);
  auto deserialize_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - serialize_end);
  auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "=== Optimized Binary Serializer Test ===" << std::endl;
  std::cout << "Original tensor size: " << test_tensor.size() << " elements" << std::endl;
  std::cout << "Serialized data size: " << serialized_data.size() << " bytes" << std::endl;
  std::cout << "Serialize time: " << serialize_time.count() << " μs" << std::endl;
  std::cout << "Deserialize time: " << deserialize_time.count() << " μs" << std::endl;
  std::cout << "Total time: " << total_time.count() << " μs" << std::endl;

  // Verify correctness
  bool correct = true;
  correct &= (deserialized_message.command_type == original_message.command_type);
  correct &= (deserialized_message.sender_id == original_message.sender_id);
  correct &= (deserialized_message.recipient_id == original_message.recipient_id);
  correct &= (deserialized_message.sequence_number == original_message.sequence_number);
  correct &= (deserialized_message.has_task() == original_message.has_task());

  if (deserialized_message.has_task()) {
    const auto &orig_task = original_message.get_task();
    const auto &deser_task = deserialized_message.get_task();

    correct &= (orig_task.type == deser_task.type);
    correct &= (orig_task.micro_batch_id == deser_task.micro_batch_id);
    correct &= (orig_task.data.shape() == deser_task.data.shape());
    correct &= (orig_task.data.size() == deser_task.data.size());

    // Check tensor data
    const float *orig_data = orig_task.data.data();
    const float *deser_data = deser_task.data.data();
    for (size_t i = 0; i < orig_task.data.size() && correct; ++i) {
      correct &= (std::abs(orig_data[i] - deser_data[i]) < 1e-6f);
    }
  }

  std::cout << "Correctness: " << (correct ? "PASSED" : "FAILED") << std::endl;

  // Test endianness handling
  std::cout << "\n=== Endianness Test ===" << std::endl;

  // Create a simple message header to test endianness detection
  MessageHeader header;
  std::cout << "System endianness marker: 0x" << std::hex << header.endianness_marker << std::dec
            << std::endl;
  std::cout << "Needs byte swap: " << (header.needs_byte_swap() ? "YES" : "NO") << std::endl;

  // Test parameter serialization
  std::cout << "\n=== Parameter Serialization Test ===" << std::endl;

  std::vector<Tensor<float>> parameters;
  parameters.emplace_back(std::vector<size_t>{10, 5}); // Weight matrix
  parameters.emplace_back(std::vector<size_t>{5});     // Bias vector

  // Fill with test data
  for (auto &param : parameters) {
    float *param_data = param.data();
    for (size_t i = 0; i < param.size(); ++i) {
      param_data[i] = static_cast<float>(i) * 0.1f;
    }
  }

  auto param_start = std::chrono::high_resolution_clock::now();
  auto serialized_params = BinarySerializer::serialize_parameters(parameters);
  auto param_serialize_end = std::chrono::high_resolution_clock::now();

  auto deserialized_params = BinarySerializer::deserialize_parameters<float>(serialized_params);
  auto param_end = std::chrono::high_resolution_clock::now();

  auto param_serialize_time =
      std::chrono::duration_cast<std::chrono::microseconds>(param_serialize_end - param_start);
  auto param_deserialize_time =
      std::chrono::duration_cast<std::chrono::microseconds>(param_end - param_serialize_end);

  std::cout << "Parameter count: " << parameters.size() << std::endl;
  std::cout << "Serialized params size: " << serialized_params.size() << " bytes" << std::endl;
  std::cout << "Param serialize time: " << param_serialize_time.count() << " μs" << std::endl;
  std::cout << "Param deserialize time: " << param_deserialize_time.count() << " μs" << std::endl;

  // Verify parameters
  bool param_correct = (deserialized_params.size() == parameters.size());
  for (size_t i = 0; i < parameters.size() && param_correct; ++i) {
    param_correct &= (parameters[i].shape() == deserialized_params[i].shape());
    param_correct &= (parameters[i].size() == deserialized_params[i].size());

    const float *orig = parameters[i].data();
    const float *deser = deserialized_params[i].data();
    for (size_t j = 0; j < parameters[i].size() && param_correct; ++j) {
      param_correct &= (std::abs(orig[j] - deser[j]) < 1e-6f);
    }
  }

  std::cout << "Parameter correctness: " << (param_correct ? "PASSED" : "FAILED") << std::endl;

  return 0;
}