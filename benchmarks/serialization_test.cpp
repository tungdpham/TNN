#include "pipeline/network_serialization.hpp"
#include "tensor/tensor.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>

signed main() {
  Tensor<float> tensor(16, 16, 1024, 1024);
  tensor.fill_random_normal(0.0f, 1.0f);
  [[maybe_unused]] float *original_data = tensor.data();
  auto naive_serialize_start = std::chrono::high_resolution_clock::now();
  tpipeline::TBuffer serialized;
  tpipeline::BinarySerializer::serialize(tensor, serialized);
  auto naive_serialize_end = std::chrono::high_resolution_clock::now();
  size_t offset = 0;
  auto naive_deserialize_start = std::chrono::high_resolution_clock::now();
  Tensor<float> deserialized;
  tpipeline::BinarySerializer::deserialize(serialized, offset, deserialized);
  auto naive_deserialize_end = std::chrono::high_resolution_clock::now();
  [[maybe_unused]] float *deserialized_data = deserialized.data();
  for (size_t i = 0; i < tensor.size(); ++i) {
    assert(original_data[i] == deserialized_data[i] && "Data mismatch after serialization");
  }
  std::cout << "Naive Serialization and deserialization successful and data matches!" << std::endl;
  auto serialize_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                                naive_serialize_end - naive_serialize_start)
                                .count();
  auto deserialize_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                                  naive_deserialize_end - naive_deserialize_start)
                                  .count();
  std::cout << "Naive Serialization time: " << serialize_duration << " microseconds" << std::endl;
  std::cout << "Naive Deserialization time: " << deserialize_duration << " microseconds"
            << std::endl;
  std::cout << "Naive Serialized size: " << serialized.size() << " bytes" << std::endl;

  // Size comparison
  std::cout << "Original tensor size: " << tensor.size() << " elements" << std::endl;
  std::cout << "Original tensor data: " << (tensor.size() * sizeof(float)) << " bytes" << std::endl;
  std::cout << "Binary serialized: " << serialized.size()
            << " bytes (overhead: " << (serialized.size() - tensor.size() * sizeof(float))
            << " bytes)" << std::endl;

  double binary_ratio = static_cast<double>(serialized.size()) / (tensor.size() * sizeof(float));
  std::cout << "Binary size ratio: " << binary_ratio << "x" << std::endl;
  return 0;
}