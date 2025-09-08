#include "pipeline/network_serialization.hpp"
#include "tensor/tensor.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>

signed main() {
  Tensor<float> tensor(2, 2, 16, 16);
  tensor.fill_random_normal(0.0f, 1.0f);
  auto serialize_start = std::chrono::high_resolution_clock::now();
  auto serialized = tpipeline::BinarySerializer::serialize_tensor(tensor);
  auto serialize_end = std::chrono::high_resolution_clock::now();
  size_t offset = 0;
  auto deserialize_start = std::chrono::high_resolution_clock::now();
  Tensor<float> deserialized =
      tpipeline::BinarySerializer::deserialize_tensor<float>(serialized, offset);
  auto deserialize_end = std::chrono::high_resolution_clock::now();
  float* original_data = tensor.data();
  float* deserialized_data = deserialized.data();
  for (size_t i = 0; i < tensor.size(); ++i) {
    assert(original_data[i] == deserialized_data[i] && "Data mismatch after serialization");
  }
  std::cout << "Serialization and deserialization successful and data matches!" << std::endl;
  auto serialize_duration = std::chrono::duration_cast<std::chrono::microseconds>(serialize_end - serialize_start).count();
  auto deserialize_duration = std::chrono::duration_cast<std::chrono::microseconds>(deserialize_end - deserialize_start).count();
  std::cout << "Serialization time: " << serialize_duration << " microseconds" << std::endl;
  std::cout << "Deserialization time: " << deserialize_duration << " microseconds" << std::endl;
  return 0;
}