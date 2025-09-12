#include "pipeline/binary_serializer.hpp"
#include "pipeline/compression_impl/internal_compressor.tpp"
#include "tensor/tensor.hpp"
#include <iostream>
#include <vector>

int main() {
  Tensor<float, NCHW> tensor(32, 3, 1028, 1028);
  tensor.fill(0.5f);
  std::cout << "Tensor created with shape: " << tensor.batch_size() << "x"
            << tensor.channels() << "x" << tensor.height() << "x"
            << tensor.width() << std::endl;
  std::vector<uint8_t> serialized_data =
      tpipeline::BinarySerializer::serialize_tensor(tensor);
  auto compression_start = std::chrono::high_resolution_clock::now();
  std::vector<uint8_t> compressed_data =
      tpipeline::ZstdCompressor::compress(serialized_data, 3);
  auto compression_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> compression_duration =
      compression_end - compression_start;
  std::cout << "Compression took " << compression_duration.count() << " seconds"
            << std::endl;
  size_t original_size = serialized_data.size() * sizeof(uint8_t);
  size_t compressed_size = compressed_data.size() * sizeof(uint8_t);
  std::cout << "Original size: " << original_size << " bytes" << std::endl;
  std::cout << "Compressed size: " << compressed_size << " bytes" << std::endl;

  auto decompression_start = std::chrono::high_resolution_clock::now();
  std::vector<uint8_t> decompressed_data =
      tpipeline::ZstdCompressor::decompress(compressed_data);
  auto decompression_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> decompression_duration =
      decompression_end - decompression_start;
  std::cout << "Decompression took " << decompression_duration.count()
            << " seconds" << std::endl;
  size_t offset = 0;
  Tensor<float, NCHW> deserialized_tensor =
      tpipeline::BinarySerializer::deserialize_tensor<float>(decompressed_data,
                                                             offset);
  assert(deserialized_tensor.batch_size() == tensor.batch_size());
  assert(deserialized_tensor.channels() == tensor.channels());
  assert(deserialized_tensor.height() == tensor.height());
  assert(deserialized_tensor.width() == tensor.width());
  for (size_t i = 0; i < deserialized_tensor.size(); i++) {
    assert(tensor.data()[i] == deserialized_tensor.data()[i]);
  }
  std::cout << "Test passed: Deserialized tensor matches original tensor."
            << std::endl;
  return 0;
}