#pragma once

#include "internal_compressor.hpp"

namespace tpipeline {
  std::vector<uint8_t> ZstdCompressor::compress(const std::vector<uint8_t>& data, int compression_level) {
    // Placeholder for compression logic
    return data; // No compression applied
  }

  std::vector<uint8_t> ZstdCompressor::decompress(const std::vector<uint8_t>& data) {
    // Placeholder for decompression logic
    return data; // No decompression applied
  }

  std::vector<uint8_t> Lz4hcCompressor::compress(const std::vector<uint8_t>& data, int compression_level) {
    // Placeholder for compression logic
    return data; // No compression applied
  }

  std::vector<uint8_t> Lz4hcCompressor::decompress(const std::vector<uint8_t>& data) {
    // Placeholder for decompression logic
    return data; // No decompression applied
  }
} // namespace tpipeline