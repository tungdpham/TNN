#pragma once

#include "meta_compressor.hpp"

namespace tpipeline {
  std::vector<uint8_t> BloscCompressor::compress(const std::vector<uint8_t>& data, int clevel, int shuffle) {
    // Placeholder for compression logic
    return data; // No compression applied
  }

  std::vector<uint8_t> BloscCompressor::decompress(const std::vector<uint8_t>& data) {
    // Placeholder for decompression logic
    return data; // No decompression applied
  }
} // namespace tpipeline