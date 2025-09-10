#pragma once

#include "internal_compressor.hpp"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tpipeline {
static void internal_compress(const std::vector<uint8_t> &data,
                              const std::string name) {
  if (name == "zstd") {
    ZstdCompressor::compress(data);
  } else if (name == "lz4hc") {
    Lz4hcCompressor::compress(data);
  } else {
    throw new std::invalid_argument("Unsupported compression type: " + name);
  }
}

static void internal_decompress(const std::vector<uint8_t> &data,
                                const std::string name) {
  if (name == "zstd") {
    ZstdCompressor::decompress(data);
  } else if (name == "lz4hc") {
    Lz4hcCompressor::decompress(data);
  } else {
    throw new std::invalid_argument("Unsupported decompression type: " + name);
  }
}

class BloscCompressor {
public:
  static std::vector<uint8_t> compress(const std::vector<uint8_t> &data,
                                       int clevel = 5, int shuffle = 1);

  static std::vector<uint8_t> decompress(const std::vector<uint8_t> &data);
};
} // namespace tpipeline