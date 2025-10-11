#pragma once

#include "internal_compressor.hpp"
#include "pipeline/tbuffer.hpp"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tpipeline {
static void internal_compress(const TBuffer &data, const std::string name) {
  if (name == "zstd") {
    ZstdCompressor::compress(data);
  } else if (name == "lz4hc") {
    Lz4hcCompressor::compress(data);
  } else {
    throw new std::invalid_argument("Unsupported compression type: " + name);
  }
}

static void internal_decompress(const TBuffer &data, const std::string name) {
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
  static TBuffer compress(const TBuffer &data, int clevel = 5, int shuffle = 1);

  static TBuffer decompress(const TBuffer &data);
};
} // namespace tpipeline