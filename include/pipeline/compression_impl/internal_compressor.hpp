#pragma once
#include "pipeline/tbuffer.hpp"
#include <cstdint>

namespace tnn {
class ZstdCompressor {
public:
  static void compress(const TBuffer &input, TBuffer &output, int compression_level = 3);
  static void decompress(const TBuffer &input, TBuffer &output);
};

class Lz4hcCompressor {
public:
  static void compress(const TBuffer &input, TBuffer &output, int compression_level = 3);
  static void decompress(const TBuffer &input, TBuffer &output);
};
} // namespace tnn