#pragma once
#include "pipeline/tbuffer.hpp"

namespace tnn {
class ZstdCompressor {
public:
  static TBuffer compress(const TBuffer &data, int compression_level = 3);
  static TBuffer decompress(const TBuffer &data);
};

class Lz4hcCompressor {
public:
  static TBuffer compress(const TBuffer &data, int compression_level = 3);
  static TBuffer decompress(const TBuffer &data);
};
} // namespace tnn