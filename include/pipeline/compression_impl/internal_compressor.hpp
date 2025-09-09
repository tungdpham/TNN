#pragma once

#include <vector>
#include <cstdint>

namespace tpipeline {
  class ZstdCompressor {
  public:
    static std::vector<uint8_t> compress(const std::vector<uint8_t>& data, int compression_level = 3);
    static std::vector<uint8_t> decompress(const std::vector<uint8_t>& data); 
  };

  class Lz4hcCompressor {
  public:
    static std::vector<uint8_t> compress(const std::vector<uint8_t>& data, int compression_level = 3);
    static std::vector<uint8_t> decompress(const std::vector<uint8_t>& data); 
  };
} // namespace tpipeline