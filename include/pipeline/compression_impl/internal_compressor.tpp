#pragma once

#include "internal_compressor.hpp"

namespace tpipeline {
std::vector<uint8_t> ZstdCompressor::compress(const std::vector<uint8_t> &data,
                                              int compression_level) {

  return data;
}

std::vector<uint8_t>
ZstdCompressor::decompress(const std::vector<uint8_t> &data) {

  return data;
}

std::vector<uint8_t> Lz4hcCompressor::compress(const std::vector<uint8_t> &data,
                                               int compression_level) {

  return data;
}

std::vector<uint8_t>
Lz4hcCompressor::decompress(const std::vector<uint8_t> &data) {

  return data;
}
} // namespace tpipeline