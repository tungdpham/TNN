#pragma once

#include "meta_compressor.hpp"

namespace tpipeline {
std::vector<uint8_t> BloscCompressor::compress(const std::vector<uint8_t> &data,
                                               int clevel, int shuffle) {

  return data;
}

std::vector<uint8_t>
BloscCompressor::decompress(const std::vector<uint8_t> &data) {

  return data;
}
} // namespace tpipeline