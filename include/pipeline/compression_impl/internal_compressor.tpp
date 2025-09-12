#pragma once

#include "internal_compressor.hpp"
#include <memory>
#include <stdexcept>
#include <zstd.h>

namespace tpipeline {

std::vector<uint8_t> ZstdCompressor::compress(const std::vector<uint8_t> &data,
                                              int compression_level) {
  if (data.empty()) {
    return data;
  }
  size_t max_compressed_size = ZSTD_compressBound(data.size());
  std::vector<uint8_t> compressed_data(max_compressed_size);

  size_t compressed_size =
      ZSTD_compress(compressed_data.data(), max_compressed_size, data.data(),
                    data.size(), compression_level);

  if (ZSTD_isError(compressed_size)) {
    throw std::runtime_error("Zstd compression failed: " +
                             std::string(ZSTD_getErrorName(compressed_size)));
  }

  compressed_data.resize(compressed_size);
  return compressed_data;
}

std::vector<uint8_t>
ZstdCompressor::decompress(const std::vector<uint8_t> &data) {
  if (data.empty()) {
    return data;
  }

  unsigned long long decompressed_size =
      ZSTD_getFrameContentSize(data.data(), data.size());

  if (decompressed_size == ZSTD_CONTENTSIZE_ERROR) {
    throw std::runtime_error("Invalid zstd compressed data");
  }
  if (decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
    throw std::runtime_error("Cannot determine decompressed size");
  }

  std::vector<uint8_t> decompressed_data(decompressed_size);

  size_t result = ZSTD_decompress(decompressed_data.data(), decompressed_size,
                                  data.data(), data.size());

  if (ZSTD_isError(result)) {
    throw std::runtime_error("Zstd decompression failed: " +
                             std::string(ZSTD_getErrorName(result)));
  }

  return decompressed_data;
}

std::vector<uint8_t> Lz4hcCompressor::compress(const std::vector<uint8_t> &data,
                                               int compression_level) {
  // TODO: Implement LZ4HC compression
  return data;
}

std::vector<uint8_t>
Lz4hcCompressor::decompress(const std::vector<uint8_t> &data) {
  // TODO: Implement LZ4HC decompression
  return data;
}

} // namespace tpipeline