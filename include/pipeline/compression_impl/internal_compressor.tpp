#pragma once

#include "internal_compressor.hpp"
#include <memory>
#include <stdexcept>
#include <zstd.h>

namespace tpipeline {

TBuffer ZstdCompressor::compress(const TBuffer &data, int compression_level) {
  if (data.empty()) {
    return data;
  }
  size_t max_compressed_size = ZSTD_compressBound(data.size());
  TBuffer compressed_data(max_compressed_size);

  size_t compressed_size = ZSTD_compress(compressed_data.get(), max_compressed_size, data.get(),
                                         data.size(), compression_level);

  if (ZSTD_isError(compressed_size)) {
    throw std::runtime_error("Zstd compression failed: " +
                             std::string(ZSTD_getErrorName(compressed_size)));
  }

  compressed_data.resize(compressed_size);
  return compressed_data;
}

TBuffer ZstdCompressor::decompress(const TBuffer &data) {
  if (data.empty()) {
    return data;
  }

  unsigned long long decompressed_size = ZSTD_getFrameContentSize(data.get(), data.size());

  if (decompressed_size == ZSTD_CONTENTSIZE_ERROR) {
    throw std::runtime_error("Invalid zstd compressed data");
  }
  if (decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
    throw std::runtime_error("Cannot determine decompressed size");
  }

  TBuffer decompressed_data(decompressed_size);

  size_t result =
      ZSTD_decompress(decompressed_data.get(), decompressed_size, data.get(), data.size());

  if (ZSTD_isError(result)) {
    throw std::runtime_error("Zstd decompression failed: " +
                             std::string(ZSTD_getErrorName(result)));
  }

  return decompressed_data;
}

TBuffer Lz4hcCompressor::compress(const TBuffer &data, int compression_level) {
  // TODO: Implement LZ4HC compression
  return data;
}

TBuffer Lz4hcCompressor::decompress(const TBuffer &data) {
  // TODO: Implement LZ4HC decompression
  return data;
}

} // namespace tpipeline