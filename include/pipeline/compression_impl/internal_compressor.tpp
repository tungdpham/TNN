#pragma once

#include "internal_compressor.hpp"
#include "pipeline/tbuffer.hpp"
#include <stdexcept>
#include <zstd.h>

namespace tnn {

void ZstdCompressor::compress(const TBuffer &input, TBuffer &output, int compression_level) {
  if (input.empty()) {
    return;
  }
  size_t max_compressed_size = ZSTD_compressBound(input.size());
  output.resize(max_compressed_size);

  size_t compressed_size = ZSTD_compress(output.get(), max_compressed_size, input.get(),
                                         input.size(), compression_level);

  if (ZSTD_isError(compressed_size)) {
    throw std::runtime_error("Zstd compression failed: " +
                             std::string(ZSTD_getErrorName(compressed_size)));
  }
}

void ZstdCompressor::decompress(const TBuffer &input, TBuffer &output) {
  if (input.empty()) {
    return;
  }

  unsigned long long decompressed_size = ZSTD_getFrameContentSize(input.get(), input.size());
  if (decompressed_size == ZSTD_CONTENTSIZE_ERROR) {
    throw std::runtime_error("Invalid zstd compressed data");
  }
  if (decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
    throw std::runtime_error("Cannot determine decompressed size");
  }

  TBuffer decompressed_data(decompressed_size);

  size_t result = ZSTD_decompress(output.get(), decompressed_size, input.get(), input.size());

  if (ZSTD_isError(result)) {
    throw std::runtime_error("Zstd decompression failed: " +
                             std::string(ZSTD_getErrorName(result)));
  }

  return;
}

void Lz4hcCompressor::compress(const TBuffer &input, TBuffer &output, int compression_level) {
  // TODO: Implement LZ4HC compression
  throw new std::runtime_error("LZ4HC compression not implemented");
}

void Lz4hcCompressor::decompress(const TBuffer &input, TBuffer &output) {
  // TODO: Implement LZ4HC decompression
  throw new std::runtime_error("LZ4HC decompression not implemented");
}

} // namespace tnn