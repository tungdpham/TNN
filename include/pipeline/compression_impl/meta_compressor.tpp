#pragma once

#include "meta_compressor.hpp"

namespace tnn {
TBuffer BloscCompressor::compress(const TBuffer &data, int clevel, int shuffle) { return data; }

TBuffer BloscCompressor::decompress(const TBuffer &data) { return data; }
} // namespace tnn