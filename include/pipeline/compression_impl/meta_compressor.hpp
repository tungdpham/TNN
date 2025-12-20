#pragma once

#include "pipeline/tbuffer.hpp"
#include <cstring>

namespace tnn {
class BloscCompressor {
public:
  static void shuffle_buffer(const TBuffer &input, TBuffer &output, size_t typesize);

  template <typename InCompT>
  static void compress(const TBuffer &input, TBuffer &output, int clevel = 5, int shuffle = 1);

  template <typename InCompT> static void decompress(const TBuffer &input, TBuffer &output);
};
} // namespace tnn