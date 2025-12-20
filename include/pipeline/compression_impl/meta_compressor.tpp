#pragma once

#include "meta_compressor.hpp"

namespace tnn {

void BloscCompressor::shuffle_buffer(const TBuffer &input, TBuffer &output, size_t typesize) {
  size_t n = input.size();
  output.resize(n);
  size_t nelements = n / typesize;
  for (size_t i = 0; i < typesize; ++i) {
    for (size_t j = 0; j < nelements; ++j) {
      output.get()[i * nelements + j] = input.get()[j * typesize + i];
    }
  }
}

template <typename InCompT>
void BloscCompressor::compress(const TBuffer &input, TBuffer &output, int clevel, int shuffle) {
  if (shuffle == 0 || input.size() % 4 != 0) {
    InCompT::compress(input, output, clevel);
    return;
  }
  TBuffer shuffled;
  shuffle_buffer(input, shuffled, 4);
  InCompT::compress(shuffled, output, clevel);
  return;
}

template <typename InCompT>
void BloscCompressor::decompress(const TBuffer &input, TBuffer &output) {
  InCompT::decompress(input, output);
  return;
}

} // namespace tnn