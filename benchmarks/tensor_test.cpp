#include "tensor/tensor.hpp"

int main() {
  Tensor<float> tensor(64, 1, 32, 32);
  tensor.fill_random_normal(0.0f, 1.0f);
  std::cout << "Original tensor shape: " << tensor.shape_str() << std::endl;
  // tensor.print_data();

  auto sliced_tensor = tensor.slice_batch(0, 16);
  std::cout << "Sliced tensor shape: " << sliced_tensor.shape_str() << std::endl;
  // sliced_tensor.print_data();

  // verify
  for (size_t i = 0; i < 16; ++i) {
    for (size_t c = 0; c < 1; ++c) {
      for (size_t h = 0; h < 32; ++h) {
        for (size_t w = 0; w < 32; ++w) {
          if (sliced_tensor(i, c, h, w) != tensor(i, c, h, w)) {
            std::cerr << "Mismatch at (" << i << ", " << c << ", " << h << ", " << w
                      << "): " << sliced_tensor(i, c, h, w) << " != " << tensor(i, c, h, w)
                      << std::endl;
            return -1;
          }
        }
      }
    }
  }
  std::cout << "Slice verification passed!" << std::endl;

  auto splits = tensor.split(4);
  for (size_t i = 0; i < splits.size(); ++i) {
    std::cout << "Split " << i << " shape: " << splits[i].shape_str() << std::endl;
    // splits[i].print_data();
  }
  // verify
  for (size_t i = 0; i < splits.size(); ++i) {
    for (size_t n = 0; n < splits[i].batch_size(); ++n) {
      for (size_t c = 0; c < splits[i].channels(); ++c) {
        for (size_t h = 0; h < splits[i].height(); ++h) {
          for (size_t w = 0; w < splits[i].width(); ++w) {
            size_t original_n = i * splits[i].batch_size() + n;
            if (splits[i](n, c, h, w) != tensor(original_n, c, h, w)) {
              std::cerr << "Mismatch in split " << i << " at (" << n << ", " << c << ", " << h
                        << ", " << w << "): " << splits[i](n, c, h, w)
                        << " != " << tensor(original_n, c, h, w) << std::endl;
              return -1;
            }
          }
        }
      }
    }
  }
  std::cout << "Split verification passed!" << std::endl;
  return 0;
}