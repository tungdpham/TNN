#include "tensor/tensor.hpp"
#include <chrono>
#include <iostream>
#include <vector>

signed main() {
  Tensor<float> t(32, 256, 32, 32);
  t.fill_random_normal(0.0f, 0.5f);
  std::vector<float> v;
  auto resize_start = std::chrono::high_resolution_clock::now();
  v.resize(1e7 + 5);
  auto data_ptr = v.data();
  std::memcpy(data_ptr, t.data(), t.size() * sizeof(float));
  auto resize_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> resize_duration = resize_end - resize_start;
  std::cout << "Resizing took " << resize_duration.count() << " ms\n";
  return 0;
}