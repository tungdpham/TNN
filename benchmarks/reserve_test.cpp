#include "tensor/tensor.hpp"
#include <chrono>
#include <iostream>
#include <vector>

signed main() {
  Tensor<float> t(32, 256, 32, 32);
  t.fill_random_normal(0.0f, 0.5f);
  auto manual_start = std::chrono::high_resolution_clock::now();
  uint8_t *data_ptr = (uint8_t *)malloc(t.size() * sizeof(float));
  std::memcpy(data_ptr, t.data(), t.size() * sizeof(float));
  auto manual_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> manual_duration = manual_end - manual_start;
  std::cout << "Manual copying took " << manual_duration.count() << " ms\n";
  free(data_ptr);
  return 0;
}