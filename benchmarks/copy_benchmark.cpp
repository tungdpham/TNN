
#include "threading/thread_handler.hpp"
#include "threading/thread_wrapper.hpp"
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace tnn;

signed main() {
  // Raw copy speed
  uint8_t *src_data = (uint8_t *)std::aligned_alloc(64, 128 * 512 * 16 * 16 * sizeof(float));
  size_t data_size = 128 * 512 * 16 * 16 * sizeof(float);
  uint8_t *dst_data = (uint8_t *)std::aligned_alloc(64, data_size);
  ThreadWrapper thread_wrapper({8});
  thread_wrapper.execute([&]() -> void {
    for (int i = 0; i < 10; i++) {
      auto copy_start = std::chrono::high_resolution_clock::now();
      size_t num_threads = get_num_threads();
      std::cout << "Using " << num_threads << " threads for copy benchmark." << std::endl;
      size_t block_size = data_size / num_threads;
      parallel_for<size_t>(0, num_threads, [&](size_t thread_id) {
        size_t start = thread_id * block_size;
        size_t end = (thread_id == num_threads - 1) ? data_size : start + block_size;
        std::memcpy(dst_data + start, src_data + start, end - start);
      });
      auto copy_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> copy_duration = copy_end - copy_start;
      std::cout << "Raw copy took " << copy_duration.count() << " ms" << std::endl;
    }
  });
  std::free(dst_data);
  std::free(src_data);
  return 0;
}