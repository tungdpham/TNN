#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/example_graphs.hpp"
#include "nn/graph.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"

using namespace tunx;
using namespace std;

signed main() {
  ExampleGraphs::register_defaults();
  auto &allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);

  Graph graph = ExampleGraphs::create("gpt2_small", allocator);

  int passes = 10;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    Vec<Tensor *> grads = graph.gradients();
    for (auto &grad : grads) {
      fill(*grad, 0.0f);
    }
    auto flow = getGPU().getFlow(defaultFlowHandle);
    flow->synchronize();
    auto pass_end = std::chrono::high_resolution_clock::now();
    auto pass_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(pass_end - pass_start);
    std::cout << "Manual Clear: Pass " << i + 1 << " took " << pass_duration.count() << " ms"
              << std::endl;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Manual Clear Average time per forward pass: " << duration.count() / passes << " ms"
            << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    graph.zero_grads();
    Flow *flow = getGPU().getFlow(defaultFlowHandle);
    flow->synchronize();
    auto pass_end = std::chrono::high_resolution_clock::now();
    auto pass_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(pass_end - pass_start);
    std::cout << "Bulk Clear: Pass " << i + 1 << " took " << pass_duration.count() << " ms"
              << std::endl;
  }
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Bulk Clear Average time per forward pass: " << duration.count() / passes << " ms"
            << std::endl;
  return 0;
}
