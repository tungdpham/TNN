#include "device/device_manager.hpp"
#include "nn/blocks_impl/sequential.hpp"
#include "nn/example_models.hpp"
#include "nn/graph_api.hpp"
using namespace tnn;
using namespace std;

signed main() {
  ExampleModels::register_defaults();
  auto &allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);

  Sequential model = ExampleModels::create("gpt2_small");
  model->set_seed(123456);
  Graph graph;
  auto input = graph.make_node("input");
  auto output = model(input);
  output->set_uid("output");
  graph.compile(allocator);

  int passes = 10;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    auto grads = model->gradients();
    for (auto &grad : grads) {
      grad->fill(0.0);
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
    graph.context()->zero_grads();
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
