#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/graph.hpp"
#include "nn/layer_factory.hpp"
#include "nn/layers_impl/legacy_dense_layer.hpp"
#include "nn/loss.hpp"
#include "tensor/tensor.hpp"

using namespace tunx;
using namespace std;

constexpr size_t INPUT_FEATURES = 262144;
constexpr size_t OUTPUT_FEATURES = 1024;

constexpr float EPSILON = 1e-3f;

signed main() {
  auto &device = getGPU();
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);

  Graph graph;
  auto input = graph.make_node("input");

  auto dense_layer = DenseLayer(INPUT_FEATURES, OUTPUT_FEATURES, "dense_test");
  auto dense_output = dense_layer(input);

  auto legacy_layer = LegacyDenseLayer(INPUT_FEATURES, OUTPUT_FEATURES, true, "legacy_dense_test");
  auto legacy_dense_output = legacy_layer(input);

  graph.compile(allocator);

  Vec<ParamDescriptor> current_params = dense_layer.param_descriptors();
  Vec<ParamDescriptor> legacy_params = legacy_layer.param_descriptors();
  for (size_t i = 0; i < current_params.size(); ++i) {
    current_params[i].data_ptr->copy_to(*legacy_params[i].data_ptr);
  }

  Tensor input_data = Tensor({128, INPUT_FEATURES}, DType_t::FP32, getGPU());
  fill_normal(input_data, 0.5f, 0.2f, 676767);

  Residuals residuals;
  // cold pass
  Tensor current_output = dense_layer.forward({input_data}, residuals)[0];

  int passes = 10;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    current_output = dense_layer.forward({input_data}, residuals)[0];
    Flow *flow = getGPU().getFlow(defaultFlowHandle);
    flow->synchronize();

    auto pass_end = std::chrono::high_resolution_clock::now();
    auto pass_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(pass_end - pass_start);
    std::cout << "Dense: Pass " << i + 1 << " took " << pass_duration.count() << " ms" << std::endl;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Dense Average time per forward pass: " << duration.count() / passes << " ms"
            << std::endl;

  // legacy dense benchmark
  Residuals legacy_residuals;
  // cold pass
  Tensor legacy_output = legacy_layer.forward({input_data}, legacy_residuals)[0];
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    legacy_output = legacy_layer.forward({input_data}, legacy_residuals)[0];
    Flow *flow = getGPU().getFlow(defaultFlowHandle);
    flow->synchronize();
    auto pass_end = std::chrono::high_resolution_clock::now();
    auto pass_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(pass_end - pass_start);
    std::cout << "Legacy Dense: Pass " << i + 1 << " took " << pass_duration.count() << " ms"
              << std::endl;
  }
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Legacy Dense Average time per forward pass: " << duration.count() / passes << " ms"
            << std::endl;

  auto cpu_current_output = current_output.to_device(getHost());
  auto cpu_legacy_output = legacy_output.to_device(getHost());

  float *current_data = cpu_current_output.data_as<float>();
  float *legacy_data = cpu_legacy_output.data_as<float>();
  size_t total_elements = 128 * OUTPUT_FEATURES;
  float max_diff = 0.0f;
  for (size_t i = 0; i < total_elements; ++i) {
    float diff = std::abs(current_data[i] - legacy_data[i]);
    if (diff > EPSILON) {
      std::cout << "Mismatch at index " << i << ": current = " << current_data[i]
                << ", legacy = " << legacy_data[i] << ", diff = " << diff << std::endl;
    }
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  std::cout << "Max diff: " << max_diff << std::endl;

  // test backward
  auto criterion = LossFactory::create_crossentropy();
  Tensor target = Tensor({128, OUTPUT_FEATURES}, DType_t::FP32, getGPU());
  fill_normal(target, 0.5f, 0.2f);
  Tensor grad = Tensor({128, OUTPUT_FEATURES}, DType_t::FP32, getGPU());
  criterion->compute_gradient(current_output, target, grad);

  // cold pass
  Tensor grad_input_current = dense_layer.backward({grad}, residuals)[0];
  Tensor grad_input_legacy = legacy_layer.backward({grad}, legacy_residuals)[0];

  for (int i = 0; i < passes; ++i) {
    // forward pass to have cached data
    dense_layer.forward({input_data}, residuals);
    auto pass_start = std::chrono::high_resolution_clock::now();
    grad_input_current = dense_layer.backward({grad}, residuals)[0];
    Flow *flow = getGPU().getFlow(defaultFlowHandle);
    flow->synchronize();
    auto pass_end = std::chrono::high_resolution_clock::now();
    auto pass_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(pass_end - pass_start);
    std::cout << "Dense Backward: Pass " << i + 1 << " took " << pass_duration.count() << " ms"
              << std::endl;
  }

  for (int i = 0; i < passes; ++i) {
    // forward pass to have cached data
    legacy_layer.forward({input_data}, legacy_residuals);
    auto pass_start = std::chrono::high_resolution_clock::now();
    grad_input_legacy = legacy_layer.backward({grad}, legacy_residuals)[0];
    Flow *flow = getGPU().getFlow(defaultFlowHandle);
    flow->synchronize();
    auto pass_end = std::chrono::high_resolution_clock::now();
    auto pass_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(pass_end - pass_start);
    std::cout << "Legacy Dense Backward: Pass " << i + 1 << " took " << pass_duration.count()
              << " ms" << std::endl;
  }

  auto cpu_grad_input_current = grad_input_current.to_host();
  auto cpu_grad_input_legacy = grad_input_legacy.to_host();
  float *grad_input_current_data = (float *)cpu_grad_input_current.data_as<void>();
  float *grad_input_legacy_data = (float *)cpu_grad_input_legacy.data_as<void>();
  max_diff = 0.0f;
  for (size_t i = 0; i < total_elements; ++i) {
    float diff = std::abs(grad_input_current_data[i] - grad_input_legacy_data[i]);
    if (diff > EPSILON) {
      std::cout << "Mismatch at index " << i << ": current = " << grad_input_current_data[i]
                << ", legacy = " << grad_input_legacy_data[i] << ", diff = " << diff << std::endl;
    }
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  std::cout << "Max grad diff: " << max_diff << std::endl;

  // check wgrad

  Vec<ParamDescriptor> grad_weights_current = dense_layer.param_descriptors();
  Vec<ParamDescriptor> grad_weights_legacy = legacy_layer.param_descriptors();
  for (size_t i = 0; i < grad_weights_current.size(); ++i) {
    auto cpu_grad_current = grad_weights_current[i].grad_ptr->to_host();
    auto cpu_grad_legacy = grad_weights_legacy[i].grad_ptr->to_host();
    float *grad_current_data = (float *)cpu_grad_current.data_as<void>();
    float *grad_legacy_data = (float *)cpu_grad_legacy.data_as<void>();
    size_t grad_elements = cpu_grad_current.size();
    max_diff = 0.0f;
    for (size_t j = 0; j < grad_elements; ++j) {
      float diff = std::abs(grad_current_data[j] - grad_legacy_data[j]);
      if (diff > EPSILON) {
        std::cout << "Weight Grad Mismatch at index " << j << ": current = " << grad_current_data[j]
                  << ", legacy = " << grad_legacy_data[j] << ", diff = " << diff << std::endl;
      }
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
    std::cout << "Max weight grad diff for parameter " << i << ": " << max_diff << std::endl;
  }

  return 0;
}
