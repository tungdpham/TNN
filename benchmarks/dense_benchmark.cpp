#include "device/device_manager.hpp"
#include "nn/graph.hpp"
#include "nn/graph_builder.hpp"
#include "nn/layers_impl/dense_layer.hpp"
#include "nn/layers_impl/legacy_dense_layer.hpp"
#include "nn/loss.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;
using namespace std;

constexpr size_t INPUT_FEATURES = 262144;
constexpr size_t OUTPUT_FEATURES = 1024;

constexpr float EPSILON = 1e-3f;

signed main() {
  auto &allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  GraphBuilder builder;
  auto dense_layer = make_unique<DenseLayer>(INPUT_FEATURES, OUTPUT_FEATURES, "dense_test");
  auto &dense_op = builder.add_layer(std::move(dense_layer));

  auto legacy_layer =
      make_unique<LegacyDenseLayer>(INPUT_FEATURES, OUTPUT_FEATURES, true, "legacy_dense_test");
  auto &legacy_op = builder.add_layer(std::move(legacy_layer));

  Graph graph = builder.compile(allocator);

  auto current_params = dense_op.parameters();
  auto legacy_params = legacy_op.parameters();
  for (size_t i = 0; i < current_params.size(); ++i) {
    current_params[i]->copy_to(legacy_params[i]);
  }

  Tensor input = make_tensor<float>({128, INPUT_FEATURES}, getGPU());
  input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor current_output = make_tensor<float>({128, OUTPUT_FEATURES}, getGPU());
  Tensor legacy_output = make_tensor<float>({128, OUTPUT_FEATURES}, getGPU());
  // cold pass
  dense_op.forward({input}, {current_output});

  int passes = 10;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    dense_op.forward({input}, {current_output});
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
  // cold pass
  legacy_op.forward({input}, {legacy_output});
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    legacy_op.forward({input}, {legacy_output});
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

  auto cpu_current_output = current_output->to_device(getHost());
  auto cpu_legacy_output = legacy_output->to_device(getHost());

  float *current_data = cpu_current_output->data_as<float>();
  float *legacy_data = cpu_legacy_output->data_as<float>();
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
  auto criterion = LossFactory::create_logsoftmax_crossentropy();
  Tensor target = make_tensor<float>({128, OUTPUT_FEATURES}, getGPU());
  target->fill_random_normal(0.5f, 0.2f);
  Tensor grad = make_tensor<float>({128, OUTPUT_FEATURES}, getGPU());
  criterion->compute_gradient(current_output, target, grad);

  Tensor grad_input_current = make_tensor<float>({128, INPUT_FEATURES}, getGPU());
  Tensor grad_input_legacy = make_tensor<float>({128, INPUT_FEATURES}, getGPU());

  // cold pass
  dense_op.backward({grad}, {grad_input_current});
  legacy_op.backward({grad}, {grad_input_legacy});

  for (int i = 0; i < passes; ++i) {
    // forward pass to have cached data
    dense_op.forward({input}, {current_output});
    auto pass_start = std::chrono::high_resolution_clock::now();
    dense_op.backward({grad}, {grad_input_current});
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
    legacy_op.forward({input}, {legacy_output});
    auto pass_start = std::chrono::high_resolution_clock::now();
    legacy_op.backward({grad}, {grad_input_legacy});
    Flow *flow = getGPU().getFlow(defaultFlowHandle);
    flow->synchronize();
    auto pass_end = std::chrono::high_resolution_clock::now();
    auto pass_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(pass_end - pass_start);
    std::cout << "Legacy Dense Backward: Pass " << i + 1 << " took " << pass_duration.count()
              << " ms" << std::endl;
  }

  auto cpu_grad_input_current = grad_input_current->to_host();
  auto cpu_grad_input_legacy = grad_input_legacy->to_host();
  float *grad_input_current_data = (float *)cpu_grad_input_current->data();
  float *grad_input_legacy_data = (float *)cpu_grad_input_legacy->data();
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

  auto grad_weights_current = dense_op.gradients();
  auto grad_weights_legacy = legacy_op.gradients();
  for (size_t i = 0; i < grad_weights_current.size(); ++i) {
    auto cpu_grad_current = grad_weights_current[i]->to_host();
    auto cpu_grad_legacy = grad_weights_legacy[i]->to_host();
    float *grad_current_data = (float *)cpu_grad_current->data();
    float *grad_legacy_data = (float *)cpu_grad_legacy->data();
    size_t grad_elements = cpu_grad_current->size();
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
