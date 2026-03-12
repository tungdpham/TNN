#include <gtest/gtest.h>

#include "device/device_manager.hpp"
#include "nn/example_models.hpp"
#include "nn/graph_builder.hpp"
#include "nn/layers.hpp"
#include "nn/layers_impl/dense_layer.hpp"
#include "type/type.hpp"

using namespace std;
using namespace tnn;

class FP16Test : public ::testing::Test {
protected:
  void SetUp() override { ExampleModels::register_defaults(); }
};

TEST_F(FP16Test, Dense) {
  auto &allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  GraphBuilder builder;

  auto fp32_dense_layer = std::make_unique<DenseLayer>(128, 64, false, "fp32_dense");
  fp32_dense_layer->set_io_dtype(DType_t::FP32);
  auto &fp32_node = builder.add_layer(std::move(fp32_dense_layer));

  auto fp16_dense_layer = std::make_unique<DenseLayer>(128, 64, false, "fp16_dense");
  fp16_dense_layer->set_io_dtype(DType_t::FP16);
  fp16_dense_layer->set_param_dtype(DType_t::FP16);
  auto &fp16_node = builder.add_layer(std::move(fp16_dense_layer));

  Graph graph = builder.compile(allocator);

  auto fp16_params = fp16_node.parameters();
  auto fp32_params = fp32_node.parameters();
  for (size_t i = 0; i < fp16_params.size(); ++i) {
    Tensor cpu_fp16_param = fp16_params[i]->to_host();
    Tensor cpu_fp32_param = fp32_params[i]->to_host();
    fp16 *fp16_data = cpu_fp16_param->data_as<fp16>();
    float *fp32_data = cpu_fp32_param->data_as<float>();
    for (size_t j = 0; j < cpu_fp16_param->size(); ++j) {
      fp32_data[j] = static_cast<float>(fp16_data[j]);
    }
    cpu_fp32_param->copy_to(fp32_params[i]);
  }

  Tensor fp16_input = make_tensor(DType_t::FP16, {32, 128}, getHost());
  fp16_input->fill_random_uniform(0.0f, 1.0f);
  Tensor fp32_input = make_tensor(DType_t::FP32, {32, 128}, getHost());

  fp16 *input_data = fp16_input->data_as<fp16>();
  float *input_data_fp32 = fp32_input->data_as<float>();
  for (size_t i = 0; i < fp16_input->size(); ++i) {
    input_data_fp32[i] = static_cast<float>(input_data[i]);
  }

  Tensor input_fp32 = fp32_input->to_device(getGPU());
  Tensor input_fp16 = fp16_input->to_device(getGPU());

  Tensor output_fp32, output_fp16;
  output_fp32 = make_tensor(DType_t::FP32, {32, 64}, getGPU());
  output_fp16 = make_tensor(DType_t::FP16, {32, 64}, getGPU());

  fp32_node.forward({input_fp32}, {output_fp32});
  fp16_node.forward({input_fp16}, {output_fp16});

  Tensor cpu_output_fp32 = output_fp32->to_host();
  Tensor cpu_output_fp16 = output_fp16->to_host();

  float *output_data_fp32 = cpu_output_fp32->data_as<float>();
  fp16 *output_data_fp16 = cpu_output_fp16->data_as<fp16>();
  constexpr double tolerance = 1e-4;
  for (size_t i = 0; i < cpu_output_fp32->size(); ++i) {
    EXPECT_NEAR(static_cast<double>(output_data_fp32[i]), static_cast<double>(output_data_fp16[i]),
                tolerance)
        << "At index " << i;
  }
}
