#include <gtest/gtest.h>

#include <cstddef>

#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/blocks_impl/attention_block.hpp"
#include "nn/example_graphs.hpp"
#include "nn/graph.hpp"
#include "nn/layer_factory.hpp"
#include "nn/layers_impl/dense_layer.hpp"
#include "nn/loss.hpp"
#include "type/type.hpp"

using namespace std;
using namespace tunx;

class BF16Test : public ::testing::Test {
protected:
  void SetUp() override { ExampleGraphs::register_defaults(); }
};

TEST_F(BF16Test, Dense) {
  constexpr size_t batch_size = 8;
  constexpr size_t input_dim = 32;
  constexpr size_t output_dim = 16;
  auto &allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);

  auto fp32_dense_layer = DenseLayer(input_dim, output_dim, false, "fp32_dense");
  fp32_dense_layer.set_io_dtype(DType_t::FP32);

  auto bf16_dense_layer = DenseLayer(input_dim, output_dim, false, "bf16_dense");
  bf16_dense_layer.set_io_dtype(DType_t::BF16);
  bf16_dense_layer.set_param_dtype(DType_t::BF16);

  Graph graph;
  Node input = graph.make_node("input");
  Node fp32_output = fp32_dense_layer(input);
  fp32_output->set_uid("fp32_output");
  Node bf16_output = bf16_dense_layer(input);
  bf16_output->set_uid("bf16_output");
  graph.compile(allocator);

  auto bf16_params = bf16_dense_layer.parameters();
  auto fp32_params = fp32_dense_layer.parameters();
  for (size_t i = 0; i < bf16_params.size(); ++i) {
    bf16_params[i]->copy_to(*fp32_params[i]);
  }

  Tensor bf16_input = Tensor({batch_size, input_dim}, DType_t::BF16, getHost());
  fill_normal(bf16_input, 0.0f, 1.0f);
  Tensor fp32_input = Tensor({batch_size, input_dim}, DType_t::FP32, getHost());

  bf16 *input_data = bf16_input.data_as<bf16>();
  float *input_data_fp32 = fp32_input.data_as<float>();
  for (size_t i = 0; i < bf16_input.size(); ++i) {
    input_data_fp32[i] = static_cast<float>(input_data[i]);
  }

  Tensor input_fp32 = fp32_input.to_device(getGPU());
  Tensor input_bf16 = bf16_input.to_device(getGPU());

  Residuals fp32_residuals, bf16_residuals;
  Tensor output_fp32 = fp32_dense_layer.forward({input_fp32}, fp32_residuals)[0];
  Tensor output_bf16 = bf16_dense_layer.forward({input_bf16}, bf16_residuals)[0];

  Tensor cpu_output_fp32 = output_fp32.to_host();
  Tensor cpu_output_bf16 = output_bf16.to_host();

  float *output_data_fp32 = cpu_output_fp32.data_as<float>();
  bf16 *output_data_bf16 = cpu_output_bf16.data_as<bf16>();
  constexpr double tolerance = 2e-3;
  for (size_t i = 0; i < cpu_output_fp32.size(); ++i) {
    EXPECT_NEAR(static_cast<double>(output_data_fp32[i]), static_cast<double>(output_data_bf16[i]),
                tolerance)
        << "At index " << i;
  }

  Tensor target_fp32 = Tensor({batch_size, output_dim}, DType_t::FP32);
  Tensor target_bf16 = Tensor({batch_size, output_dim}, DType_t::BF16);
  fill(target_fp32, 0.0f);
  fill(target_bf16, bf16(0.0f));

  for (size_t i = 0; i < batch_size; ++i) {
    target_fp32.at<float>({i, i % output_dim}) = 1.0f;
    target_bf16.at<bf16>({i, i % output_dim}) = bf16(1.0f);
  }

  auto criterion = LossFactory::create_crossentropy();

  auto gradient_fp32 = Tensor({batch_size, output_dim}, DType_t::FP32);
  auto gradient_bf16 = Tensor({batch_size, output_dim}, DType_t::BF16);

  criterion->compute_gradient(cpu_output_fp32, target_fp32, gradient_fp32);
  criterion->compute_gradient(cpu_output_bf16, target_bf16, gradient_bf16);

  auto gpu_gradient_fp32 = gradient_fp32.to_device(getGPU());
  auto gpu_gradient_bf16 = gradient_bf16.to_device(getGPU());

  Tensor grad_input_bf16 = bf16_dense_layer.backward({gpu_gradient_bf16}, bf16_residuals)[0];
  Tensor grad_input_fp32 = fp32_dense_layer.backward({gpu_gradient_fp32}, fp32_residuals)[0];

  Tensor cpu_grad_input_fp32 = grad_input_fp32.to_host();
  Tensor cpu_grad_input_bf16 = grad_input_bf16.to_host();
  float *grad_input_data_fp32 = cpu_grad_input_fp32.data_as<float>();
  bf16 *grad_input_data_bf16 = cpu_grad_input_bf16.data_as<bf16>();
  for (size_t i = 0; i < cpu_grad_input_fp32.size(); ++i) {
    EXPECT_NEAR(static_cast<double>(grad_input_data_fp32[i]),
                static_cast<double>(grad_input_data_bf16[i]), tolerance)
        << "At index " << i;
  }
}

TEST_F(BF16Test, Attention) {
  constexpr size_t batch_size = 8;
  constexpr size_t seq_len = 16;
  constexpr size_t embed_dim = 16;
  constexpr size_t num_heads = 4;
  auto &allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);

  auto fp32_attention_layer = AttentionBlock(embed_dim, num_heads, false, "fp32_attention");
  fp32_attention_layer.set_io_dtype(DType_t::FP32);

  auto bf16_attention_layer = AttentionBlock(embed_dim, num_heads, false, "bf16_attention");
  bf16_attention_layer.set_io_dtype(DType_t::BF16);
  bf16_attention_layer.set_param_dtype(DType_t::BF16);

  Graph graph;
  Node input = graph.make_node("input");
  Node fp32_output = fp32_attention_layer(input);
  fp32_output->set_uid("fp32_output");
  Node bf16_output = bf16_attention_layer(input);
  bf16_output->set_uid("bf16_output");
  graph.compile(allocator);

  auto bf16_params = bf16_attention_layer.parameters();
  auto fp32_params = fp32_attention_layer.parameters();
  for (size_t i = 0; i < bf16_params.size(); ++i) {
    Tensor cpu_bf16_param = bf16_params[i]->to_host();
    Tensor cpu_fp32_param = fp32_params[i]->to_host();
    bf16 *bf16_data = cpu_bf16_param.data_as<bf16>();
    float *fp32_data = cpu_fp32_param.data_as<float>();
    for (size_t j = 0; j < cpu_bf16_param.size(); ++j) {
      fp32_data[j] = static_cast<float>(bf16_data[j]);
    }
    cpu_fp32_param.copy_to(*fp32_params[i]);
  }

  Tensor bf16_input = Tensor({batch_size, seq_len, embed_dim}, DType_t::BF16, getHost());
  fill_normal(bf16_input, 0.0f, 1.0f);
  Tensor fp32_input = Tensor({batch_size, seq_len, embed_dim}, DType_t::FP32, getHost());

  bf16 *input_data = bf16_input.data_as<bf16>();
  float *input_data_fp32 = fp32_input.data_as<float>();
  for (size_t i = 0; i < bf16_input.size(); ++i) {
    input_data_fp32[i] = static_cast<float>(input_data[i]);
  }

  Tensor input_fp32 = fp32_input.to_device(getGPU());
  Tensor input_bf16 = bf16_input.to_device(getGPU());

  Residuals fp32_residuals, bf16_residuals;

  Tensor output_fp32 = fp32_attention_layer.forward({input_fp32}, fp32_residuals)[0];
  Tensor output_bf16 = bf16_attention_layer.forward({input_bf16}, bf16_residuals)[0];

  Tensor cpu_output_fp32 = output_fp32.to_host();
  Tensor cpu_output_bf16 = output_bf16.to_host();

  float *output_data_fp32 = cpu_output_fp32.data_as<float>();
  bf16 *output_data_bf16 = cpu_output_bf16.data_as<bf16>();
  constexpr double tolerance = 2e-3;
  for (size_t i = 0; i < cpu_output_fp32.size(); ++i) {
    EXPECT_NEAR(static_cast<double>(output_data_fp32[i]), static_cast<double>(output_data_bf16[i]),
                tolerance)
        << "At index " << i;
  }

  Tensor target_fp32 = Tensor({batch_size, seq_len, embed_dim}, DType_t::FP32);
  Tensor target_bf16 = Tensor({batch_size, seq_len, embed_dim}, DType_t::BF16);
  fill(target_fp32, 0.0f);
  fill(target_bf16, bf16(0.0f));

  for (size_t i = 0; i < 32; ++i) {
    target_fp32.at<float>({i, i % 16, i / 16}) = 1.0f;
    target_bf16.at<bf16>({i, i % 16, i / 16}) = bf16(1.0f);
  }

  auto criterion = LossFactory::create_crossentropy();

  auto gradient_fp32 = Tensor({batch_size, seq_len, embed_dim}, DType_t::FP32);
  auto gradient_bf16 = Tensor({batch_size, seq_len, embed_dim}, DType_t::BF16);

  criterion->compute_gradient(cpu_output_fp32, target_fp32, gradient_fp32);
  criterion->compute_gradient(cpu_output_bf16, target_bf16, gradient_bf16);

  auto gpu_gradient_fp32 = gradient_fp32.to_device(getGPU());
  auto gpu_gradient_bf16 = gradient_bf16.to_device(getGPU());

  Tensor grad_input_bf16 = bf16_attention_layer.backward({gpu_gradient_bf16}, bf16_residuals)[0];
  Tensor grad_input_fp32 = fp32_attention_layer.backward({gpu_gradient_fp32}, fp32_residuals)[0];

  Tensor cpu_grad_input_fp32 = grad_input_fp32.to_host();
  Tensor cpu_grad_input_bf16 = grad_input_bf16.to_host();
  float *grad_input_data_fp32 = cpu_grad_input_fp32.data_as<float>();
  bf16 *grad_input_data_bf16 = cpu_grad_input_bf16.data_as<bf16>();
  for (size_t i = 0; i < cpu_grad_input_fp32.size(); ++i) {
    EXPECT_NEAR(static_cast<double>(grad_input_data_fp32[i]),
                static_cast<double>(grad_input_data_bf16[i]), tolerance)
        << "At index " << i;
  }
}
