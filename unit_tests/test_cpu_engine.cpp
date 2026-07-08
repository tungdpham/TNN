/*
 * Copyright (c) 2026 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <gtest/gtest.h>

#include "device/device_manager.hpp"
#include "engine_test_utils.hpp"
#include "nn/engines/cpu_engine.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"

using namespace tunx;

class CPUEngineTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    initializeDefaultDevices();
    engine_ = std::make_unique<CPUEngine>();
  }

  static void TearDownTestSuite() { engine_.reset(); }

  static std::unique_ptr<CPUEngine> engine_;
};

std::unique_ptr<CPUEngine> CPUEngineTest::engine_;

TEST_F(CPUEngineTest, DenseFwdReturnsCorrectResults) {
  size_t batch_size = 16;
  size_t in_features = 64;
  size_t out_features = 32;

  DenseStats stats{
      .batch_size = batch_size,
      .in_features = in_features,
      .out_features = out_features,
      .use_bias = true,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor input({batch_size, in_features}, DType_t::FP32, getHost());
  fill_normal(input, 0.0, 0.5);
  Tensor weight({out_features, in_features}, DType_t::FP32, getHost());
  fill_normal(weight, 0.0, 0.1);
  Tensor bias({out_features}, DType_t::FP32, getHost());
  fill_normal(bias, 0.0, 0.1);
  Tensor output({batch_size, out_features}, DType_t::FP32, getHost());

  void* handle = nullptr;
  WorkspaceReq req = engine_->query_dense_graph(handle, stats, type_desc);

  size_t ws_size = req.fwd_workspace > 0 ? req.fwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getHost());

  engine_->dense_fwd(handle, stats, input.data_as<void>(), weight.data_as<void>(),
                     bias.data_as<void>(), output.data_as<void>(), workspace.data_as<void>(),
                     type_desc);

  Tensor expected_output({batch_size, out_features}, DType_t::FP32, getHost());
  math_dense_fwd(input.data_as<float>(), weight.data_as<float>(), bias.data_as<float>(),
                 expected_output.data_as<float>(), batch_size, in_features, out_features);

  compare_tensor(output, expected_output);
}

TEST_F(CPUEngineTest, DenseWgradReturnsCorrectResults) {
  size_t batch_size = 16;
  size_t in_features = 64;
  size_t out_features = 32;

  DenseStats stats{
      .batch_size = batch_size,
      .in_features = in_features,
      .out_features = out_features,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor input({batch_size, in_features}, DType_t::FP32, getHost());
  fill_normal(input, 0.0, 0.5);
  Tensor grad_output({batch_size, out_features}, DType_t::FP32, getHost());
  fill_normal(grad_output, 0.0, 0.2);
  Tensor grad_weight_temp({out_features, in_features}, DType_t::FP32, getHost());
  fill(grad_weight_temp, 0.0f);
  Tensor grad_weight({out_features, in_features}, DType_t::FP32, getHost());
  fill(grad_weight, 0.0f);

  void* handle = nullptr;
  WorkspaceReq req = engine_->query_dense_graph(handle, stats, type_desc);

  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getHost());

  engine_->dense_wgrad(handle, stats, grad_output.data_as<void>(), input.data_as<void>(),
                       grad_weight.data_as<void>(), workspace.data_as<void>(), type_desc);

  Tensor expected_grad_weight({out_features, in_features}, DType_t::FP32, getHost());

  math_dense_wgrad(input.data_as<float>(), grad_output.data_as<float>(),
                   expected_grad_weight.data_as<float>(), batch_size, in_features, out_features);

  compare_tensor(grad_weight, expected_grad_weight);
}

TEST_F(CPUEngineTest, DenseDgradReturnsCorrectResults) {
  size_t batch_size = 16;
  size_t in_features = 64;
  size_t out_features = 32;

  DenseStats stats{
      .batch_size = batch_size,
      .in_features = in_features,
      .out_features = out_features,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor grad_output({batch_size, out_features}, DType_t::FP32, getHost());
  fill_normal(grad_output, 0.0, 0.5);
  Tensor weight({out_features, in_features}, DType_t::FP32, getHost());
  fill_normal(weight, 0.0, 0.1);
  Tensor grad_input({batch_size, in_features}, DType_t::FP32, getHost());

  void* handle = nullptr;
  WorkspaceReq req = engine_->query_dense_graph(handle, stats, type_desc);

  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getHost());

  engine_->dense_dgrad(handle, stats, grad_output.data_as<void>(), weight.data_as<void>(),
                       grad_input.data_as<void>(), workspace.data_as<void>(), type_desc);

  Tensor expected_grad_input({batch_size, in_features}, DType_t::FP32, getHost());
  math_dense_dgrad(grad_output.data_as<float>(), weight.data_as<float>(),
                   expected_grad_input.data_as<float>(), batch_size, in_features, out_features);

  compare_tensor(grad_input, expected_grad_input);
}

TEST_F(CPUEngineTest, DenseBgradReturnsCorrectResults) {
  size_t batch_size = 16;
  size_t in_features = 64;
  size_t out_features = 32;

  DenseStats stats{
      .batch_size = batch_size,
      .in_features = in_features,
      .out_features = out_features,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor grad_output({batch_size, out_features}, DType_t::FP32, getHost());
  fill_normal(grad_output, 0.0, 0.5);
  Tensor grad_bias_temp({out_features}, DType_t::FP32, getHost());
  fill(grad_bias_temp, 0.0f);
  Tensor grad_bias({out_features}, DType_t::FP32, getHost());
  fill(grad_bias, 0.0f);

  void* handle = nullptr;
  WorkspaceReq req = engine_->query_dense_graph(handle, stats, type_desc);

  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getHost());

  engine_->dense_bgrad(handle, stats, grad_output.data_as<void>(), grad_bias.data_as<void>(),
                       workspace.data_as<void>(), type_desc);

  Tensor expected_grad_bias({out_features}, DType_t::FP32, getHost());

  math_dense_bgrad(grad_output.data_as<float>(), expected_grad_bias.data_as<float>(), batch_size,
                   out_features);

  compare_tensor(grad_bias, expected_grad_bias);
}

TEST_F(CPUEngineTest, Conv2DWgradReturnsCorrectResult) {
  Conv2DStats stats{
      .batch_size = 2,
      .in_channels = 3,
      .out_channels = 4,
      .input_h = 8,
      .input_w = 8,
      .kernel_h = 3,
      .kernel_w = 3,
      .stride_h = 1,
      .stride_w = 1,
      .pad_h = 1,
      .pad_w = 1,
      .use_bias = true,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
  size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;

  Tensor input({2, 8, 8, 3}, DType_t::FP32, getHost());
  fill_normal(input, 0.0, 0.5);
  Tensor grad_output({2, output_h, output_w, 4}, DType_t::FP32, getHost());
  fill_normal(grad_output, 0.0, 0.2);

  Tensor grad_weight_temp({4, 3, 3, 3}, DType_t::FP32, getHost());
  fill(grad_weight_temp, 0.0f);
  Tensor grad_weight({4, 3, 3, 3}, DType_t::FP32, getHost());
  fill(grad_weight, 0.0f);

  void* handle = nullptr;
  WorkspaceReq req = engine_->query_conv2d_graph(handle, stats, type_desc);
  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getHost());

  engine_->conv2d_wgrad(handle, stats, grad_output.data_as<void>(), input.data_as<void>(),
                        grad_weight.data_as<void>(), workspace.data_as<void>(), type_desc);

  Tensor expected_grad_weight({4, 3, 3, 3}, DType_t::FP32, getHost());
  fill(expected_grad_weight, 0.0f);
  math_conv2d_wgrad_naive(grad_output.data_as<float>(), input.data_as<float>(),
                          expected_grad_weight.data_as<float>(), stats.batch_size, stats.input_h,
                          stats.input_w, stats.in_channels, stats.out_channels, stats.kernel_h,
                          stats.kernel_w, stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w,
                          output_h, output_w);

  compare_tensor(grad_weight, expected_grad_weight);
}

TEST_F(CPUEngineTest, Conv2DBgradReturnsCorrectResult) {
  Conv2DStats stats{
      .batch_size = 2,
      .in_channels = 3,
      .out_channels = 4,
      .input_h = 8,
      .input_w = 8,
      .kernel_h = 3,
      .kernel_w = 3,
      .stride_h = 1,
      .stride_w = 1,
      .pad_h = 1,
      .pad_w = 1,
      .use_bias = true,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
  size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;

  Tensor grad_output({stats.batch_size, output_h, output_w, stats.out_channels}, DType_t::FP32,
                     getHost());
  fill_normal(grad_output, 0.0, 0.2);

  Tensor grad_bias_temp({stats.out_channels}, DType_t::FP32, getHost());
  fill(grad_bias_temp, 0.0f);
  Tensor grad_bias({stats.out_channels}, DType_t::FP32, getHost());
  fill(grad_bias, 0.0f);

  void* handle = nullptr;
  WorkspaceReq req = engine_->query_conv2d_graph(handle, stats, type_desc);
  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getHost());

  engine_->conv2d_bgrad(handle, stats, grad_output.data_as<void>(), grad_bias.data_as<void>(),
                        workspace.data_as<void>(), type_desc);

  Tensor expected_grad_bias({stats.out_channels}, DType_t::FP32, getHost());
  fill(expected_grad_bias, 0.0f);
  math_conv2d_bgrad_naive(grad_output.data_as<float>(), expected_grad_bias.data_as<float>(),
                          stats.batch_size, stats.out_channels, output_h, output_w);

  compare_tensor(grad_bias, expected_grad_bias);
}

TEST_F(CPUEngineTest, BatchNormBwdReturnsCorrectResult) {
  BatchNormStats stats{
      .batch_size = 2,
      .height = 8,
      .width = 8,
      .channels = 4,
      .epsilon = 1e-5,
      .momentum = 0.1,
      .use_relu = false,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor input({2, 8, 8, 4}, DType_t::FP32, getHost());
  fill_normal(input, 0.0, 0.5);
  Tensor grad_output({2, 8, 8, 4}, DType_t::FP32, getHost());
  fill_normal(grad_output, 0.0, 0.2);
  Tensor gamma({4}, DType_t::FP32, getHost());
  fill_normal(gamma, 1.0, 0.1);
  Tensor mean({4}, DType_t::FP32, getHost());
  fill(mean, 0.0f);
  Tensor invar({4}, DType_t::FP32, getHost());
  fill(invar, 1.0f);

  Tensor grad_input({2, 8, 8, 4}, DType_t::FP32, getHost());
  Tensor grad_gamma_temp({4}, DType_t::FP32, getHost());
  Tensor grad_beta_temp({4}, DType_t::FP32, getHost());
  fill(grad_gamma_temp, 0.0);
  fill(grad_beta_temp, 0.0);
  Tensor grad_gamma({4}, DType_t::FP32, getHost());
  fill(grad_gamma, 0.0);
  Tensor grad_beta({4}, DType_t::FP32, getHost());
  fill(grad_beta, 0.0);

  void* handle = nullptr;
  WorkspaceReq req = engine_->query_batchnorm_graph(handle, stats, type_desc);
  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getHost());

  engine_->batchnorm_bwd(handle, stats, grad_output.data_as<void>(), input.data_as<void>(), nullptr,
                         gamma.data_as<void>(), grad_input.data_as<void>(),
                         grad_gamma.data_as<void>(), grad_beta.data_as<void>(),
                         mean.data_as<void>(), invar.data_as<void>(), workspace.data_as<void>(),
                         type_desc);

  Tensor expected_grad_input({2, 8, 8, 4}, DType_t::FP32, getHost());
  Tensor expected_grad_gamma({4}, DType_t::FP32, getHost());
  Tensor expected_grad_beta({4}, DType_t::FP32, getHost());
  fill(expected_grad_gamma, 0.0);
  fill(expected_grad_beta, 0.0);

  math_batchnorm_bwd(grad_output.data_as<float>(), input.data_as<float>(), mean.data_as<float>(),
                     invar.data_as<float>(), gamma.data_as<float>(),
                     expected_grad_gamma.data_as<float>(), expected_grad_beta.data_as<float>(),
                     expected_grad_input.data_as<float>(), nullptr, stats.batch_size,
                     stats.channels, stats.height * stats.width, true, false);

  compare_tensor(grad_input, expected_grad_input);
  compare_tensor(grad_gamma, expected_grad_gamma);
  compare_tensor(grad_beta, expected_grad_beta);
}

TEST_F(CPUEngineTest, LayerNormBwdReturnsCorrectResult) {
  LayerNormStats stats{
      .batch_size = 2,
      .channels = 4,
      .epsilon = 1e-5f,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor input({stats.batch_size, stats.channels}, DType_t::FP32, getHost());
  fill_normal(input, 0.0, 0.5);
  Tensor grad_output({stats.batch_size, stats.channels}, DType_t::FP32, getHost());
  fill_normal(grad_output, 0.0, 0.2);
  Tensor gamma({stats.channels}, DType_t::FP32, getHost());
  fill_normal(gamma, 1.0, 0.1);

  Tensor beta({stats.channels}, DType_t::FP32, getHost());
  fill(beta, 0.0f);
  Tensor mean({stats.batch_size, 1}, DType_t::FP32, getHost());
  Tensor invar({stats.batch_size, 1}, DType_t::FP32, getHost());

  Tensor grad_input({stats.batch_size, stats.channels}, DType_t::FP32, getHost());
  Tensor grad_gamma_temp({stats.channels}, DType_t::FP32, getHost());
  Tensor grad_beta_temp({stats.channels}, DType_t::FP32, getHost());
  fill(grad_gamma_temp, 0.0f);
  fill(grad_beta_temp, 0.0f);
  Tensor grad_gamma({stats.channels}, DType_t::FP32, getHost());
  Tensor grad_beta({stats.channels}, DType_t::FP32, getHost());
  fill(grad_gamma, 0.0f);
  fill(grad_beta, 0.0f);

  void* handle = nullptr;
  WorkspaceReq req = engine_->query_layernorm_graph(handle, stats, type_desc);
  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getHost());

  engine_->layernorm_bwd(handle, stats, grad_output.data_as<void>(), input.data_as<void>(),
                         gamma.data_as<void>(), mean.data_as<void>(), invar.data_as<void>(),
                         grad_input.data_as<void>(), grad_gamma.data_as<void>(),
                         grad_beta.data_as<void>(), workspace.data_as<void>(), type_desc);

  Tensor expected_grad_input({stats.batch_size, stats.channels}, DType_t::FP32, getHost());
  Tensor expected_grad_gamma({stats.channels}, DType_t::FP32, getHost());
  Tensor expected_grad_beta({stats.channels}, DType_t::FP32, getHost());
  fill(expected_grad_gamma, 0.0f);
  fill(expected_grad_beta, 0.0f);

  math_layernorm_bwd(grad_output.data_as<float>(), input.data_as<float>(), gamma.data_as<float>(),
                     expected_grad_input.data_as<float>(), expected_grad_gamma.data_as<float>(),
                     expected_grad_beta.data_as<float>(), stats.batch_size, stats.channels,
                     static_cast<float>(stats.epsilon));

  compare_tensor(grad_input, expected_grad_input);
  compare_tensor(grad_gamma, expected_grad_gamma);
  compare_tensor(grad_beta, expected_grad_beta);
}

TEST_F(CPUEngineTest, AvgPoolFwdReturnsCorrectResult) {
  AvgPool2DStats stats{
      .batch_size = 2,
      .height = 4,
      .width = 4,
      .channels = 3,
      .pool_h = 2,
      .pool_w = 2,
      .stride_h = 2,
      .stride_w = 2,
      .pad_h = 0,
      .pad_w = 0,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  void* handle = nullptr;
  WorkspaceReq req = engine_->query_avgpool_graph(handle, stats, type_desc);
  Tensor workspace({req.fwd_workspace > 0 ? req.fwd_workspace : 1}, DType_t::BYTE, getHost());
  Tensor input({2, 4, 4, 3}, DType_t::FP32, getHost());
  fill_normal(input, 0.0, 0.5);
  Tensor output({2, 2, 2, 3}, DType_t::FP32, getHost());

  engine_->avgpool_fwd(handle, stats, input.data_as<void>(), output.data_as<void>(),
                       workspace.data_as<void>(), type_desc);

  Tensor expected_output({2, 2, 2, 3}, DType_t::FP32, getHost());
  math_avgpool_fwd(input.data_as<float>(), expected_output.data_as<float>(), stats.batch_size,
                   stats.height, stats.width, stats.channels, stats.pool_h, stats.pool_w,
                   stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w, 2, 2);

  compare_tensor(output, expected_output);
}

TEST_F(CPUEngineTest, AvgPoolBwdReturnsCorrectResult) {
  AvgPool2DStats stats{
      .batch_size = 2,
      .height = 4,
      .width = 4,
      .channels = 3,
      .pool_h = 2,
      .pool_w = 2,
      .stride_h = 2,
      .stride_w = 2,
      .pad_h = 0,
      .pad_w = 0,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  void* handle = nullptr;
  WorkspaceReq req = engine_->query_avgpool_graph(handle, stats, type_desc);
  Tensor workspace({req.bwd_workspace > 0 ? req.bwd_workspace : 1}, DType_t::BYTE, getHost());
  Tensor grad_output({2, 2, 2, 3}, DType_t::FP32, getHost());
  fill_normal(grad_output, 0.0, 0.5);
  Tensor grad_input({2, 4, 4, 3}, DType_t::FP32, getHost());
  fill(grad_input, 0.0f);

  engine_->avgpool_bwd(handle, stats, grad_output.data_as<void>(), grad_input.data_as<void>(),
                       workspace.data_as<void>(), type_desc);

  Tensor expected_grad_input({2, 4, 4, 3}, DType_t::FP32, getHost());
  fill(expected_grad_input, 0.0f);
  math_avgpool_bwd(grad_output.data_as<float>(), expected_grad_input.data_as<float>(),
                   stats.batch_size, stats.height, stats.width, stats.channels, stats.pool_h,
                   stats.pool_w, stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w, 2, 2);

  compare_tensor(grad_input, expected_grad_input);
}

TEST_F(CPUEngineTest, MaxPoolFwdReturnsCorrectResult) {
  MaxPool2DStats stats{
      .batch_size = 1,
      .height = 4,
      .width = 4,
      .channels = 1,
      .pool_h = 2,
      .pool_w = 2,
      .stride_h = 2,
      .stride_w = 2,
      .pad_h = 0,
      .pad_w = 0,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  void* handle = nullptr;
  WorkspaceReq req = engine_->query_maxpool2d_graph(handle, stats, type_desc);
  Tensor workspace({req.fwd_workspace > 0 ? req.fwd_workspace : 1}, DType_t::BYTE, getHost());
  Tensor input({1, 4, 4, 1}, DType_t::FP32, getHost());
  float input_host[] = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  std::memcpy(input.data_as<void>(), input_host, sizeof(input_host));
  Tensor output({1, 2, 2, 1}, DType_t::FP32, getHost());
  Tensor mask({1, 2, 2, 1}, DType_t::INT32, getHost());

  engine_->maxpool2d_fwd(handle, stats, input.data_as<void>(), output.data_as<void>(),
                         mask.data_as<void>(), workspace.data_as<void>(), type_desc);

  Tensor expected_output({1, 2, 2, 1}, DType_t::FP32, getHost());
  Tensor expected_mask({1, 2, 2, 1}, DType_t::INT32, getHost());
  math_maxpool2d_fwd(input.data_as<float>(), expected_output.data_as<float>(),
                     expected_mask.data_as<int32_t>(), stats.batch_size, stats.height, stats.width,
                     stats.channels, stats.pool_h, stats.pool_w, stats.stride_h, stats.stride_w,
                     stats.pad_h, stats.pad_w, 2, 2);

  compare_tensor(output, expected_output);
  compare_array_t<bool>(mask.data_as<bool>(), expected_mask.data_as<bool>(), mask.size());
}

TEST_F(CPUEngineTest, MaxPoolBwdReturnsCorrectResult) {
  MaxPool2DStats stats{
      .batch_size = 2,
      .height = 4,
      .width = 4,
      .channels = 3,
      .pool_h = 2,
      .pool_w = 2,
      .stride_h = 2,
      .stride_w = 2,
      .pad_h = 0,
      .pad_w = 0,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  void* handle = nullptr;
  WorkspaceReq req = engine_->query_maxpool2d_graph(handle, stats, type_desc);
  Tensor workspace({req.bwd_workspace > 0 ? req.bwd_workspace : 1}, DType_t::BYTE, getHost());

  Tensor input({2, 4, 4, 3}, DType_t::FP32, getHost());
  fill_normal(input, 0.0, 0.5);
  Tensor output({2, 2, 2, 3}, DType_t::FP32, getHost());
  Tensor mask({2, 2, 2, 3}, DType_t::INT32, getHost());

  engine_->maxpool2d_fwd(handle, stats, input.data_as<void>(), output.data_as<void>(),
                         mask.data_as<void>(), workspace.data_as<void>(), type_desc);

  Tensor grad_output({2, 2, 2, 3}, DType_t::FP32, getHost());
  fill_normal(grad_output, 0.0, 0.5);
  Tensor grad_input({2, 4, 4, 3}, DType_t::FP32, getHost());
  fill(grad_input, 0.0f);

  engine_->maxpool2d_bwd(handle, stats, grad_output.data_as<void>(), grad_input.data_as<void>(),
                         mask.data_as<void>(), workspace.data_as<void>(), type_desc);

  Tensor expected_grad_input({2, 4, 4, 3}, DType_t::FP32, getHost());
  fill(expected_grad_input, 0.0f);
  math_maxpool2d_bwd(grad_output.data_as<float>(), expected_grad_input.data_as<float>(),
                     mask.data_as<int32_t>(), stats.batch_size, stats.channels, 2, 2, stats.height,
                     stats.width, stats.pool_w, stats.stride_h, stats.stride_w, stats.pad_h,
                     stats.pad_w);

  compare_tensor(grad_input, expected_grad_input);
}

TEST_F(CPUEngineTest, ClassTokenFwdReturnsCorrectResult) {
  ClassTokenStats stats{
      .batch_size = 2,
      .seq_len = 3,
      .embed_dim = 4,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  void* handle = nullptr;
  WorkspaceReq req = engine_->query_class_token_graph(handle, stats, type_desc);
  Tensor workspace({req.fwd_workspace > 0 ? req.fwd_workspace : 1}, DType_t::BYTE, getHost());

  Tensor input({2, 3, 4}, DType_t::FP32, getHost());
  fill_normal(input, 0.0, 0.5);
  Tensor token({2, 4}, DType_t::FP32, getHost());
  Tensor output({2, 4, 4}, DType_t::FP32, getHost());

  engine_->class_token_fwd(handle, stats, input.data_as<void>(), token.data_as<void>(),
                           output.data_as<void>(), workspace.data_as<void>(), type_desc);

  Tensor expected_output({2, 4, 4}, DType_t::FP32, getHost());
  math_class_token_fwd(input.data_as<float>(), token.data_as<float>(),
                       expected_output.data_as<float>(), stats.batch_size, stats.seq_len,
                       stats.embed_dim);

  compare_tensor(output, expected_output);
}

TEST_F(CPUEngineTest, ClassTokenBwdReturnsCorrectResult) {
  ClassTokenStats stats{
      .batch_size = 2,
      .seq_len = 3,
      .embed_dim = 4,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  void* handle = nullptr;
  WorkspaceReq req = engine_->query_class_token_graph(handle, stats, type_desc);
  Tensor workspace({req.bwd_workspace > 0 ? req.bwd_workspace : 1}, DType_t::BYTE, getHost());

  Tensor grad_output({2, 4, 4}, DType_t::FP32, getHost());
  fill_normal(grad_output, 0.0, 0.2);
  Tensor grad_input({2, 3, 4}, DType_t::FP32, getHost());
  Tensor grad_token_temp({4}, DType_t::FP32, getHost());
  fill(grad_token_temp, 0.0f);
  Tensor grad_token({4}, DType_t::FP32, getHost());
  fill(grad_token, 0.0f);

  engine_->class_token_bwd(handle, stats, grad_output.data_as<void>(), grad_input.data_as<void>(),
                           grad_token.data_as<void>(), workspace.data_as<void>(), type_desc);

  Tensor expected_grad_input({2, 3, 4}, DType_t::FP32, getHost());
  Tensor expected_grad_token({4}, DType_t::FP32, getHost());
  fill(expected_grad_token, 0.0f);
  math_class_token_bwd(grad_output.data_as<float>(), expected_grad_input.data_as<float>(),
                       expected_grad_token.data_as<float>(), stats.batch_size, stats.seq_len,
                       stats.embed_dim);

  compare_tensor(grad_input, expected_grad_input);
  compare_tensor(grad_token, expected_grad_token);
}

TEST_F(CPUEngineTest, EmbeddingFwdReturnsCorrectResult) {
  EmbeddingStats stats{
      .num_indices = 4,
      .vocab_size = 10,
      .embed_dim = 8,
      .padding_idx = 0,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  void* handle = nullptr;
  WorkspaceReq req = engine_->query_embedding_graph(handle, stats, type_desc);
  Tensor workspace({req.fwd_workspace > 0 ? req.fwd_workspace : 1}, DType_t::BYTE, getHost());

  Tensor input({4}, DType_t::FP32, getHost());
  float host_input[] = {1.0f, 2.0f, 1.0f, 0.0f};
  std::memcpy(input.data_as<void>(), host_input, sizeof(host_input));
  Tensor weight({10, 8}, DType_t::FP32, getHost());
  fill_normal(weight, 0.0, 0.5);
  Tensor output({4, 8}, DType_t::FP32, getHost());

  engine_->embedding_fwd(handle, stats, input.data_as<void>(), weight.data_as<void>(),
                         output.data_as<void>(), workspace.data_as<void>(), type_desc);

  Tensor expected_output({4, 8}, DType_t::FP32, getHost());
  math_embedding_fwd(input.data_as<float>(), weight.data_as<float>(),
                     expected_output.data_as<float>(), stats.num_indices, stats.vocab_size,
                     stats.embed_dim, stats.padding_idx);

  compare_tensor(output, expected_output);
}

TEST_F(CPUEngineTest, EmbeddingBwdReturnsCorrectResult) {
  EmbeddingStats stats{
      .num_indices = 4,
      .vocab_size = 10,
      .embed_dim = 8,
      .padding_idx = 0,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  void* handle = nullptr;
  WorkspaceReq req = engine_->query_embedding_graph(handle, stats, type_desc);
  Tensor workspace({req.bwd_workspace > 0 ? req.bwd_workspace : 1}, DType_t::BYTE, getHost());

  Tensor input({4}, DType_t::FP32, getHost());
  float host_input[] = {1.0f, 2.0f, 1.0f, 0.0f};
  std::memcpy(input.data_as<void>(), host_input, sizeof(host_input));
  Tensor grad_output({4, 8}, DType_t::FP32, getHost());
  fill_normal(grad_output, 0.0, 0.2);
  Tensor grad_weight_temp({10, 8}, DType_t::FP32, getHost());
  fill(grad_weight_temp, 0.0f);
  Tensor grad_weight({10, 8}, DType_t::FP32, getHost());
  fill(grad_weight, 0.0f);

  engine_->embedding_bwd(handle, stats, grad_output.data_as<void>(), input.data_as<void>(),
                         grad_weight.data_as<void>(), workspace.data_as<void>(), type_desc);

  Tensor expected_grad_weight({10, 8}, DType_t::FP32, getHost());
  fill(expected_grad_weight, 0.0f);
  math_embedding_bwd(input.data_as<float>(), grad_output.data_as<float>(),
                     expected_grad_weight.data_as<float>(), stats.num_indices, stats.vocab_size,
                     stats.embed_dim, stats.padding_idx);

  compare_tensor(grad_weight, expected_grad_weight);
}

TEST_F(CPUEngineTest, ReLUFwdReturnsCorrectResult) {
  ReLUStats stats{
      .batch_size = 16,
      .spatial_size = 256,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  void* handle = nullptr;
  WorkspaceReq req = engine_->query_relu_graph(handle, stats, type_desc);
  size_t ws_size = req.fwd_workspace > 0 ? req.fwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getHost());

  Tensor input({16, 256}, DType_t::FP32, getHost());
  fill_normal(input, 0.0, 1.0);
  Tensor output({16, 256}, DType_t::FP32, getHost());
  Tensor mask({16, 256}, DType_t::BOOL, getHost());

  engine_->relu_fwd(handle, stats, input.data_as<void>(), output.data_as<void>(),
                    mask.data_as<bool>(), workspace.data_as<void>(), type_desc);

  Tensor expected_output({16, 256}, DType_t::FP32, getHost());
  Tensor expected_mask({16, 256}, DType_t::BOOL, getHost());
  math_relu_fwd(input.data_as<float>(), expected_output.data_as<float>(),
                expected_mask.data_as<bool>(), 16 * 256);

  compare_tensor(output, expected_output);
  compare_array_t<bool>(mask.data_as<bool>(), expected_mask.data_as<bool>(), mask.size());
}

TEST_F(CPUEngineTest, ReLUBwdReturnsCorrectResult) {
  ReLUStats stats{
      .batch_size = 16,
      .spatial_size = 256,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  void* handle = nullptr;
  WorkspaceReq req = engine_->query_relu_graph(handle, stats, type_desc);
  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getHost());

  Tensor grad_output({16, 256}, DType_t::FP32, getHost());
  fill(grad_output, 1.0f);
  Tensor mask({16, 256}, DType_t::BOOL, getHost());

  // Fill mask deterministically: every other element is active
  size_t mask_elements = 16 * 256;
  std::vector<uint8_t> mask_raw(mask_elements);
  for (size_t i = 0; i < mask_elements; ++i) mask_raw[i] = static_cast<uint8_t>(i % 2 == 0);
  std::memcpy(mask.data_as<bool>(), mask_raw.data(), mask_elements * sizeof(bool));

  Tensor grad_input({16, 256}, DType_t::FP32, getHost());

  engine_->relu_bwd(handle, stats, grad_output.data_as<void>(), grad_input.data_as<void>(),
                    mask.data_as<bool>(), workspace.data_as<void>(), type_desc);

  Tensor expected_grad_input({16, 256}, DType_t::FP32, getHost());
  math_relu_bwd(grad_output.data_as<float>(), expected_grad_input.data_as<float>(),
                mask.data_as<bool>(), 16 * 256);

  compare_tensor(grad_input, expected_grad_input);
}

TEST_F(CPUEngineTest, DropoutFwdReturnsCorrectResult) {
  DropoutStats stats{
      .batch_size = 2,
      .channels = 3,
      .spatial_size = 4,
      .dropout_rate = 0.5,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  void* handle = nullptr;
  WorkspaceReq req = engine_->query_dropout_graph(handle, stats, type_desc);
  Tensor workspace({req.fwd_workspace > 0 ? req.fwd_workspace : 1}, DType_t::BYTE, getHost());
  Tensor input({2, 3, 2, 2}, DType_t::FP32, getHost());
  fill_normal(input, 0.0, 0.5);
  Tensor output({2, 3, 2, 2}, DType_t::FP32, getHost());
  Tensor mask({2, 3, 2, 2}, DType_t::BOOL, getHost());

  EXPECT_NO_THROW({
    engine_->dropout_fwd(handle, stats, input.data_as<void>(), output.data_as<void>(),
                         mask.data_as<bool>(), workspace.data_as<void>(), type_desc);
  });

  Tensor expected_output({2, 3, 2, 2}, DType_t::FP32, getHost());
  math_dropout_fwd(input.data_as<float>(), expected_output.data_as<float>(), mask.data_as<bool>(),
                   stats.batch_size, stats.channels, stats.spatial_size,
                   static_cast<float>(stats.dropout_rate));

  compare_tensor(output, expected_output);
}

TEST_F(CPUEngineTest, DropoutBwdReturnsCorrectResult) {
  size_t batch_size = 2;
  size_t channels = 3;
  size_t spatial_size = 4;
  double dropout_rate = 0.5;
  float bwd_scale = 2.0f;
  size_t spatial_h = 2;
  size_t spatial_w = 2;

  DropoutStats stats{
      .batch_size = batch_size,
      .channels = channels,
      .spatial_size = spatial_size,
      .dropout_rate = dropout_rate,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  void* handle = nullptr;
  WorkspaceReq req = engine_->query_dropout_graph(handle, stats, type_desc);
  Tensor workspace({req.bwd_workspace > 0 ? req.bwd_workspace : 1}, DType_t::BYTE, getHost());

  Tensor grad_output({batch_size, channels, spatial_h, spatial_w}, DType_t::FP32, getHost());
  fill_normal(grad_output, 0.0, 0.5);
  Tensor grad_input({batch_size, channels, spatial_h, spatial_w}, DType_t::FP32, getHost());
  Tensor mask({batch_size, channels, spatial_h, spatial_w}, DType_t::BOOL, getHost());

  size_t mask_elements = batch_size * channels * spatial_h * spatial_w;
  std::vector<uint8_t> mask_raw(mask_elements);
  for (size_t i = 0; i < mask_elements; ++i) mask_raw[i] = static_cast<uint8_t>(i % 2 == 0);
  std::memcpy(mask.data_as<bool>(), mask_raw.data(), mask_elements * sizeof(bool));

  EXPECT_NO_THROW({
    engine_->dropout_bwd(handle, stats, grad_output.data_as<void>(), grad_input.data_as<void>(),
                         mask.data_as<bool>(), bwd_scale, workspace.data_as<void>(), type_desc);
  });

  Tensor expected_grad_input({batch_size, channels, spatial_h, spatial_w}, DType_t::FP32,
                             getHost());
  math_dropout_bwd(grad_output.data_as<float>(), expected_grad_input.data_as<float>(),
                   mask.data_as<bool>(), batch_size, channels, spatial_size, bwd_scale);

  compare_tensor(grad_input, expected_grad_input);
}
