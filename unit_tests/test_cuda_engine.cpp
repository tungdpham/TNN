/*
 * Copyright (c) 2026 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <gtest/gtest.h>

#include "device/device_manager.hpp"
#include "engine_test_utils.hpp"
#include "nn/engines/cuda_engine.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"

using namespace tunx;

#ifdef USE_CUDA

class CUDAEngineTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    initializeDefaultDevices();
    DeviceManager& manager = DeviceManager::getInstance();
    Vec<std::string> device_ids = manager.getAvailableDeviceIDs();

    has_gpu_ = false;
    for (const std::string& id : device_ids) {
      const Device& device = manager.getDevice(id);
      if (device.device_type() == DeviceType::CUDA) {
        has_gpu_ = true;
        break;
      }
    }

    if (!has_gpu_) {
      GTEST_SKIP() << "No CUDA device available, skipping CUDA engine tests";
    }

    engine_ = std::make_unique<CUDAEngine>();
    handle_ = engine_->create_backend_handle();
  }

  static void TearDownTestSuite() {
    engine_.reset();
    handle_ = nullptr;
  }

  static bool has_gpu_;
  static std::unique_ptr<CUDAEngine> engine_;
  static void* handle_;
};

bool CUDAEngineTest::has_gpu_ = false;
std::unique_ptr<CUDAEngine> CUDAEngineTest::engine_;
void* CUDAEngineTest::handle_ = nullptr;

TEST_F(CUDAEngineTest, DenseFwdReturnsCorrectResults) {
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

  Tensor input({batch_size, in_features}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 0.5);
  Tensor weight({out_features, in_features}, DType_t::FP32, getGPU());
  fill_normal(weight, 0.0, 0.1);
  Tensor bias({out_features}, DType_t::FP32, getGPU());
  fill_normal(bias, 0.0, 0.1);
  Tensor output({batch_size, out_features}, DType_t::FP32, getGPU());

  WorkspaceReq req = engine_->query_dense_graph(handle_, stats, type_desc);

  size_t ws_size = req.fwd_workspace > 0 ? req.fwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getGPU());

  engine_->dense_fwd(handle_, stats, input.data_as<void>(), weight.data_as<void>(),
                     bias.data_as<void>(), output.data_as<void>(), workspace.data_as<void>(),
                     type_desc);
  cudaDeviceSynchronize();

  Tensor expected_output({batch_size, out_features}, DType_t::FP32, getHost());

  Tensor input_host = input.to_host();
  Tensor weight_host = weight.to_host();
  Tensor bias_host = bias.to_host();

  math_dense_fwd(input_host.data_as<float>(), weight_host.data_as<float>(),
                 bias_host.data_as<float>(), expected_output.data_as<float>(), batch_size,
                 in_features, out_features);

  compare_tensor(output.to_host(), expected_output);
}

TEST_F(CUDAEngineTest, DenseWgradReturnsCorrectResults) {
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

  Tensor input({batch_size, in_features}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 0.5);
  Tensor grad_output({batch_size, out_features}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.2);
  Tensor grad_weight_temp({out_features, in_features}, DType_t::FP32, getGPU());
  fill(grad_weight_temp, 0.0f);
  Tensor grad_weight({out_features, in_features}, DType_t::FP32, getGPU());
  fill(grad_weight, 0.0f);

  WorkspaceReq req = engine_->query_dense_graph(handle_, stats, type_desc);

  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getGPU());

  engine_->dense_wgrad(handle_, stats, grad_output.data_as<void>(), input.data_as<void>(),
                       grad_weight.data_as<void>(), workspace.data_as<void>(), type_desc);
  cudaDeviceSynchronize();

  Tensor expected_grad_weight({out_features, in_features}, DType_t::FP32, getHost());

  Tensor input_host = input.to_host();
  Tensor grad_out_host = grad_output.to_host();

  math_dense_wgrad(input_host.data_as<float>(), grad_out_host.data_as<float>(),
                   expected_grad_weight.data_as<float>(), batch_size, in_features, out_features);

  compare_tensor(grad_weight.to_host(), expected_grad_weight);
}

TEST_F(CUDAEngineTest, DenseDgradReturnsCorrectResults) {
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

  Tensor grad_output({batch_size, out_features}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.5);
  Tensor weight({out_features, in_features}, DType_t::FP32, getGPU());
  fill_normal(weight, 0.0, 0.1);
  Tensor grad_input({batch_size, in_features}, DType_t::FP32, getGPU());

  WorkspaceReq req = engine_->query_dense_graph(handle_, stats, type_desc);

  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getGPU());

  engine_->dense_dgrad(handle_, stats, grad_output.data_as<void>(), weight.data_as<void>(),
                       grad_input.data_as<void>(), workspace.data_as<void>(), type_desc);
  cudaDeviceSynchronize();

  Tensor expected_grad_input({batch_size, in_features}, DType_t::FP32, getHost());

  Tensor grad_out_host = grad_output.to_host();
  Tensor weight_host = weight.to_host();

  math_dense_dgrad(grad_out_host.data_as<float>(), weight_host.data_as<float>(),
                   expected_grad_input.data_as<float>(), batch_size, in_features, out_features);

  compare_tensor(grad_input.to_host(), expected_grad_input);
}

TEST_F(CUDAEngineTest, DenseBgradReturnsCorrectResults) {
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

  Tensor grad_output({batch_size, out_features}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.5);
  Tensor grad_bias_temp({out_features}, DType_t::FP32, getGPU());
  fill(grad_bias_temp, 0.0f);
  Tensor grad_bias({out_features}, DType_t::FP32, getGPU());
  fill(grad_bias, 0.0f);

  WorkspaceReq req = engine_->query_dense_graph(handle_, stats, type_desc);

  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getGPU());

  engine_->dense_bgrad(handle_, stats, grad_output.data_as<void>(), grad_bias.data_as<void>(),
                       workspace.data_as<void>(), type_desc);
  cudaDeviceSynchronize();

  Tensor expected_grad_bias({out_features}, DType_t::FP32, getHost());

  Tensor grad_out_host = grad_output.to_host();

  math_dense_bgrad(grad_out_host.data_as<float>(), expected_grad_bias.data_as<float>(), batch_size,
                   out_features);

  compare_tensor(grad_bias.to_host(), expected_grad_bias);
}

#endif  // USE_CUDA
