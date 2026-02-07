/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/cudnn_layer_norm_ops.hpp"

#ifdef USE_CUDNN

#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <string>

#include "type/type.hpp"

namespace tnn {
namespace cuda {
namespace cudnn_layer_norm {

namespace fe = cudnn_frontend;

struct feHandle_t {
  cudnnHandle_t cudnn_handle = nullptr;
  cudnnDataType_t io_data_type;
  cudnnDataType_t compute_data_type;
  std::shared_ptr<fe::KernelCache> kernel_cache;

  std::shared_ptr<fe::graph::Graph> fwd_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_x;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_scale;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_bias;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_y;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_mean;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_inv_variance;

  std::shared_ptr<fe::graph::Graph> bwd_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_dy;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_x;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_scale;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_mean;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_inv_variance;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_dx;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_dscale;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_dbias;
};

cudnnDataType_t get_cudnn_data_type(DType_t dtype) {
  switch (dtype) {
    case DType_t::FP32:
      return CUDNN_DATA_FLOAT;
    case DType_t::FP16:
      return CUDNN_DATA_HALF;
    case DType_t::FP64:
      return CUDNN_DATA_DOUBLE;
    case DType_t::BF16:
      return CUDNN_DATA_BFLOAT16;
    default:
      throw std::runtime_error("Unsupported data type for cuDNN LayerNorm");
  }
}

static void ensure_ok(fe::error_t status, std::string stage) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string("cuDNN frontend error at ") + stage + ": " +
                             status.get_message());
  }
}

static fe::DataType_t to_fe_data_type(cudnnDataType_t data_type) {
  switch (data_type) {
    case CUDNN_DATA_HALF:
      return fe::DataType_t::HALF;
    case CUDNN_DATA_FLOAT:
      return fe::DataType_t::FLOAT;
    case CUDNN_DATA_DOUBLE:
      return fe::DataType_t::DOUBLE;
    case CUDNN_DATA_BFLOAT16:
      return fe::DataType_t::BFLOAT16;
    default:
      throw std::runtime_error("Unsupported cuDNN data type for LayerNorm");
  }
}

static fe::DataType_t to_fe_compute_type(cudnnDataType_t data_type) {
  if (data_type == CUDNN_DATA_HALF || data_type == CUDNN_DATA_BFLOAT16) {
    return fe::DataType_t::FLOAT;
  }
  return to_fe_data_type(data_type);
}

static void build_fwd_graph(feHandle_t* handle, LayerNormStats& stats) {
  const int64_t n = static_cast<int64_t>(stats.batch_size);
  const int64_t c = static_cast<int64_t>(stats.channels);

  auto io_type = to_fe_data_type(handle->io_data_type);
  auto compute_type = to_fe_compute_type(handle->compute_data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type);

  // Input tensor: [N, C, 1, 1] with strides for cuDNN compatibility
  auto X = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("input")
                             .set_dim({n, c, 1, 1})
                             .set_stride({c, 1, c, c})
                             .set_data_type(io_type));

  auto epsilon = graph->tensor(stats.epsilon);

  auto ln_options = fe::graph::Layernorm_attributes()
                        .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                        .set_epsilon(epsilon);

  // Scale (gamma) and Bias (beta) tensors: [1, C, 1, 1]
  // cuDNN layernorm always requires scale and bias
  auto Scale = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("scale")
                                 .set_dim({1, c, 1, 1})
                                 .set_stride({c, 1, c, c})
                                 .set_data_type(io_type));

  auto Bias = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("bias")
                                .set_dim({1, c, 1, 1})
                                .set_stride({c, 1, c, c})
                                .set_data_type(io_type));

  auto [Y, Mean, InvVariance] = graph->layernorm(X, Scale, Bias, ln_options);
  Y->set_output(true).set_data_type(io_type);
  Mean->set_output(true).set_data_type(compute_type);
  InvVariance->set_output(true).set_data_type(compute_type);

  ensure_ok(graph->validate(), "layernorm_fwd validate");
  ensure_ok(graph->build_operation_graph(handle->cudnn_handle), "layernorm_fwd build op graph");
  ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A}), "layernorm_fwd create plans");
  ensure_ok(graph->check_support(), "layernorm_fwd check support");
  ensure_ok(graph->build_plans(), "layernorm_fwd build plans");

  int64_t workspace_size = 0;
  ensure_ok(graph->get_workspace_size(workspace_size), "layernorm_fwd workspace");

  handle->fwd_graph = graph;
  handle->fwd_x = X;
  handle->fwd_scale = Scale;
  handle->fwd_bias = Bias;
  handle->fwd_y = Y;
  handle->fwd_mean = Mean;
  handle->fwd_inv_variance = InvVariance;
  stats.fwd_workspace_size = static_cast<size_t>(workspace_size);
}

static void build_bwd_graph(feHandle_t* handle, LayerNormStats& stats) {
  const int64_t n = static_cast<int64_t>(stats.batch_size);
  const int64_t c = static_cast<int64_t>(stats.channels);

  auto io_type = to_fe_data_type(handle->io_data_type);
  auto compute_type = to_fe_compute_type(handle->compute_data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type);

  // Gradient output: [N, C, 1, 1]
  auto DY = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("grad_output")
                              .set_dim({n, c, 1, 1})
                              .set_stride({c, 1, c, c})
                              .set_data_type(io_type));

  // Input from forward pass: [N, C, 1, 1]
  auto X = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("input")
                             .set_dim({n, c, 1, 1})
                             .set_stride({c, 1, c, c})
                             .set_data_type(io_type));

  // Mean and inv_variance from forward pass: [N, 1, 1, 1]
  auto Mean = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("mean")
                                .set_dim({n, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_data_type(compute_type));

  auto InvVariance = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("inv_variance")
                                       .set_dim({n, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(compute_type));

  auto ln_bwd_options =
      fe::graph::Layernorm_backward_attributes().set_saved_mean_and_inv_variance(Mean, InvVariance);

  // cuDNN layernorm_backward always requires scale and returns dscale, dbias
  auto Scale = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("scale")
                                 .set_dim({1, c, 1, 1})
                                 .set_stride({c, 1, c, c})
                                 .set_data_type(io_type));

  auto [DX, DScale, DBias] = graph->layernorm_backward(DY, X, Scale, ln_bwd_options);

  DX->set_output(true).set_data_type(io_type);
  DScale->set_output(true).set_data_type(io_type);
  DBias->set_output(true).set_data_type(io_type);

  ensure_ok(graph->validate(), "layernorm_bwd validate");
  ensure_ok(graph->build_operation_graph(handle->cudnn_handle), "layernorm_bwd build op graph");
  ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A}), "layernorm_bwd create plans");
  ensure_ok(graph->check_support(), "layernorm_bwd check support");
  ensure_ok(graph->build_plans(), "layernorm_bwd build plans");

  int64_t workspace_size = 0;
  ensure_ok(graph->get_workspace_size(workspace_size), "layernorm_bwd workspace");

  handle->bwd_graph = graph;
  handle->bwd_dy = DY;
  handle->bwd_x = X;
  handle->bwd_mean = Mean;
  handle->bwd_inv_variance = InvVariance;
  handle->bwd_scale = Scale;
  handle->bwd_dx = DX;
  handle->bwd_dscale = DScale;
  handle->bwd_dbias = DBias;
  stats.bwd_workspace_size = static_cast<size_t>(workspace_size);
}

static void rebuild_all_graphs(feHandle_t* handle, LayerNormStats& stats) {
  build_fwd_graph(handle, stats);
  build_bwd_graph(handle, stats);
}

feHandle_t* initialize_fe_handle(cudnnHandle_t cudnn_handle, cudnnDataType_t io_data_type,
                                 cudnnDataType_t compute_data_type, LayerNormStats& stats) {
  auto* handle = new feHandle_t();
  handle->cudnn_handle = cudnn_handle;
  handle->io_data_type = io_data_type;
  handle->compute_data_type = compute_data_type;
  handle->kernel_cache = std::make_shared<fe::KernelCache>();

  rebuild_all_graphs(handle, stats);
  round_workspace_size(stats);

  return handle;
}

void destroy_fe_handle(feHandle_t* handle) {
  if (!handle) {
    return;
  }
  delete handle;
}

void run_forward(feHandle_t* handle, const LayerNormStats& stats, const void* input_data,
                 const void* gamma_data, const void* beta_data, void* output_data, void* mean_data,
                 void* inv_variance_data, void* workspace_data, cudaStream_t stream) {
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;

  variant_pack[handle->fwd_x] = const_cast<void*>(input_data);
  variant_pack[handle->fwd_scale] = const_cast<void*>(gamma_data);
  variant_pack[handle->fwd_bias] = const_cast<void*>(beta_data);
  variant_pack[handle->fwd_y] = output_data;
  variant_pack[handle->fwd_mean] = mean_data;
  variant_pack[handle->fwd_inv_variance] = inv_variance_data;

  cudnnHandle_t cudnn_handle = handle->cudnn_handle;
  cudnnSetStream(cudnn_handle, stream);

  ensure_ok(handle->fwd_graph->execute(cudnn_handle, variant_pack, workspace_data),
            "layernorm_fwd execute");
}

void run_backward(feHandle_t* handle, const LayerNormStats& stats, const void* gradient_data,
                  const void* input_data, const void* gamma_data, const void* mean_data,
                  const void* inv_variance_data, void* grad_input_data, void* gamma_grad_data,
                  void* beta_grad_data, void* workspace_data, cudaStream_t stream) {
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;

  variant_pack[handle->bwd_dy] = const_cast<void*>(gradient_data);
  variant_pack[handle->bwd_x] = const_cast<void*>(input_data);
  variant_pack[handle->bwd_mean] = const_cast<void*>(mean_data);
  variant_pack[handle->bwd_inv_variance] = const_cast<void*>(inv_variance_data);
  variant_pack[handle->bwd_scale] = const_cast<void*>(gamma_data);
  variant_pack[handle->bwd_dx] = grad_input_data;
  variant_pack[handle->bwd_dscale] = gamma_grad_data;
  variant_pack[handle->bwd_dbias] = beta_grad_data;

  cudnnHandle_t cudnn_handle = handle->cudnn_handle;
  cudnnSetStream(cudnn_handle, stream);

  ensure_ok(handle->bwd_graph->execute(cudnn_handle, variant_pack, workspace_data),
            "layernorm_bwd execute");
}

}  // namespace cudnn_layer_norm
}  // namespace cuda
}  // namespace tnn

#endif
