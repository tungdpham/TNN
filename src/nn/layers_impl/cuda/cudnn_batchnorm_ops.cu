#include "nn/layers_impl/cuda/cudnn_batchnorm_ops.hpp"
#include <cudnn_graph.h>

#ifdef USE_CUDNN

#include "cuda/cudnn/common.hpp"
#include "type/type.hpp"

#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace tnn {
namespace cuda {
namespace cudnn_batchnorm {

namespace fe = cudnn_frontend;

struct feHandle_t {
  cudnnHandle_t cudnn_handle;
  cudnnDataType_t data_type;
  std::shared_ptr<fe::KernelCache> kernel_cache;

  std::shared_ptr<fe::graph::Graph> fwd_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_x;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_scale;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_bias;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_y;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_mean;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_invar;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_prev_mean;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_prev_var;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_next_mean;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_next_var;

  std::shared_ptr<fe::graph::Graph> inf_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> inf_x;
  std::shared_ptr<fe::graph::Tensor_attributes> inf_scale;
  std::shared_ptr<fe::graph::Tensor_attributes> inf_bias;
  std::shared_ptr<fe::graph::Tensor_attributes> inf_saved_mean;
  std::shared_ptr<fe::graph::Tensor_attributes> inf_saved_invar;
  std::shared_ptr<fe::graph::Tensor_attributes> inf_y;

  std::shared_ptr<fe::graph::Graph> bwd_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_dy;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_x;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_scale;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_mean;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_invar;
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
  case DType_t::BF16:
    return CUDNN_DATA_BFLOAT16;
  case DType_t::FP64:
    return CUDNN_DATA_DOUBLE;
  default:
    throw std::runtime_error("Unsupported data type for cuDNN BatchNorm");
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
    throw std::runtime_error("Unsupported cuDNN data type for batchnorm");
  }
}

static fe::DataType_t to_fe_compute_type(cudnnDataType_t data_type) {
  if (data_type == CUDNN_DATA_HALF || data_type == CUDNN_DATA_BFLOAT16) {
    return fe::DataType_t::FLOAT;
  }
  return to_fe_data_type(data_type);
}

static void build_fwd_graph(feHandle_t *handle, BatchNormStats &stats) {
  const int64_t n = static_cast<int64_t>(stats.batch_size);
  const int64_t c = static_cast<int64_t>(stats.channels);
  const int64_t h = static_cast<int64_t>(stats.height);
  const int64_t w = static_cast<int64_t>(stats.width);
  auto io_type = to_fe_data_type(handle->data_type);
  auto compute_type = to_fe_compute_type(handle->data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type);

  auto X = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("X")
                             .set_dim({n, c, h, w})
                             .set_stride({h * w * c, 1, w * c, c})
                             .set_data_type(io_type));

  auto scale = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("scale")
                                 .set_dim({1, c, 1, 1})
                                 .set_stride({c, 1, c, c})
                                 .set_data_type(compute_type));

  auto bias = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("bias")
                                .set_dim({1, c, 1, 1})
                                .set_stride({c, 1, c, c})
                                .set_data_type(compute_type));

  auto prev_running_mean = graph->tensor(fe::graph::Tensor_attributes()
                                             .set_name("prev_running_mean")
                                             .set_dim({1, c, 1, 1})
                                             .set_stride({c, 1, c, c})
                                             .set_data_type(compute_type));

  auto prev_running_var = graph->tensor(fe::graph::Tensor_attributes()
                                            .set_name("prev_running_var")
                                            .set_dim({1, c, 1, 1})
                                            .set_stride({c, 1, c, c})
                                            .set_data_type(compute_type));

  auto epsilon = graph->tensor(static_cast<float>(stats.epsilon));
  auto momentum = graph->tensor(static_cast<float>(stats.momentum));

  auto bn_options = fe::graph::Batchnorm_attributes().set_epsilon(epsilon);
  bn_options.set_previous_running_stats(prev_running_mean, prev_running_var, momentum);

  auto outputs = graph->batchnorm(X, scale, bias, bn_options);
  auto Y = outputs[0];
  auto saved_mean = outputs[1];
  auto saved_invar = outputs[2];
  auto next_running_mean = outputs[3];
  auto next_running_var = outputs[4];

  Y->set_output(true).set_data_type(io_type);
  saved_mean->set_output(true).set_data_type(compute_type);
  saved_invar->set_output(true).set_data_type(compute_type);
  next_running_mean->set_output(true).set_data_type(compute_type);
  next_running_var->set_output(true).set_data_type(compute_type);

  ensure_ok(graph->validate(), "batchnorm forward validate");
  ensure_ok(graph->build_operation_graph(handle->cudnn_handle), "batchnorm forward build op graph");
  ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A}), "batchnorm forward create plans");
  ensure_ok(graph->check_support(), "batchnorm forward check support");
  ensure_ok(
      graph->build_plans(handle->cudnn_handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
      "batchnorm forward build plans");

  int64_t workspace_size = 0;
  ensure_ok(graph->get_workspace_size(workspace_size), "batchnorm forward workspace");

  handle->fwd_graph = graph;
  handle->fwd_x = X;
  handle->fwd_scale = scale;
  handle->fwd_bias = bias;
  handle->fwd_y = Y;
  handle->fwd_mean = saved_mean;
  handle->fwd_invar = saved_invar;
  handle->fwd_prev_mean = prev_running_mean;
  handle->fwd_prev_var = prev_running_var;
  handle->fwd_next_mean = next_running_mean;
  handle->fwd_next_var = next_running_var;

  stats.fwd_workspace_size = static_cast<size_t>(workspace_size);
}

static void build_inf_graph(feHandle_t *handle, BatchNormStats &stats) {
  const int64_t n = static_cast<int64_t>(stats.batch_size);
  const int64_t c = static_cast<int64_t>(stats.channels);
  const int64_t h = static_cast<int64_t>(stats.height);
  const int64_t w = static_cast<int64_t>(stats.width);
  auto io_type = to_fe_data_type(handle->data_type);
  auto compute_type = to_fe_compute_type(handle->data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type);

  auto X = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("X")
                             .set_dim({n, c, h, w})
                             .set_stride({h * w * c, 1, w * c, c})
                             .set_data_type(io_type));

  auto scale = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("scale")
                                 .set_dim({1, c, 1, 1})
                                 .set_stride({c, 1, c, c})
                                 .set_data_type(compute_type));

  auto bias = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("bias")
                                .set_dim({1, c, 1, 1})
                                .set_stride({c, 1, c, c})
                                .set_data_type(compute_type));

  auto saved_mean = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("saved_mean")
                                      .set_dim({1, c, 1, 1})
                                      .set_stride({c, 1, c, c})
                                      .set_data_type(compute_type));

  auto saved_var = graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("saved_var")
                                     .set_dim({1, c, 1, 1})
                                     .set_stride({c, 1, c, c})
                                     .set_data_type(compute_type));

  // For inference, we apply the normalization manually using the saved stats
  // Y = scale * (X - mean) / sqrt(var + epsilon) + bias

  // Compute inverse standard deviation: 1 / sqrt(var + epsilon)
  auto epsilon_tensor = graph->tensor(static_cast<float>(stats.epsilon));
  auto var_plus_eps =
      graph->pointwise(saved_var, epsilon_tensor,
                       fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
  auto inv_std = graph->pointwise(
      var_plus_eps, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RSQRT));

  // X - mean
  auto x_minus_mean = graph->pointwise(
      X, saved_mean, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::SUB));

  // (X - mean) * inv_std
  auto normalized = graph->pointwise(
      x_minus_mean, inv_std, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL));

  // scale * normalized
  auto scaled = graph->pointwise(
      normalized, scale, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL));

  // scaled + bias
  auto Y = graph->pointwise(scaled, bias,
                            fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));

  Y->set_output(true).set_data_type(io_type);

  ensure_ok(graph->validate(), "batchnorm inference validate");
  ensure_ok(graph->build_operation_graph(handle->cudnn_handle),
            "batchnorm inference build op graph");
  ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A}), "batchnorm inference create plans");
  ensure_ok(graph->check_support(), "batchnorm inference check support");
  ensure_ok(
      graph->build_plans(handle->cudnn_handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
      "batchnorm inference build plans");

  int64_t workspace_size = 0;
  ensure_ok(graph->get_workspace_size(workspace_size), "batchnorm inference workspace");

  handle->inf_graph = graph;
  handle->inf_x = X;
  handle->inf_scale = scale;
  handle->inf_bias = bias;
  handle->inf_saved_mean = saved_mean;
  handle->inf_saved_invar = saved_var;
  handle->inf_y = Y;

  stats.inf_workspace_size = static_cast<size_t>(workspace_size);
}

static void build_bwd_graph(feHandle_t *handle, BatchNormStats &stats) {
  const int64_t n = static_cast<int64_t>(stats.batch_size);
  const int64_t c = static_cast<int64_t>(stats.channels);
  const int64_t h = static_cast<int64_t>(stats.height);
  const int64_t w = static_cast<int64_t>(stats.width);

  auto io_type = to_fe_data_type(handle->data_type);
  auto compute_type = to_fe_compute_type(handle->data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type);

  auto DY = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("DY")
                              .set_dim({n, c, h, w})
                              .set_stride({h * w * c, 1, w * c, c}));

  auto X = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("X")
                             .set_dim({n, c, h, w})
                             .set_stride({h * w * c, 1, w * c, c}));

  auto scale = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("scale")
                                 .set_dim({1, c, 1, 1})
                                 .set_stride({c, 1, c, c})
                                 .set_data_type(fe::DataType_t::FLOAT));

  auto mean = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("mean")
                                .set_dim({1, c, 1, 1})
                                .set_stride({c, 1, c, c})
                                .set_data_type(fe::DataType_t::FLOAT));

  auto invar = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("inv_var")
                                 .set_dim({1, c, 1, 1})
                                 .set_stride({c, 1, c, c})
                                 .set_data_type(fe::DataType_t::FLOAT));

  auto bwd_options =
      fe::graph::Batchnorm_backward_attributes().set_saved_mean_and_inv_variance(mean, invar);

  auto outputs = graph->batchnorm_backward(DY, X, scale, bwd_options);
  auto DX = outputs[0];
  auto dscale = outputs[1];
  auto dbias = outputs[2];

  DX->set_output(true).set_data_type(io_type);
  dscale->set_output(true).set_data_type(fe::DataType_t::FLOAT);
  dbias->set_output(true).set_data_type(fe::DataType_t::FLOAT);

  ensure_ok(graph->validate(), "batchnorm backward validate");
  ensure_ok(graph->build_operation_graph(handle->cudnn_handle),
            "batchnorm backward build op graph");
  ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A}), "batchnorm backward create plans");
  ensure_ok(graph->check_support(), "batchnorm backward check support");
  ensure_ok(
      graph->build_plans(handle->cudnn_handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
      "batchnorm backward build plans");

  int64_t workspace_size = 0;
  ensure_ok(graph->get_workspace_size(workspace_size), "batchnorm backward workspace");

  handle->bwd_graph = graph;
  handle->bwd_dy = DY;
  handle->bwd_x = X;
  handle->bwd_scale = scale;
  handle->bwd_mean = mean;
  handle->bwd_invar = invar;
  handle->bwd_dx = DX;
  handle->bwd_dscale = dscale;
  handle->bwd_dbias = dbias;

  stats.bwd_workspace_size = static_cast<size_t>(workspace_size);
}

static void rebuild_graphs(feHandle_t *handle, BatchNormStats &stats) {
  build_fwd_graph(handle, stats);
  build_inf_graph(handle, stats);
  build_bwd_graph(handle, stats);
}

feHandle_t *initialize_fe_handle(cudnnHandle_t cudnn_handle, cudnnDataType_t data_type,
                                 BatchNormStats &stats) {
  feHandle_t *fe_handle = new feHandle_t();
  fe_handle->cudnn_handle = cudnn_handle;
  fe_handle->data_type = data_type;
  fe_handle->kernel_cache = std::make_shared<fe::KernelCache>();
  rebuild_graphs(fe_handle, stats);
  return fe_handle;
}

void destroy_fe_handle(feHandle_t *handle) {
  if (!handle) {
    return;
  }
  delete handle;
}

void run_forward_training(feHandle_t *handle, const BatchNormStats &stats, const void *input,
                          const void *gamma, const void *beta, void *output,
                          void *prev_running_mean, void *prev_running_var, void *next_running_mean,
                          void *next_running_var, void *batch_mean, void *batch_invar,
                          void *workspace, cudaStream_t stream) {
  if (!handle) {
    throw std::runtime_error("run_forward_training called with null feHandle");
  }

  cudnnSetStream(handle->cudnn_handle, stream);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
      {handle->fwd_x, const_cast<void *>(input)},
      {handle->fwd_scale, const_cast<void *>(gamma)},
      {handle->fwd_bias, const_cast<void *>(beta)},
      {handle->fwd_y, output},
      {handle->fwd_mean, batch_mean},
      {handle->fwd_invar, batch_invar},
      {handle->fwd_prev_mean, prev_running_mean},
      {handle->fwd_prev_var, prev_running_var},
      {handle->fwd_next_mean, next_running_mean},
      {handle->fwd_next_var, next_running_var}};

  auto status = handle->fwd_graph->execute(handle->cudnn_handle, variant_pack, workspace);
  ensure_ok(status, "batchnorm forward execute");
}

void run_forward_inference(feHandle_t *handle, const BatchNormStats &stats, const void *input,
                           const void *gamma, const void *beta, void *saved_mean, void *saved_invar,
                           void *output, void *workspace, cudaStream_t stream) {
  if (!handle) {
    throw std::runtime_error("run_forward_inference called with null feHandle");
  }

  cudnnSetStream(handle->cudnn_handle, stream);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
      {handle->inf_x, const_cast<void *>(input)},   {handle->inf_scale, const_cast<void *>(gamma)},
      {handle->inf_bias, const_cast<void *>(beta)}, {handle->inf_y, output},
      {handle->inf_saved_mean, saved_mean},         {handle->inf_saved_invar, saved_invar}};

  auto status = handle->inf_graph->execute(handle->cudnn_handle, variant_pack, workspace);
  ensure_ok(status, "batchnorm inference execute");
}

void run_backward(feHandle_t *handle, const BatchNormStats &stats, const void *input,
                  const void *grad_output, const void *gamma, void *grad_input, void *grad_gamma,
                  void *grad_beta, const void *batch_mean, const void *batch_invar, void *workspace,
                  cudaStream_t stream) {
  if (!handle) {
    throw std::runtime_error("run_backward called with null feHandle");
  }

  cudnnSetStream(handle->cudnn_handle, stream);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
      {handle->bwd_dy, const_cast<void *>(grad_output)},
      {handle->bwd_x, const_cast<void *>(input)},
      {handle->bwd_scale, const_cast<void *>(gamma)},
      {handle->bwd_mean, const_cast<void *>(batch_mean)},
      {handle->bwd_invar, const_cast<void *>(batch_invar)},
      {handle->bwd_dx, grad_input},
      {handle->bwd_dscale, grad_gamma},
      {handle->bwd_dbias, grad_beta}};

  auto status = handle->bwd_graph->execute(handle->cudnn_handle, variant_pack, workspace);
  ensure_ok(status, "batchnorm backward execute");
}

} // namespace cudnn_batchnorm
} // namespace cuda
} // namespace tnn

#endif
