#include "nn/layers_impl/cuda/cudnn_conv2d_ops.hpp"

#ifdef USE_CUDNN

#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <string>

#include "type/type.hpp"

namespace tnn {
namespace cuda {
namespace cudnn_conv2d {

namespace fe = cudnn_frontend;

struct feHandle_t {
  cudnnHandle_t cudnn_handle = nullptr;
  cudnnDataType_t io_data_type;
  cudnnDataType_t compute_data_type;
  std::shared_ptr<fe::KernelCache> kernel_cache;

  std::shared_ptr<fe::graph::Graph> fwd_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_x;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_w;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_b;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_y;

  std::shared_ptr<fe::graph::Graph> dgrad_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> dgrad_dy;
  std::shared_ptr<fe::graph::Tensor_attributes> dgrad_w;
  std::shared_ptr<fe::graph::Tensor_attributes> dgrad_dx;

  std::shared_ptr<fe::graph::Graph> wgrad_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> wgrad_x;
  std::shared_ptr<fe::graph::Tensor_attributes> wgrad_dy;
  std::shared_ptr<fe::graph::Tensor_attributes> wgrad_dw;

  std::shared_ptr<fe::graph::Graph> bgrad_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> bgrad_dy;
  std::shared_ptr<fe::graph::Tensor_attributes> bgrad_db;
};

static void ensure_ok(fe::error_t status, std::string stage) {
  if (status.is_bad()) {
    throw std::runtime_error("cuDNN frontend error at " + stage + ": " + status.get_message());
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
      throw std::runtime_error("Unsupported cuDNN data type for conv2d");
  }
}

static fe::DataType_t to_fe_compute_type(cudnnDataType_t data_type) {
  if (data_type == CUDNN_DATA_HALF || data_type == CUDNN_DATA_BFLOAT16) {
    return fe::DataType_t::FLOAT;
  }
  return to_fe_data_type(data_type);
}

static void build_fwd_graph(feHandle_t* handle, ConvolutionStats& stats) {
  const int64_t n = static_cast<int64_t>(stats.batch_size);
  const int64_t c = static_cast<int64_t>(stats.in_channels);
  const int64_t h = static_cast<int64_t>(stats.input_h);
  const int64_t w = static_cast<int64_t>(stats.input_w);
  const int64_t k = static_cast<int64_t>(stats.out_channels);
  const int64_t r = static_cast<int64_t>(stats.kernel_h);
  const int64_t s = static_cast<int64_t>(stats.kernel_w);

  auto io_type = to_fe_data_type(handle->io_data_type);
  auto compute_type = to_fe_compute_type(handle->compute_data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type)
      .set_dynamic_shape_enabled(true)
      .set_kernel_cache(handle->kernel_cache);

  auto X = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("X")
                             .set_dim({n, c, h, w})
                             .set_stride({h * w * c, 1, w * c, c}));

  auto W = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("W")
                             .set_dim({k, c, r, s})
                             .set_stride({r * s * c, 1, s * c, c}));

  auto conv_options =
      fe::graph::Conv_fprop_attributes()
          .set_pre_padding({static_cast<int64_t>(stats.pad_h), static_cast<int64_t>(stats.pad_w)})
          .set_post_padding({static_cast<int64_t>(stats.pad_h), static_cast<int64_t>(stats.pad_w)})
          .set_stride({static_cast<int64_t>(stats.stride_h), static_cast<int64_t>(stats.stride_w)})
          .set_dilation({1, 1});

  auto conv_output = graph->conv_fprop(X, W, conv_options);

  std::shared_ptr<fe::graph::Tensor_attributes> Y;
  std::shared_ptr<fe::graph::Tensor_attributes> B;

  if (stats.use_bias) {
    B = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("B")
                          .set_dim({1, k, 1, 1})
                          .set_stride({k, 1, k, k}));

    auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    Y = graph->pointwise(conv_output, B, bias_options);
    Y->set_output(true).set_data_type(io_type);
  } else {
    Y = conv_output;
    Y->set_output(true).set_data_type(io_type);
  }

  ensure_ok(graph->validate(), "conv_fprop validate");
  ensure_ok(graph->build_operation_graph(handle->cudnn_handle), "conv_fprop build op graph");
  ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A}), "conv_fprop create plans");
  ensure_ok(graph->check_support(), "conv_fprop check support");
  ensure_ok(graph->build_plans(), "conv_fprop build plans");

  int64_t workspace_size = 0;
  ensure_ok(graph->get_workspace_size(workspace_size), "conv_fprop workspace");

  handle->fwd_graph = graph;
  handle->fwd_x = X;
  handle->fwd_w = W;
  handle->fwd_b = B;
  handle->fwd_y = Y;
  stats.fwd_workspace_size = static_cast<size_t>(workspace_size);
}

static void build_dgrad_graph(feHandle_t* handle, ConvolutionStats& stats) {
  const int64_t n = static_cast<int64_t>(stats.batch_size);
  const int64_t c = static_cast<int64_t>(stats.in_channels);
  const int64_t h = static_cast<int64_t>(stats.input_h);
  const int64_t w = static_cast<int64_t>(stats.input_w);
  const int64_t k = static_cast<int64_t>(stats.out_channels);
  const int64_t r = static_cast<int64_t>(stats.kernel_h);
  const int64_t s = static_cast<int64_t>(stats.kernel_w);
  const int64_t p = static_cast<int64_t>(stats.output_h);
  const int64_t q = static_cast<int64_t>(stats.output_w);

  auto io_type = to_fe_data_type(handle->io_data_type);
  auto compute_type = to_fe_compute_type(handle->compute_data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type);

  auto DY = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("DY")
                              .set_dim({n, k, p, q})
                              .set_stride({p * q * k, 1, q * k, k}));

  auto W = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("W")
                             .set_dim({k, c, r, s})
                             .set_stride({r * s * c, 1, s * c, c}));

  auto dgrad_options =
      fe::graph::Conv_dgrad_attributes()
          .set_pre_padding({static_cast<int64_t>(stats.pad_h), static_cast<int64_t>(stats.pad_w)})
          .set_post_padding({static_cast<int64_t>(stats.pad_h), static_cast<int64_t>(stats.pad_w)})
          .set_stride({static_cast<int64_t>(stats.stride_h), static_cast<int64_t>(stats.stride_w)})
          .set_dilation({1, 1});

  auto DX = graph->conv_dgrad(DY, W, dgrad_options);
  DX->set_dim({n, c, h, w})
      .set_stride({h * w * c, 1, w * c, c})
      .set_data_type(io_type)
      .set_output(true);

  ensure_ok(graph->validate(), "conv_dgrad validate");
  ensure_ok(graph->build_operation_graph(handle->cudnn_handle), "conv_dgrad build op graph");
  ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A}), "conv_dgrad create plans");
  ensure_ok(graph->check_support(), "conv_dgrad check support");
  ensure_ok(graph->build_plans(), "conv_dgrad build plans");

  int64_t workspace_size = 0;
  ensure_ok(graph->get_workspace_size(workspace_size), "conv_dgrad workspace");

  handle->dgrad_graph = graph;
  handle->dgrad_dy = DY;
  handle->dgrad_w = W;
  handle->dgrad_dx = DX;
  stats.dgrad_workspace_size = static_cast<size_t>(workspace_size);
}

static void build_wgrad_graph(feHandle_t* handle, ConvolutionStats& stats) {
  const int64_t n = static_cast<int64_t>(stats.batch_size);
  const int64_t c = static_cast<int64_t>(stats.in_channels);
  const int64_t h = static_cast<int64_t>(stats.input_h);
  const int64_t w = static_cast<int64_t>(stats.input_w);
  const int64_t k = static_cast<int64_t>(stats.out_channels);
  const int64_t r = static_cast<int64_t>(stats.kernel_h);
  const int64_t s = static_cast<int64_t>(stats.kernel_w);
  const int64_t p = static_cast<int64_t>(stats.output_h);
  const int64_t q = static_cast<int64_t>(stats.output_w);

  auto io_type = to_fe_data_type(handle->io_data_type);
  auto compute_type = to_fe_compute_type(handle->compute_data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type);

  auto X = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("X")
                             .set_dim({n, c, h, w})
                             .set_stride({h * w * c, 1, w * c, c}));

  auto DY = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("DY")
                              .set_dim({n, k, p, q})
                              .set_stride({p * q * k, 1, q * k, k}));

  auto wgrad_options =
      fe::graph::Conv_wgrad_attributes()
          .set_pre_padding({static_cast<int64_t>(stats.pad_h), static_cast<int64_t>(stats.pad_w)})
          .set_post_padding({static_cast<int64_t>(stats.pad_h), static_cast<int64_t>(stats.pad_w)})
          .set_stride({static_cast<int64_t>(stats.stride_h), static_cast<int64_t>(stats.stride_w)})
          .set_dilation({1, 1});

  auto DW = graph->conv_wgrad(DY, X, wgrad_options);
  DW->set_output(true)
      .set_dim({k, c, r, s})
      .set_stride({r * s * c, 1, s * c, c})
      .set_data_type(io_type);

  ensure_ok(graph->validate(), "conv_wgrad validate");
  ensure_ok(graph->build_operation_graph(handle->cudnn_handle), "conv_wgrad build op graph");
  ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A}), "conv_wgrad create plans");
  ensure_ok(graph->check_support(), "conv_wgrad check support");
  ensure_ok(graph->build_plans(), "conv_wgrad build plans");

  int64_t workspace_size = 0;
  ensure_ok(graph->get_workspace_size(workspace_size), "conv_wgrad workspace");

  handle->wgrad_graph = graph;
  handle->wgrad_x = X;
  handle->wgrad_dy = DY;
  handle->wgrad_dw = DW;
  stats.wgrad_workspace_size = static_cast<size_t>(workspace_size);
}

static void build_bgrad_graph(feHandle_t* handle, ConvolutionStats& stats) {
  const int64_t n = static_cast<int64_t>(stats.batch_size);
  const int64_t k = static_cast<int64_t>(stats.out_channels);
  const int64_t p = static_cast<int64_t>(stats.output_h);
  const int64_t q = static_cast<int64_t>(stats.output_w);

  auto io_type = to_fe_data_type(handle->io_data_type);
  auto compute_type = to_fe_compute_type(handle->compute_data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type);

  auto DY = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("DY")
                              .set_dim({n, k, p, q})
                              .set_stride({p * q * k, 1, q * k, k}));

  // Bias grad_output is the sum of DY over N, H, and W dimensions
  auto reduction_options = fe::graph::Reduction_attributes().set_mode(fe::ReductionMode_t::ADD);

  auto DB = graph->reduction(DY, reduction_options);
  DB->set_output(true)
      .set_dim({1, k, 1, 1})
      .set_stride({k, 1, k, k})
      .set_data_type(io_type)
      .set_name("DB");

  ensure_ok(graph->validate(), "bias_grad validate");
  ensure_ok(graph->build_operation_graph(handle->cudnn_handle), "bias_grad build op graph");
  ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A}), "bias_grad create plans");
  ensure_ok(graph->check_support(), "bias_grad check support");
  ensure_ok(graph->build_plans(), "bias_grad build plans");

  int64_t workspace_size = 0;
  ensure_ok(graph->get_workspace_size(workspace_size), "bias_grad workspace");

  handle->bgrad_graph = graph;
  handle->bgrad_dy = DY;
  handle->bgrad_db = DB;
  stats.bgrad_workspace_size = static_cast<size_t>(workspace_size);
}

static void rebuild_all_graphs(feHandle_t* handle, ConvolutionStats& stats) {
  build_fwd_graph(handle, stats);
  build_dgrad_graph(handle, stats);
  build_wgrad_graph(handle, stats);
  if (stats.use_bias) {
    build_bgrad_graph(handle, stats);
  }
}

feHandle_t* initialize_fe_handle(cudnnHandle_t cudnn_handle, cudnnDataType_t io_data_type,
                                 cudnnDataType_t compute_data_type, ConvolutionStats& stats) {
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

void run_forward(feHandle_t* handle, const ConvolutionStats& stats, const void* input_data,
                 const void* weight_data, const void* bias_data, void* output_data,
                 void* workspace_data, cudaStream_t stream) {
  if (!handle) {
    throw std::runtime_error("run_forward called with null feHandle_t");
  }

  cudnnSetStream(handle->cudnn_handle, stream);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {handle->fwd_x, const_cast<void*>(input_data)},
      {handle->fwd_w, const_cast<void*>(weight_data)},
      {handle->fwd_y, output_data}};

  if (stats.use_bias && bias_data && handle->fwd_b) {
    variant_pack[handle->fwd_b] = const_cast<void*>(bias_data);
  }

  auto status = handle->fwd_graph->execute(handle->cudnn_handle, variant_pack, workspace_data);
  ensure_ok(status, "conv_fprop execute");
}

void run_backward_data(feHandle_t* handle, const ConvolutionStats& stats, const void* gradient_data,
                       const void* weight_data, void* input_grad_data, void* workspace_data,
                       cudaStream_t stream) {
  if (!handle) {
    throw std::runtime_error("run_backward_data called with null feHandle_t");
  }

  cudnnSetStream(handle->cudnn_handle, stream);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {handle->dgrad_dy, const_cast<void*>(gradient_data)},
      {handle->dgrad_w, const_cast<void*>(weight_data)},
      {handle->dgrad_dx, input_grad_data}};

  auto status = handle->dgrad_graph->execute(handle->cudnn_handle, variant_pack, workspace_data);
  ensure_ok(status, "conv_dgrad execute");
}

void run_backward_weights_and_bias(feHandle_t* handle, const ConvolutionStats& stats,
                                   const void* input_data, const void* gradient_data,
                                   void* weight_grad_data, void* bias_grad_data,
                                   void* workspace_data, cudaStream_t stream) {
  if (!handle) {
    throw std::runtime_error("run_backward_weights_and_bias called with null feHandle_t");
  }

  cudnnSetStream(handle->cudnn_handle, stream);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {handle->wgrad_x, const_cast<void*>(input_data)},
      {handle->wgrad_dy, const_cast<void*>(gradient_data)},
      {handle->wgrad_dw, weight_grad_data}};

  auto status = handle->wgrad_graph->execute(handle->cudnn_handle, variant_pack, workspace_data);
  ensure_ok(status, "conv_wgrad execute");

  // Compute bias grad_output separately if needed
  if (stats.use_bias && bias_grad_data && handle->bgrad_graph) {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> bgrad_variant_pack = {
        {handle->bgrad_dy, const_cast<void*>(gradient_data)}, {handle->bgrad_db, bias_grad_data}};

    auto bgrad_status =
        handle->bgrad_graph->execute(handle->cudnn_handle, bgrad_variant_pack, workspace_data);
    ensure_ok(bgrad_status, "bias_grad execute");
  }
}

}  // namespace cudnn_conv2d
}  // namespace cuda
}  // namespace tnn

#endif
