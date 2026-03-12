#include "math/cuda/cudnn_gemm.hpp"

#ifdef USE_CUDNN

#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "type/type.hpp"

namespace tnn {
namespace cuda {
namespace cudnn_gemm {

namespace fe = cudnn_frontend;

struct feHandle_t {
  cudnnHandle_t cudnn_handle = nullptr;
  cudnnDataType_t io_data_type;
  cudnnDataType_t param_data_type;
  cudnnDataType_t compute_data_type;
  std::shared_ptr<fe::KernelCache> kernel_cache;

  std::shared_ptr<fe::graph::Graph> fwd_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_input;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_weight;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_output;

  std::shared_ptr<fe::graph::Graph> dgrad_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> dgrad_gradient;
  std::shared_ptr<fe::graph::Tensor_attributes> dgrad_weight;
  std::shared_ptr<fe::graph::Tensor_attributes> dgrad_grad_input;

  std::shared_ptr<fe::graph::Graph> wgrad_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> wgrad_gradient;
  std::shared_ptr<fe::graph::Tensor_attributes> wgrad_input;
  std::shared_ptr<fe::graph::Tensor_attributes> wgrad_prev_grad_weight;
  std::shared_ptr<fe::graph::Tensor_attributes> wgrad_grad_weight;
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
    case CUDNN_DATA_INT8:
      return fe::DataType_t::INT8;
    default:
      throw std::runtime_error("Unsupported cuDNN data type for GEMM");
  }
}

static void build_fwd_graph(feHandle_t* handle, GemmStats& stats) {
  const int64_t batch = static_cast<int64_t>(stats.batch_count);
  const int64_t m = static_cast<int64_t>(stats.M);
  const int64_t n = static_cast<int64_t>(stats.N);
  const int64_t k = static_cast<int64_t>(stats.K);

  auto io_type = to_fe_data_type(handle->io_data_type);
  auto param_type = to_fe_data_type(handle->param_data_type);
  auto compute_type = to_fe_data_type(handle->compute_data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type);

  auto input = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("Input")
                                 .set_dim({batch, m, k})
                                 .set_stride({m * k, k, 1})
                                 .set_data_type(io_type));

  auto weight = graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Weight")
                                  .set_dim({batch, k, n})
                                  .set_stride({0, 1, k})
                                  .set_data_type(param_type));

  std::shared_ptr<fe::graph::Tensor_attributes> weight_cast = weight;

  {
    auto identity_attributes = fe::graph::Pointwise_attributes()
                                   .set_name("Cast_Weight")
                                   .set_mode(fe::PointwiseMode_t::IDENTITY)
                                   .set_compute_data_type(compute_type);
    weight_cast = graph->pointwise(weight, identity_attributes);
    weight_cast->set_data_type(io_type);
  }

  auto matmul_attributes =
      fe::graph::Matmul_attributes().set_name("FWD_GEMM").set_compute_data_type(compute_type);

  auto output = graph->matmul(input, weight_cast, matmul_attributes);

  output->set_output(true).set_data_type(io_type);

  ensure_ok(graph->validate(), "fwd_gemm validate");
  ensure_ok(graph->build_operation_graph(handle->cudnn_handle), "fwd_gemm build op graph");
  ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A}), "fwd_gemm create plans");
  ensure_ok(graph->check_support(), "fwd_gemm check support");
  ensure_ok(graph->build_plans(), "fwd_gemm build plans");

  int64_t workspace_size = 0;
  ensure_ok(graph->get_workspace_size(workspace_size), "fwd_gemm workspace");

  handle->fwd_graph = graph;
  handle->fwd_input = input;
  handle->fwd_weight = weight;
  handle->fwd_output = output;
  stats.fwd_workspace_size = static_cast<size_t>(workspace_size);
}

static void build_dgrad_graph(feHandle_t* handle, GemmStats& stats) {
  const int64_t batch = static_cast<int64_t>(stats.batch_count);
  const int64_t m = static_cast<int64_t>(stats.M);
  const int64_t n = static_cast<int64_t>(stats.N);
  const int64_t k = static_cast<int64_t>(stats.K);

  auto io_type = to_fe_data_type(handle->io_data_type);
  auto param_type = to_fe_data_type(handle->param_data_type);
  auto compute_type = to_fe_data_type(handle->compute_data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type);

  auto grad_output = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("Gradient")
                                       .set_dim({batch, m, n})
                                       .set_stride({m * n, n, 1})
                                       .set_data_type(io_type));

  auto weight = graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Weight")
                                  .set_dim({batch, n, k})
                                  .set_stride({0, k, 1})
                                  .set_data_type(param_type));

  std::shared_ptr<fe::graph::Tensor_attributes> weight_cast = weight;

  {
    auto identity_attributes = fe::graph::Pointwise_attributes()
                                   .set_name("Cast_Weight")
                                   .set_mode(fe::PointwiseMode_t::IDENTITY)
                                   .set_compute_data_type(compute_type);
    weight_cast = graph->pointwise(weight, identity_attributes);
    weight_cast->set_data_type(io_type);
  }

  auto matmul_attributes =
      fe::graph::Matmul_attributes().set_name("DGRAD_GEMM").set_compute_data_type(compute_type);
  auto grad_input = graph->matmul(grad_output, weight_cast, matmul_attributes);
  grad_input->set_output(true).set_data_type(io_type);

  ensure_ok(graph->validate(), "dgrad_gemm validate");
  ensure_ok(graph->build_operation_graph(handle->cudnn_handle), "dgrad_gemm build op graph");
  ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A}), "dgrad_gemm create plans");
  ensure_ok(graph->check_support(), "dgrad_gemm check support");
  ensure_ok(graph->build_plans(), "dgrad_gemm build plans");

  int64_t workspace_size = 0;
  ensure_ok(graph->get_workspace_size(workspace_size), "dgrad_gemm workspace");

  handle->dgrad_graph = graph;
  handle->dgrad_gradient = grad_output;
  handle->dgrad_weight = weight;
  handle->dgrad_grad_input = grad_input;
  stats.dgrad_workspace_size = static_cast<size_t>(workspace_size);
}

static void build_wgrad_graph(feHandle_t* handle, GemmStats& stats) {
  const int64_t batch = static_cast<int64_t>(stats.batch_count);
  const int64_t m = static_cast<int64_t>(stats.M);
  const int64_t n = static_cast<int64_t>(stats.N);
  const int64_t k = static_cast<int64_t>(stats.K);

  auto io_type = to_fe_data_type(handle->io_data_type);
  auto param_type = to_fe_data_type(handle->param_data_type);
  auto compute_type = to_fe_data_type(handle->compute_data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type)
      .set_dynamic_shape_enabled(true)
      .set_kernel_cache(handle->kernel_cache);

  auto grad_output = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("Gradient")
                                       .set_dim({batch, n, m})
                                       .set_stride({m * n, 1, n})
                                       .set_data_type(io_type));

  auto input = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("Input")
                                 .set_dim({batch, m, k})
                                 .set_stride({m * k, k, 1})
                                 .set_data_type(io_type));

  auto prev_grad_weight = graph->tensor(fe::graph::Tensor_attributes()
                                            .set_name("PrevGradWeight")
                                            .set_dim({1, n, k})
                                            .set_stride({0, k, 1})
                                            .set_data_type(param_type));

  auto matmul_attributes =
      fe::graph::Matmul_attributes().set_name("WGRAD_GEMM").set_compute_data_type(compute_type);
  auto new_grad_weight = graph->matmul(grad_output, input, matmul_attributes);
  new_grad_weight->set_data_type(param_type);

  // Accumulate with previous grad_output (beta=1.0 behavior)
  auto add_attributes = fe::graph::Pointwise_attributes()
                            .set_name("AccumulateGrad")
                            .set_mode(fe::PointwiseMode_t::ADD)
                            .set_compute_data_type(compute_type);
  auto grad_weight = graph->pointwise(new_grad_weight, prev_grad_weight, add_attributes);
  grad_weight->set_output(true).set_data_type(param_type);

  ensure_ok(graph->validate(), "wgrad_gemm validate");
  ensure_ok(graph->build_operation_graph(handle->cudnn_handle), "wgrad_gemm build op graph");
  ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A}), "wgrad_gemm create plans");
  ensure_ok(graph->check_support(), "wgrad_gemm check support");
  ensure_ok(graph->build_plans(), "wgrad_gemm build plans");

  int64_t workspace_size = 0;
  ensure_ok(graph->get_workspace_size(workspace_size), "wgrad_gemm workspace");

  handle->wgrad_graph = graph;
  handle->wgrad_gradient = grad_output;
  handle->wgrad_input = input;
  handle->wgrad_prev_grad_weight = prev_grad_weight;
  handle->wgrad_grad_weight = grad_weight;
  stats.wgrad_workspace_size = static_cast<size_t>(workspace_size);
}

static void rebuild_all_graphs(feHandle_t* handle, GemmStats& stats) {
  build_fwd_graph(handle, stats);
  build_dgrad_graph(handle, stats);
  build_wgrad_graph(handle, stats);
}

feHandle_t* initialize_fe_handle(cudnnHandle_t cudnn_handle, cudnnDataType_t io_type,
                                 cudnnDataType_t param_type, cudnnDataType_t compute_type,
                                 GemmStats& stats) {
  auto* handle = new feHandle_t();
  handle->cudnn_handle = cudnn_handle;
  handle->io_data_type = io_type;
  handle->param_data_type = param_type;
  handle->compute_data_type = compute_type;
  handle->kernel_cache = std::make_shared<fe::KernelCache>();

  rebuild_all_graphs(handle, stats);

  return handle;
}

void destroy_fe_handle(feHandle_t* handle) {
  if (!handle) {
    return;
  }
  delete handle;
}

void run_forward(feHandle_t* handle, const GemmStats& stats, const void* input_data,
                 const void* weight_data, void* output_data, void* workspace_data,
                 cudaStream_t stream) {
  if (!handle) {
    throw std::runtime_error("Invalid feHandle_t in run_forward");
  }

  cudnnSetStream(handle->cudnn_handle, stream);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {handle->fwd_input, const_cast<void*>(input_data)},
      {handle->fwd_weight, const_cast<void*>(weight_data)},
      {handle->fwd_output, output_data}};

  auto status = handle->fwd_graph->execute(handle->cudnn_handle, variant_pack, workspace_data);
  ensure_ok(status, "fwd_gemm execute");
}

void run_dgrad(feHandle_t* handle, const GemmStats& stats, const void* gradient_data,
               const void* weight_data, void* grad_input_data, void* workspace_data,
               cudaStream_t stream) {
  if (!handle) {
    throw std::runtime_error("Invalid feHandle_t in run_dgrad");
  }

  cudnnSetStream(handle->cudnn_handle, stream);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {handle->dgrad_gradient, const_cast<void*>(gradient_data)},
      {handle->dgrad_weight, const_cast<void*>(weight_data)},
      {handle->dgrad_grad_input, grad_input_data}};

  auto status = handle->dgrad_graph->execute(handle->cudnn_handle, variant_pack, workspace_data);
  ensure_ok(status, "dgrad_gemm execute");
}

void run_wgrad(feHandle_t* handle, const GemmStats& stats, const void* input_data,
               const void* gradient_data, void* weight_grad_data, void* workspace_data,
               cudaStream_t stream) {
  if (!handle) {
    throw std::runtime_error("Invalid feHandle_t in run_wgrad");
  }

  cudnnSetStream(handle->cudnn_handle, stream);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {handle->wgrad_gradient, const_cast<void*>(gradient_data)},
      {handle->wgrad_input, const_cast<void*>(input_data)},
      {handle->wgrad_prev_grad_weight, weight_grad_data},
      {handle->wgrad_grad_weight, weight_grad_data}};

  auto status = handle->wgrad_graph->execute(handle->cudnn_handle, variant_pack, workspace_data);
  ensure_ok(status, "wgrad_gemm execute");
}

}  // namespace cudnn_gemm
}  // namespace cuda
}  // namespace tnn

#endif
