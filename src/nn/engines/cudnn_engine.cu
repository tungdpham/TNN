#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>
#include <cudnn_graph.h>
#include <fmt/core.h>

#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "cuda/helpers.cuh"
#include "cudnn_frontend/graph_interface.h"
#include "cudnn_frontend/graph_properties.h"
#include "cudnn_frontend_utils.h"
#include "math/cuda/axpy.hpp"
#include "nn/engines/cudnn_engine.hpp"
#include "nn/stats/stats.hpp"
#include "type/type.hpp"

namespace tunx {

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T>
__global__ void bgrad_reduce_accumulate_kernel(const T* __restrict__ dy, T* __restrict__ db,
                                               int batch_size, int out_features) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane_id = threadIdx.x % 32;

  if (warp_id >= out_features) return;

  float sum = 0.0f;
  for (int b = lane_id; b < batch_size; b += 32) {
    int idx = b * out_features + warp_id;
    sum += (float)dy[idx];
  }

  sum = warp_reduce_sum(sum);

  if (lane_id == 0) {
    db[warp_id] = (T)(sum + (float)db[warp_id]);
  }
}

namespace fe = cudnn_frontend;

static void ensure_ok(fe::error_t status, std::string stage) {
  if (status.is_bad()) {
    throw std::runtime_error("cuDNN frontend error at " + stage + ": " + status.get_message());
  }
}

[[maybe_unused]] static std::string to_string(fe::DataType_t data_type) {
  switch (data_type) {
    case fe::DataType_t::HALF:
      return "HALF";
    case fe::DataType_t::FLOAT:
      return "FLOAT";
    case fe::DataType_t::DOUBLE:
      return "DOUBLE";
    case fe::DataType_t::BFLOAT16:
      return "BFLOAT16";
    case fe::DataType_t::INT8:
      return "INT8";
    case fe::DataType_t::INT32:
      return "INT32";
    case fe::DataType_t::INT64:
      return "INT64";
    case fe::DataType_t::UINT8:
      return "UINT8";
    default:
      return "UNKNOWN";
  }
}

static fe::DataType_t to_fe_data_type(DType_t data_type) {
  switch (data_type) {
    case DType_t::FP16:
      return fe::DataType_t::HALF;
    case DType_t::FP32:
      return fe::DataType_t::FLOAT;
    case DType_t::FP64:
      return fe::DataType_t::DOUBLE;
    case DType_t::BF16:
      return fe::DataType_t::BFLOAT16;
    case DType_t::INT8:
      return fe::DataType_t::INT8;
    default:
      throw std::runtime_error("Unsupported cuDNN data type for GEMM");
  }
}

static fe::DataType_t to_fe_compute_type(DType_t data_type) {
  if (data_type == DType_t::FP16 || data_type == DType_t::BF16) {
    return fe::DataType_t::FLOAT;
  }
  return to_fe_data_type(data_type);
}

void* CuDNNEngine::create_backend_handle() {
  cudnnHandle_t handle;
  cudnnCreate(&handle);
  return handle;
}

struct dense_fwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> w;
  std::shared_ptr<fe::graph::Tensor_attributes> b;
  std::shared_ptr<fe::graph::Tensor_attributes> y;

  size_t workspace_size;

  dense_fwd_graph(cudnnHandle_t handle, const DenseStats& stats, DTypeDesc& type_desc) {
    const int64 batch = static_cast<int64>(1);
    const int64 m = static_cast<int64>(stats.batch_size);
    const int64 n = static_cast<int64>(stats.out_features);
    const int64 k = static_cast<int64>(stats.in_features);

    fe::DataType_t io_type = to_fe_data_type(type_desc.io_dtype);
    fe::DataType_t param_type = to_fe_data_type(type_desc.param_dtype);
    fe::DataType_t compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Input")
                          .set_dim({batch, m, k})
                          .set_stride({m * k, k, 1})
                          .set_data_type(io_type));

    w = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Weight")
                          .set_dim({batch, k, n})
                          .set_stride({0, 1, k})
                          .set_data_type(param_type));

    b = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Bias")
                          .set_dim({batch, 1, n})
                          .set_stride({0, 0, 1})
                          .set_data_type(param_type));

    auto weight_cast = w;

    {
      auto identity_attributes = fe::graph::Pointwise_attributes()
                                     .set_name("Cast_Weight")
                                     .set_mode(fe::PointwiseMode_t::IDENTITY)
                                     .set_compute_data_type(compute_type);
      weight_cast = graph->pointwise(w, identity_attributes);
      weight_cast->set_data_type(io_type);
    }

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("FWD_GEMM").set_compute_data_type(compute_type);
    y = graph->matmul(x, weight_cast, matmul_attributes);
    if (stats.use_bias) {
      y->set_is_virtual(true);

      auto add_bias_attributes = fe::graph::Pointwise_attributes()
                                     .set_name("Add_Bias")
                                     .set_mode(fe::PointwiseMode_t::ADD)
                                     .set_compute_data_type(compute_type);

      y = graph->pointwise(y, b, add_bias_attributes);
    }

    y->set_output(true).set_data_type(io_type);

    ensure_ok(graph->validate(), "fwd_gemm validate");
    ensure_ok(graph->build_operation_graph(handle), "fwd_gemm build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "fwd_gemm create plans");
    ensure_ok(graph->check_support(), "fwd_gemm check support");
    ensure_ok(graph->build_plans(), "fwd_gemm build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "fwd_gemm workspace");
    assert(ws >= 0);
    workspace_size = ws;
  }
};

struct dense_dgrad_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> dy;
  std::shared_ptr<fe::graph::Tensor_attributes> w;
  std::shared_ptr<fe::graph::Tensor_attributes> dx;

  size_t workspace_size;

  dense_dgrad_graph(cudnnHandle_t handle, const DenseStats& stats, DTypeDesc& type_desc) {
    const int64 batch = static_cast<int64>(1);
    const int64 m = static_cast<int64>(stats.batch_size);
    const int64 n = static_cast<int64>(stats.out_features);
    const int64 k = static_cast<int64>(stats.in_features);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto param_type = to_fe_data_type(type_desc.param_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    dy = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("Gradient")
                           .set_dim({batch, m, n})
                           .set_stride({m * n, n, 1})
                           .set_data_type(io_type));

    w = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Weight")
                          .set_dim({batch, n, k})
                          .set_stride({0, k, 1})
                          .set_data_type(param_type));

    auto weight_cast = w;

    {
      auto identity_attributes = fe::graph::Pointwise_attributes()
                                     .set_name("Cast_Weight")
                                     .set_mode(fe::PointwiseMode_t::IDENTITY)
                                     .set_compute_data_type(compute_type);
      weight_cast = graph->pointwise(w, identity_attributes);
      weight_cast->set_data_type(io_type);
    }

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("DGRAD_GEMM").set_compute_data_type(compute_type);
    dx = graph->matmul(dy, weight_cast, matmul_attributes);
    dx->set_output(true).set_data_type(io_type);

    ensure_ok(graph->validate(), "dgrad_gemm validate");
    ensure_ok(graph->build_operation_graph(handle), "dgrad_gemm build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "dgrad_gemm create plans");
    ensure_ok(graph->check_support(), "dgrad_gemm check support");
    ensure_ok(graph->build_plans(), "dgrad_gemm build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "dgrad_gemm workspace");
    assert(ws >= 0);
    workspace_size = ws;
  }
};

struct dense_wgrad_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> dy;
  std::shared_ptr<fe::graph::Tensor_attributes> dw;
  std::shared_ptr<fe::graph::Tensor_attributes> dw_temp;

  size_t workspace_size;

  dense_wgrad_graph(cudnnHandle_t handle, const DenseStats& stats, DTypeDesc& type_desc) {
    const int64 batch = static_cast<int64>(1);
    const int64 m = static_cast<int64>(stats.batch_size);
    const int64 n = static_cast<int64>(stats.out_features);
    const int64 k = static_cast<int64>(stats.in_features);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto param_type = to_fe_data_type(type_desc.param_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    dy = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("Gradient")
                           .set_dim({batch, n, m})
                           .set_stride({m * n, 1, n})
                           .set_data_type(io_type));

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Input")
                          .set_dim({batch, m, k})
                          .set_stride({m * k, k, 1})
                          .set_data_type(io_type));

    dw = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("dw")
                           .set_dim({batch, n, k})
                           .set_stride({n * k, k, 1})
                           .set_data_type(param_type));

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("WGRAD_GEMM").set_compute_data_type(compute_type);
    auto matmul_output = graph->matmul(dy, x, matmul_attributes);

    auto add_attributes = fe::graph::Pointwise_attributes()
                              .set_name("DW_accumulate")
                              .set_mode(fe::PointwiseMode_t::ADD);

    dw_temp = graph->pointwise(dw, matmul_output, add_attributes);
    dw_temp->set_output(true).set_data_type(param_type);

    ensure_ok(graph->validate(), "wgrad_gemm validate");
    ensure_ok(graph->build_operation_graph(handle), "wgrad_gemm build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "wgrad_gemm create plans");
    ensure_ok(graph->check_support(), "wgrad_gemm check support");
    ensure_ok(graph->build_plans(), "wgrad_gemm build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "wgrad_gemm workspace");

    workspace_size = static_cast<size_t>(ws);
  }
};

struct conv2d_fwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> w;
  std::shared_ptr<fe::graph::Tensor_attributes> b;
  std::shared_ptr<fe::graph::Tensor_attributes> y;

  size_t workspace_size;

  conv2d_fwd_graph(cudnnHandle_t handle, const Conv2DStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.in_channels);
    const int64 h = static_cast<int64>(stats.input_h);
    const int64 w_dim = static_cast<int64>(stats.input_w);
    const int64 k = static_cast<int64>(stats.out_channels);
    const int64 r = static_cast<int64>(stats.kernel_h);
    const int64 s = static_cast<int64>(stats.kernel_w);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim({n, c, h, w_dim})
                          .set_stride({h * w_dim * c, 1, w_dim * c, c}));

    w = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("W")
                          .set_dim({k, c, r, s})
                          .set_stride({r * s * c, 1, s * c, c}));

    auto conv_options =
        fe::graph::Conv_fprop_attributes()
            .set_pre_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_post_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_stride({static_cast<int64>(stats.stride_h), static_cast<int64>(stats.stride_w)})
            .set_dilation({1, 1});

    auto conv_output = graph->conv_fprop(x, w, conv_options);

    if (stats.use_bias) {
      b = graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({1, k, 1, 1})
                            .set_stride({k, 1, k, k}));

      auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
      y = graph->pointwise(conv_output, b, bias_options);
      y->set_output(true).set_data_type(io_type);
    } else {
      y = conv_output;
      y->set_output(true).set_data_type(io_type);
    }

    ensure_ok(graph->validate(), "conv_fprop validate");
    ensure_ok(graph->build_operation_graph(handle), "conv_fprop build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "conv_fprop create plans");
    ensure_ok(graph->check_support(), "conv_fprop check support");
    ensure_ok(graph->build_plans(), "conv_fprop build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "conv_fprop workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct conv2d_dgrad_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> dy;
  std::shared_ptr<fe::graph::Tensor_attributes> w;
  std::shared_ptr<fe::graph::Tensor_attributes> dx;

  size_t workspace_size;

  conv2d_dgrad_graph(cudnnHandle_t handle, const Conv2DStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.in_channels);
    const int64 h = static_cast<int64>(stats.input_h);
    const int64 w_dim = static_cast<int64>(stats.input_w);
    const int64 k = static_cast<int64>(stats.out_channels);
    const int64 r = static_cast<int64>(stats.kernel_h);
    const int64 s = static_cast<int64>(stats.kernel_w);
    const int64 output_h = (stats.input_h + stats.pad_h * 2 - stats.kernel_h) / stats.stride_h + 1;
    const int64 output_w = (stats.input_w + stats.pad_w * 2 - stats.kernel_w) / stats.stride_w + 1;
    const int64 p = static_cast<int64>(output_h);
    const int64 q = static_cast<int64>(output_w);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    dy = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("DY")
                           .set_dim({n, k, p, q})
                           .set_stride({p * q * k, 1, q * k, k}));

    w = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("W")
                          .set_dim({k, c, r, s})
                          .set_stride({r * s * c, 1, s * c, c}));

    auto dgrad_options =
        fe::graph::Conv_dgrad_attributes()
            .set_pre_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_post_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_stride({static_cast<int64>(stats.stride_h), static_cast<int64>(stats.stride_w)})
            .set_dilation({1, 1});

    dx = graph->conv_dgrad(dy, w, dgrad_options);
    dx->set_dim({n, c, h, w_dim})
        .set_stride({h * w_dim * c, 1, w_dim * c, c})
        .set_data_type(io_type)
        .set_output(true);

    ensure_ok(graph->validate(), "conv_dgrad validate");
    ensure_ok(graph->build_operation_graph(handle), "conv_dgrad build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "conv_dgrad create plans");
    ensure_ok(graph->check_support(), "conv_dgrad check support");
    ensure_ok(graph->build_plans(), "conv_dgrad build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "conv_dgrad workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct conv2d_wgrad_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> dy;
  std::shared_ptr<fe::graph::Tensor_attributes> dw;

  size_t workspace_size;

  conv2d_wgrad_graph(cudnnHandle_t handle, const Conv2DStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.in_channels);
    const int64 h = static_cast<int64>(stats.input_h);
    const int64 w_dim = static_cast<int64>(stats.input_w);
    const int64 k = static_cast<int64>(stats.out_channels);
    const int64 r = static_cast<int64>(stats.kernel_h);
    const int64 s = static_cast<int64>(stats.kernel_w);
    const int64 output_h = (stats.input_h + stats.pad_h * 2 - stats.kernel_h) / stats.stride_h + 1;
    const int64 output_w = (stats.input_w + stats.pad_w * 2 - stats.kernel_w) / stats.stride_w + 1;
    const int64 p = static_cast<int64>(output_h);
    const int64 q = static_cast<int64>(output_w);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim({n, c, h, w_dim})
                          .set_stride({h * w_dim * c, 1, w_dim * c, c}));

    dy = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("DY")
                           .set_dim({n, k, p, q})
                           .set_stride({p * q * k, 1, q * k, k}));

    auto wgrad_options =
        fe::graph::Conv_wgrad_attributes()
            .set_pre_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_post_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_stride({static_cast<int64>(stats.stride_h), static_cast<int64>(stats.stride_w)})
            .set_dilation({1, 1});

    dw = graph->conv_wgrad(dy, x, wgrad_options);
    dw->set_output(true)
        .set_dim({k, c, r, s})
        .set_stride({r * s * c, 1, s * c, c})
        .set_data_type(io_type);

    ensure_ok(graph->validate(), "conv_wgrad validate");
    ensure_ok(graph->build_operation_graph(handle), "conv_wgrad build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "conv_wgrad create plans");
    ensure_ok(graph->check_support(), "conv_wgrad check support");
    ensure_ok(graph->build_plans(), "conv_wgrad build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "conv_wgrad workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct conv2d_bgrad_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> dy;
  std::shared_ptr<fe::graph::Tensor_attributes> db;

  size_t workspace_size;

  conv2d_bgrad_graph(cudnnHandle_t handle, const Conv2DStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 k = static_cast<int64>(stats.out_channels);
    const int64 output_h = (stats.input_h + stats.pad_h * 2 - stats.kernel_h) / stats.stride_h + 1;
    const int64 output_w = (stats.input_w + stats.pad_w * 2 - stats.kernel_w) / stats.stride_w + 1;
    const int64 p = static_cast<int64>(output_h);
    const int64 q = static_cast<int64>(output_w);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    dy = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("DY")
                           .set_dim({n, k, p, q})
                           .set_stride({p * q * k, 1, q * k, k}));

    auto reduction_options = fe::graph::Reduction_attributes().set_mode(fe::ReductionMode_t::ADD);

    db = graph->reduction(dy, reduction_options);
    db->set_output(true)
        .set_dim({1, k, 1, 1})
        .set_stride({k, 1, k, k})
        .set_data_type(io_type)
        .set_name("DB");

    ensure_ok(graph->validate(), "grad_bias validate");
    ensure_ok(graph->build_operation_graph(handle), "grad_bias build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "grad_bias create plans");
    ensure_ok(graph->check_support(), "grad_bias check support");
    ensure_ok(graph->build_plans(), "grad_bias build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "grad_bias workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct batchnorm_fwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> scale;
  std::shared_ptr<fe::graph::Tensor_attributes> bias;
  std::shared_ptr<fe::graph::Tensor_attributes> y;
  std::shared_ptr<fe::graph::Tensor_attributes> mean;
  std::shared_ptr<fe::graph::Tensor_attributes> invar;
  std::shared_ptr<fe::graph::Tensor_attributes> prev_mean;
  std::shared_ptr<fe::graph::Tensor_attributes> prev_var;
  std::shared_ptr<fe::graph::Tensor_attributes> next_mean;
  std::shared_ptr<fe::graph::Tensor_attributes> next_var;
  std::shared_ptr<fe::graph::Tensor_attributes> relu_mask;
  size_t workspace_size;

  batchnorm_fwd_graph(cudnnHandle_t handle, const BatchNormStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);
    const int64 h = static_cast<int64>(stats.height);
    const int64 w = static_cast<int64>(stats.width);
    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim({n, c, h, w})
                          .set_stride({h * w * c, 1, w * c, c})
                          .set_data_type(io_type));

    scale = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("scale")
                              .set_dim({1, c, 1, 1})
                              .set_stride({c, 1, c, c})
                              .set_data_type(compute_type));

    bias = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("bias")
                             .set_dim({1, c, 1, 1})
                             .set_stride({c, 1, c, c})
                             .set_data_type(compute_type));

    prev_mean = graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("prev_running_mean")
                                  .set_dim({1, c, 1, 1})
                                  .set_stride({c, 1, c, c})
                                  .set_data_type(compute_type));

    prev_var = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("prev_running_var")
                                 .set_dim({1, c, 1, 1})
                                 .set_stride({c, 1, c, c})
                                 .set_data_type(compute_type));

    auto epsilon = graph->tensor(stats.epsilon);
    auto momentum = graph->tensor(stats.momentum);

    auto bn_options = fe::graph::Batchnorm_attributes().set_epsilon(epsilon);
    bn_options.set_previous_running_stats(prev_mean, prev_var, momentum);

    auto outputs = graph->batchnorm(x, scale, bias, bn_options);
    y = outputs[0];
    mean = outputs[1];
    invar = outputs[2];
    next_mean = outputs[3];
    next_var = outputs[4];

    relu_mask = y;
    if (stats.use_relu) {
      y = graph->pointwise(
          y, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD));

      float lower_clip = 0.0f;
      auto s_lower_clip = graph->tensor(lower_clip);
      auto relu_lower_clip_mask_attr = fe::graph::Pointwise_attributes()
                                           .set_mode(fe::PointwiseMode_t::CMP_GT)
                                           .set_compute_data_type(compute_type);
      relu_mask = graph->pointwise(y, s_lower_clip, relu_lower_clip_mask_attr);
      relu_mask->set_output(true).set_data_type(fe::DataType_t::BOOLEAN);
    }

    y->set_output(true).set_data_type(io_type);
    mean->set_output(true).set_data_type(compute_type);
    invar->set_output(true).set_data_type(compute_type);
    next_mean->set_output(true).set_data_type(compute_type);
    next_var->set_output(true).set_data_type(compute_type);

    ensure_ok(graph->validate(), "batchnorm forward validate");
    ensure_ok(graph->build_operation_graph(handle), "batchnorm forward build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "batchnorm forward create plans");
    ensure_ok(graph->check_support(), "batchnorm forward check support");
    ensure_ok(graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
              "batchnorm forward build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "batchnorm forward workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct batchnorm_inf_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> scale;
  std::shared_ptr<fe::graph::Tensor_attributes> bias;
  std::shared_ptr<fe::graph::Tensor_attributes> saved_mean;
  std::shared_ptr<fe::graph::Tensor_attributes> saved_var;
  std::shared_ptr<fe::graph::Tensor_attributes> y;
  size_t workspace_size;

  batchnorm_inf_graph(cudnnHandle_t handle, const BatchNormStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);
    const int64 h = static_cast<int64>(stats.height);
    const int64 w = static_cast<int64>(stats.width);
    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim({n, c, h, w})
                          .set_stride({h * w * c, 1, w * c, c})
                          .set_data_type(io_type));

    scale = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("scale")
                              .set_dim({1, c, 1, 1})
                              .set_stride({c, 1, c, c})
                              .set_data_type(compute_type));

    bias = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("bias")
                             .set_dim({1, c, 1, 1})
                             .set_stride({c, 1, c, c})
                             .set_data_type(compute_type));

    saved_mean = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("saved_mean")
                                   .set_dim({1, c, 1, 1})
                                   .set_stride({c, 1, c, c})
                                   .set_data_type(compute_type));

    saved_var = graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("saved_var")
                                  .set_dim({1, c, 1, 1})
                                  .set_stride({c, 1, c, c})
                                  .set_data_type(compute_type));

    auto epsilon_tensor = graph->tensor(stats.epsilon);
    auto var_plus_eps =
        graph->pointwise(saved_var, epsilon_tensor,
                         fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
    auto inv_std = graph->pointwise(
        var_plus_eps, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RSQRT));
    auto x_minus_mean = graph->pointwise(
        x, saved_mean, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::SUB));
    auto normalized =
        graph->pointwise(x_minus_mean, inv_std,
                         fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL));
    auto scaled = graph->pointwise(
        normalized, scale, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL));
    y = graph->pointwise(scaled, bias,
                         fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));

    if (stats.use_relu) {
      y = graph->pointwise(
          y, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD));
    }

    y->set_output(true).set_data_type(io_type);

    ensure_ok(graph->validate(), "batchnorm inference validate");
    ensure_ok(graph->build_operation_graph(handle), "batchnorm inference build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "batchnorm inference create plans");
    ensure_ok(graph->check_support(), "batchnorm inference check support");
    ensure_ok(graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
              "batchnorm inference build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "batchnorm inference workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct batchnorm_bwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> dy;
  std::shared_ptr<fe::graph::Tensor_attributes> mask;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> scale;
  std::shared_ptr<fe::graph::Tensor_attributes> mean;
  std::shared_ptr<fe::graph::Tensor_attributes> invar;
  std::shared_ptr<fe::graph::Tensor_attributes> dx;
  std::shared_ptr<fe::graph::Tensor_attributes> dscale;
  std::shared_ptr<fe::graph::Tensor_attributes> dbias;
  size_t workspace_size;

  batchnorm_bwd_graph(cudnnHandle_t handle, const BatchNormStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);
    const int64 h = static_cast<int64>(stats.height);
    const int64 w = static_cast<int64>(stats.width);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    dy = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("DY")
                           .set_dim({n, c, h, w})
                           .set_stride({h * w * c, 1, w * c, c}));
    auto DX_drelu = dy;
    if (stats.use_relu) {
      mask = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Mask")
                               .set_dim({n, c, h, w})
                               .set_stride({h * w * c, 1, w * c, c})
                               .set_data_type(fe::DataType_t::BOOLEAN));

      auto mul_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
      DX_drelu = graph->pointwise(dy, mask, mul_options);
      DX_drelu->set_output(false).set_data_type(io_type);
    }

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim({n, c, h, w})
                          .set_stride({h * w * c, 1, w * c, c}));

    scale = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("scale")
                              .set_dim({1, c, 1, 1})
                              .set_stride({c, 1, c, c})
                              .set_data_type(fe::DataType_t::FLOAT));

    mean = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("mean")
                             .set_dim({1, c, 1, 1})
                             .set_stride({c, 1, c, c})
                             .set_data_type(fe::DataType_t::FLOAT));

    invar = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("inv_variance")
                              .set_dim({1, c, 1, 1})
                              .set_stride({c, 1, c, c})
                              .set_data_type(fe::DataType_t::FLOAT));

    auto DBN_options =
        fe::graph::Batchnorm_backward_attributes().set_saved_mean_and_inv_variance(mean, invar);
    auto outputs = graph->batchnorm_backward(DX_drelu, x, scale, DBN_options);
    dx = outputs[0];
    dscale = outputs[1];
    dbias = outputs[2];

    dx->set_output(true);
    dscale->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    dbias->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    ensure_ok(graph->validate(), "batchnorm backward validate");
    ensure_ok(graph->build_operation_graph(handle), "batchnorm backward build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "batchnorm backward create plans");
    ensure_ok(graph->check_support(), "batchnorm backward check support");
    ensure_ok(graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
              "batchnorm backward build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "batchnorm backward workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct layernorm_fwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> scale;
  std::shared_ptr<fe::graph::Tensor_attributes> bias;
  std::shared_ptr<fe::graph::Tensor_attributes> y;
  std::shared_ptr<fe::graph::Tensor_attributes> mean;
  std::shared_ptr<fe::graph::Tensor_attributes> inv_variance;
  size_t workspace_size;

  layernorm_fwd_graph(cudnnHandle_t handle, const LayerNormStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("input")
                          .set_dim({n, c, 1, 1})
                          .set_stride({c, 1, c, c})
                          .set_data_type(io_type));

    auto epsilon = graph->tensor(stats.epsilon);

    auto ln_options = fe::graph::Layernorm_attributes()
                          .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                          .set_epsilon(epsilon);

    scale = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("scale")
                              .set_dim({1, c, 1, 1})
                              .set_stride({c, 1, c, c})
                              .set_data_type(io_type));

    bias = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("bias")
                             .set_dim({1, c, 1, 1})
                             .set_stride({c, 1, c, c})
                             .set_data_type(io_type));

    auto outputs = graph->layernorm(x, scale, bias, ln_options);
    y = outputs[0];
    mean = outputs[1];
    inv_variance = outputs[2];

    y->set_output(true).set_data_type(io_type);
    mean->set_output(true).set_data_type(compute_type);
    inv_variance->set_output(true).set_data_type(compute_type);

    ensure_ok(graph->validate(), "layernorm_fwd validate");
    ensure_ok(graph->build_operation_graph(handle), "layernorm_fwd build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "layernorm_fwd create plans");
    ensure_ok(graph->check_support(), "layernorm_fwd check support");
    ensure_ok(graph->build_plans(), "layernorm_fwd build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "layernorm_fwd workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct layernorm_inf_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> scale;
  std::shared_ptr<fe::graph::Tensor_attributes> bias;
  std::shared_ptr<fe::graph::Tensor_attributes> y;
  size_t workspace_size;

  layernorm_inf_graph(cudnnHandle_t handle, const LayerNormStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("input")
                          .set_dim({n, c, 1, 1})
                          .set_stride({c, 1, c, c})
                          .set_data_type(io_type));

    auto epsilon = graph->tensor(stats.epsilon);

    auto ln_options = fe::graph::Layernorm_attributes()
                          .set_forward_phase(fe::NormFwdPhase_t::INFERENCE)
                          .set_epsilon(epsilon);

    scale = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("scale")
                              .set_dim({1, c, 1, 1})
                              .set_stride({c, 1, c, c})
                              .set_data_type(io_type));

    bias = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("bias")
                             .set_dim({1, c, 1, 1})
                             .set_stride({c, 1, c, c})
                             .set_data_type(io_type));

    auto outputs = graph->layernorm(x, scale, bias, ln_options);
    y = outputs[0];

    y->set_output(true).set_data_type(io_type);

    ensure_ok(graph->validate(), "layernorm_inf validate");
    ensure_ok(graph->build_operation_graph(handle), "layernorm_inf build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "layernorm_inf create plans");
    ensure_ok(graph->check_support(), "layernorm_inf check support");
    ensure_ok(graph->build_plans(), "layernorm_inf build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "layernorm_inf workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct layernorm_bwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> dy;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> scale;
  std::shared_ptr<fe::graph::Tensor_attributes> mean;
  std::shared_ptr<fe::graph::Tensor_attributes> inv_variance;
  std::shared_ptr<fe::graph::Tensor_attributes> dx;
  std::shared_ptr<fe::graph::Tensor_attributes> dscale;
  std::shared_ptr<fe::graph::Tensor_attributes> dbias;
  size_t workspace_size;

  layernorm_bwd_graph(cudnnHandle_t handle, const LayerNormStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    dy = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("grad_output")
                           .set_dim({n, c, 1, 1})
                           .set_stride({c, 1, c, c})
                           .set_data_type(io_type));

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("input")
                          .set_dim({n, c, 1, 1})
                          .set_stride({c, 1, c, c})
                          .set_data_type(io_type));

    mean = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("mean")
                             .set_dim({n, 1, 1, 1})
                             .set_stride({1, 1, 1, 1})
                             .set_data_type(compute_type));

    inv_variance = graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("inv_variance")
                                     .set_dim({n, 1, 1, 1})
                                     .set_stride({1, 1, 1, 1})
                                     .set_data_type(compute_type));

    auto ln_bwd_options =
        fe::graph::Layernorm_backward_attributes().set_saved_mean_and_inv_variance(mean,
                                                                                   inv_variance);

    scale = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("scale")
                              .set_dim({1, c, 1, 1})
                              .set_stride({c, 1, c, c})
                              .set_data_type(io_type));

    auto outputs = graph->layernorm_backward(dy, x, scale, ln_bwd_options);
    dx = outputs[0];
    dscale = outputs[1];
    dbias = outputs[2];

    dx->set_output(true).set_data_type(io_type);
    dscale->set_output(true).set_data_type(compute_type);
    dbias->set_output(true).set_data_type(compute_type);

    ensure_ok(graph->validate(), "layernorm_bwd validate");
    ensure_ok(graph->build_operation_graph(handle), "layernorm_bwd build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "layernorm_bwd create plans");
    ensure_ok(graph->check_support(), "layernorm_bwd check support");
    ensure_ok(graph->build_plans(), "layernorm_bwd build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "layernorm_bwd workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

template <typename T>
__global__ void avgpool_bwd_kernel(const T* __restrict__ grad_output, T* __restrict__ grad_input,
                                   int batch_size, int channels, int input_h, int input_w,
                                   int output_h, int output_w, int pool_h, int pool_w, int stride_h,
                                   int stride_w, int pad_h, int pad_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = batch_size * output_h * output_w * channels;
  if (idx >= total_outputs) return;

  int c = idx % channels;
  int ow = (idx / channels) % output_w;
  int oh = (idx / (channels * output_w)) % output_h;
  int b = idx / (channels * output_w * output_h);

  float grad = (float)grad_output[idx];

  int h_start = oh * stride_h - pad_h;
  int w_start = ow * stride_w - pad_w;
  int h_end = min(h_start + pool_h, input_h);
  int w_end = min(w_start + pool_w, input_w);
  h_start = max(h_start, 0);
  w_start = max(w_start, 0);

  int count = (h_end - h_start) * (w_end - w_start);
  if (count == 0) return;

  float grad_per_element = grad / count;
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      int input_idx = ((b * input_h + h) * input_w + w) * channels + c;
      tunx::cuda::gpu_atomic_add(&grad_input[input_idx], (T)grad_per_element);
    }
  }
}

template <typename T>
__global__ void class_token_fwd_kernel(const T* __restrict__ input, const T* __restrict__ token,
                                       T* __restrict__ output, size_t seq_len, size_t embed_dim,
                                       size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  size_t output_seq_len = seq_len + 1;
  size_t e = idx % embed_dim;
  size_t tmp = idx / embed_dim;
  size_t s_out = tmp % output_seq_len;
  size_t n = tmp / output_seq_len;

  if (s_out == 0) {
    output[idx] = token[e];
  } else {
    size_t s_in = s_out - 1;
    size_t in_idx = n * seq_len * embed_dim + s_in * embed_dim + e;
    output[idx] = input[in_idx];
  }
}

template <typename T>
__global__ void class_token_bwd_kernel(const T* __restrict__ grad_output,
                                       T* __restrict__ grad_input, T* __restrict__ grad_token,
                                       size_t batch_size, size_t seq_len, size_t embed_dim) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t output_seq_len = seq_len + 1;
  size_t total_elements = batch_size * output_seq_len * embed_dim;

  if (idx >= total_elements) return;

  size_t e = idx % embed_dim;
  size_t tmp = idx / embed_dim;
  size_t s_out = tmp % output_seq_len;
  size_t n = tmp / output_seq_len;

  if (s_out == 0) {
    tunx::cuda::gpu_atomic_add(&grad_token[e], grad_output[idx]);
  } else {
    size_t s_in = s_out - 1;
    size_t in_idx = n * seq_len * embed_dim + s_in * embed_dim + e;
    grad_input[in_idx] = grad_output[idx];
  }
}

template <typename T>
__global__ void dropout_fwd_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   bool* __restrict__ mask, float dropout_rate, float scale,
                                   unsigned long long seed, size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  unsigned int state = (unsigned int)(seed ^ idx ^ 0x12345678);
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  float rand_val = (state % 100000) / 100000.0f;

  bool keep = rand_val > dropout_rate;
  mask[idx] = keep;

  if (keep) {
    output[idx] = (T)((float)input[idx] * scale);
  } else {
    output[idx] = (T)0.0f;
  }
}

template <typename T>
__global__ void dropout_bwd_kernel(const T* __restrict__ grad_output, T* __restrict__ grad_input,
                                   const bool* __restrict__ mask, float scale,
                                   size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  if (mask[idx]) {
    grad_input[idx] = (T)((float)grad_output[idx] * scale);
  } else {
    grad_input[idx] = (T)0.0f;
  }
}

template <typename T>
__global__ void embedding_fwd_kernel(const T* __restrict__ input, const T* __restrict__ weight,
                                     T* __restrict__ output, size_t num_indices, size_t vocab_size,
                                     size_t embed_dim, size_t padding_idx) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_indices * embed_dim) return;

  size_t token_idx = tid / embed_dim;
  size_t dim_idx = tid % embed_dim;

  size_t vocab_idx = static_cast<size_t>(input[token_idx]);
  if (vocab_idx >= vocab_size) vocab_idx = 0;

  if (padding_idx < vocab_size && vocab_idx == padding_idx) {
    output[tid] = (T)0.0f;
    return;
  }

  output[tid] = weight[vocab_idx * embed_dim + dim_idx];
}

template <typename T>
__global__ void embedding_bwd_kernel(const T* __restrict__ input, const T* __restrict__ grad_output,
                                     T* __restrict__ grad_weight, size_t num_indices,
                                     size_t vocab_size, size_t embed_dim, size_t padding_idx) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_indices * embed_dim) return;

  size_t token_idx = tid / embed_dim;
  size_t dim_idx = tid % embed_dim;

  size_t vocab_idx = static_cast<size_t>(input[token_idx]);
  if (vocab_idx >= vocab_size) vocab_idx = 0;

  if (padding_idx < vocab_size && vocab_idx == padding_idx) return;

  T g_val = grad_output[tid];
  tunx::cuda::gpu_atomic_add(&grad_weight[vocab_idx * embed_dim + dim_idx], g_val);
}

struct avgpool2d_fwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> y;

  size_t workspace_size;

  avgpool2d_fwd_graph(cudnnHandle_t handle, const AvgPool2DStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);
    const int64 h = static_cast<int64>(stats.height);
    const int64 w = static_cast<int64>(stats.width);
    const int64 output_h = (h + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
    const int64 output_w = (w + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim({n, c, h, w})
                          .set_stride({h * w * c, 1, w * c, c})
                          .set_data_type(io_type));

    auto resample_options =
        fe::graph::Resample_attributes()
            .set_name("AvgPool")
            .set_resampling_mode(fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING)
            .set_padding_mode(fe::PaddingMode_t::ZERO_PAD)
            .set_window({static_cast<int64>(stats.pool_h), static_cast<int64>(stats.pool_w)})
            .set_pre_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_post_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_stride({static_cast<int64>(stats.stride_h), static_cast<int64>(stats.stride_w)})
            .set_compute_data_type(compute_type)
            .set_generate_index(false);

    auto outputs = graph->resample(x, resample_options);
    y = outputs[0];
    y->set_output(true)
        .set_dim({n, c, output_h, output_w})
        .set_stride({output_h * output_w * c, 1, output_w * c, c})
        .set_data_type(io_type);

    ensure_ok(graph->validate(), "avgpool2d_fwd validate");
    ensure_ok(graph->build_operation_graph(handle), "avgpool2d_fwd build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "avgpool2d_fwd create plans");
    ensure_ok(graph->check_support(), "avgpool2d_fwd check support");
    ensure_ok(graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
              "avgpool2d_fwd build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "avgpool2d_fwd workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

template <typename T>
__global__ void relu_fwd_kernel(const T* __restrict__ input, T* __restrict__ output,
                                bool* __restrict__ mask, size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  float val = (float)input[idx];
  bool m = val > 0.0f;
  mask[idx] = m;
  output[idx] = m ? (T)val : (T)0.0f;
}

template <typename T>
__global__ void relu_bwd_kernel(const T* __restrict__ grad_output, T* __restrict__ grad_input,
                                const bool* __restrict__ mask, size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  grad_input[idx] = mask[idx] ? grad_output[idx] : (T)0.0f;
}

struct maxpool2d_inf_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> y;

  size_t workspace_size;

  maxpool2d_inf_graph(cudnnHandle_t handle, const MaxPool2DStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);
    const int64 h = static_cast<int64>(stats.height);
    const int64 w = static_cast<int64>(stats.width);
    const int64 output_h = (h + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
    const int64 output_w = (w + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim({n, c, h, w})
                          .set_stride({h * w * c, 1, w * c, c})
                          .set_data_type(io_type));

    auto resample_options =
        fe::graph::Resample_attributes()
            .set_name("MaxPoolInfer")
            .set_resampling_mode(fe::ResampleMode_t::MAXPOOL)
            .set_padding_mode(fe::PaddingMode_t::NEG_INF_PAD)
            .set_window({static_cast<int64>(stats.pool_h), static_cast<int64>(stats.pool_w)})
            .set_pre_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_post_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_stride({static_cast<int64>(stats.stride_h), static_cast<int64>(stats.stride_w)})
            .set_compute_data_type(compute_type)
            .set_generate_index(false);

    auto outputs = graph->resample(x, resample_options);
    y = outputs[0];
    y->set_output(true)
        .set_dim({n, c, output_h, output_w})
        .set_stride({output_h * output_w * c, 1, output_w * c, c})
        .set_data_type(io_type);

    ensure_ok(graph->validate(), "maxpool2d_inf validate");
    ensure_ok(graph->build_operation_graph(handle), "maxpool2d_inf build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "maxpool2d_inf create plans");
    ensure_ok(graph->check_support(), "maxpool2d_inf check support");
    ensure_ok(graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
              "maxpool2d_inf build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "maxpool2d_inf workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct maxpool2d_fwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> y;
  std::shared_ptr<fe::graph::Tensor_attributes> mask;

  size_t workspace_size;

  maxpool2d_fwd_graph(cudnnHandle_t handle, const MaxPool2DStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);
    const int64 h = static_cast<int64>(stats.height);
    const int64 w = static_cast<int64>(stats.width);
    const int64 output_h = (h + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
    const int64 output_w = (w + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim({n, c, h, w})
                          .set_stride({h * w * c, 1, w * c, c})
                          .set_data_type(io_type));

    auto resample_options =
        fe::graph::Resample_attributes()
            .set_name("MaxPool")
            .set_resampling_mode(fe::ResampleMode_t::MAXPOOL)
            .set_padding_mode(fe::PaddingMode_t::NEG_INF_PAD)
            .set_window({static_cast<int64>(stats.pool_h), static_cast<int64>(stats.pool_w)})
            .set_pre_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_post_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_stride({static_cast<int64>(stats.stride_h), static_cast<int64>(stats.stride_w)})
            .set_compute_data_type(compute_type)
            .set_generate_index(true);

    auto outputs = graph->resample(x, resample_options);
    y = outputs[0];
    mask = outputs[1];
    y->set_output(true)
        .set_dim({n, c, output_h, output_w})
        .set_stride({output_h * output_w * c, 1, output_w * c, c})
        .set_data_type(io_type);
    mask->set_output(true)
        .set_dim({n, c, output_h, output_w})
        .set_stride({output_h * output_w * c, 1, output_w * c, c})
        .set_data_type(fe::DataType_t::INT32);

    ensure_ok(graph->validate(), "maxpool2d_fwd validate");
    ensure_ok(graph->build_operation_graph(handle), "maxpool2d_fwd build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "maxpool2d_fwd create plans");
    ensure_ok(graph->check_support(), "maxpool2d_fwd check support");
    ensure_ok(graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
              "maxpool2d_fwd build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "maxpool2d_fwd workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

template <typename T>
__global__ void maxpool2d_bwd_kernel(const T* grad_output, T* grad_input, const int32* mask_indices,
                                     size_t batch_size, size_t channels, size_t output_h,
                                     size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                                     size_t stride_w, size_t pad_h, size_t pad_w, size_t input_h,
                                     size_t input_w) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs) return;

  int32 rel_idx = mask_indices[idx];
  if (rel_idx >= 0) {
    size_t c = idx % channels;
    size_t ow = (idx / channels) % output_w;
    size_t oh = (idx / (channels * output_w)) % output_h;
    size_t b = idx / (channels * output_w * output_h);

    int rel_h = rel_idx / pool_w;
    int rel_w = rel_idx % pool_w;

    int h = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h) + rel_h;
    int w = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w) + rel_w;

    if (h >= 0 && h < static_cast<int>(input_h) && w >= 0 && w < static_cast<int>(input_w)) {
      size_t in_idx = ((b * input_h + h) * input_w + w) * channels + c;
      tunx::cuda::gpu_atomic_add(&grad_input[in_idx], static_cast<T>(grad_output[idx]));
    }
  }
}

struct sdpa_fwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> Q, K, V, O, Stats;

  sdpa_fwd_graph(cudnnHandle_t handle, const AttentionStats& stats, DTypeDesc type_desc) {
    const int64_t b = static_cast<int64_t>(stats.batch_size);
    const int64_t h = static_cast<int64_t>(stats.num_heads);
    const int64_t s = static_cast<int64_t>(stats.seq_len);
    const int64_t d = static_cast<int64_t>(stats.head_dim);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    Q = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Q")
                          .set_uid(1)
                          .set_dim({b, h, s, d})
                          .set_stride({h * s * d, s * d, d, 1}));

    K = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("K")
                          .set_uid(2)
                          .set_dim({b, h, s, d})
                          .set_stride({h * s * d, s * d, d, 1}));

    V = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("V")
                          .set_uid(3)
                          .set_dim({b, h, s, d})
                          .set_stride({h * s * d, s * d, d, 1}));

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("flash_attention")
                            .set_attn_scale(stats.attn_scale)
                            .set_generate_stats(true);

    if (stats.is_causal) {
      sdpa_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
          .set_diagonal_band_right_bound(0);
    }

    auto [out_O, out_Stats] = graph->sdpa(Q, K, V, sdpa_options);
    O = out_O;
    Stats = out_Stats;

    O->set_output(true).set_dim({b, h, s, d}).set_stride({h * s * d, s * d, d, 1}).set_uid(4);
    Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(5);

    ensure_ok(graph->validate(), "sdpa validate");
    ensure_ok(graph->build_operation_graph(handle), "sdpa build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "sdpa create plans");
    ensure_ok(graph->check_support(), "sdpa check support");
    ensure_ok(graph->build_plans(), "sdpa build plans");
  }
};

struct sdpa_bwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> Q, K, V, O, dO, Stats, dQ, dK, dV;

  sdpa_bwd_graph(cudnnHandle_t handle, const AttentionStats& stats, DTypeDesc type_desc) {
    const int64_t b = static_cast<int64_t>(stats.batch_size);
    const int64_t h = static_cast<int64_t>(stats.num_heads);
    const int64_t s = static_cast<int64_t>(stats.seq_len);
    const int64_t d = static_cast<int64_t>(stats.head_dim);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    Q = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Q")
                          .set_uid(1)
                          .set_dim({b, h, s, d})
                          .set_stride({h * s * d, s * d, d, 1}));

    K = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("K")
                          .set_uid(2)
                          .set_dim({b, h, s, d})
                          .set_stride({h * s * d, s * d, d, 1}));

    V = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("V")
                          .set_uid(3)
                          .set_dim({b, h, s, d})
                          .set_stride({h * s * d, s * d, d, 1}));

    O = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("O")
                          .set_uid(4)
                          .set_dim({b, h, s, d})
                          .set_stride({h * s * d, s * d, d, 1}));

    dO = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("dO")
                           .set_uid(5)
                           .set_dim({b, h, s, d})
                           .set_stride({h * s * d, s * d, d, 1}));

    Stats = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("Stats")
                              .set_uid(6)
                              .set_dim({b, h, s, 1})
                              .set_stride({h * s, s, 1, 1})
                              .set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_options = fe::graph::SDPA_backward_attributes()
                            .set_name("flash_attention_backward")
                            .set_attn_scale(stats.attn_scale);

    if (stats.is_causal) {
      sdpa_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
          .set_diagonal_band_right_bound(0);
    }

    auto [out_dQ, out_dK, out_dV] = graph->sdpa_backward(Q, K, V, O, dO, Stats, sdpa_options);
    dQ = out_dQ;
    dK = out_dK;
    dV = out_dV;

    dQ->set_output(true).set_uid(7).set_dim({b, h, s, d}).set_stride({h * s * d, s * d, d, 1});
    dK->set_output(true).set_uid(8).set_dim({b, h, s, d}).set_stride({h * s * d, s * d, d, 1});
    dV->set_output(true).set_uid(9).set_dim({b, h, s, d}).set_stride({h * s * d, s * d, d, 1});

    ensure_ok(graph->validate(), "sdpa_backward validate");
    ensure_ok(graph->build_operation_graph(handle), "sdpa_backward build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "sdpa_backward create plans");
    ensure_ok(graph->check_support(), "sdpa_backward check support");
    ensure_ok(graph->build_plans(), "sdpa_backward build plans");
  }
};

WorkspaceReq CuDNNEngine::query_dense_graph(void* backend_handle, const DenseStats& stats,
                                            DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);

  GraphCacheKey fwd_key{
      .op_type = OpType::DENSE_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_features, stats.out_features},
      .attributes = {{"use_bias", stats.use_bias}},
  };
  auto it_fwd = graph_cache_.find(fwd_key);
  if (it_fwd == graph_cache_.end()) {
    it_fwd = graph_cache_.emplace(fwd_key, dense_fwd_graph(handle, stats, type_desc)).first;
  }
  GraphCacheKey wgrad_key{
      .op_type = OpType::DENSE_WGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_features, stats.out_features},
      .attributes = {{"use_bias", stats.use_bias}},
  };
  auto it_wgrad = graph_cache_.find(wgrad_key);
  if (it_wgrad == graph_cache_.end()) {
    it_wgrad = graph_cache_.emplace(wgrad_key, dense_wgrad_graph(handle, stats, type_desc)).first;
  }
  GraphCacheKey dgrad_key{
      .op_type = OpType::DENSE_DGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_features, stats.out_features},
      .attributes = {{"use_bias", stats.use_bias}},
  };
  auto it_dgrad = graph_cache_.find(dgrad_key);
  if (it_dgrad == graph_cache_.end()) {
    it_dgrad = graph_cache_.emplace(dgrad_key, dense_dgrad_graph(handle, stats, type_desc)).first;
  }

  auto& fwd_graph = std::any_cast<dense_fwd_graph&>(it_fwd->second);
  auto& dgrad_graph = std::any_cast<dense_dgrad_graph&>(it_dgrad->second);
  auto& wgrad_graph = std::any_cast<dense_wgrad_graph&>(it_wgrad->second);
  size_t fwd_workspace = fwd_graph.workspace_size;
  size_t bwd_workspace = std::max({dgrad_graph.workspace_size, wgrad_graph.workspace_size});
  // TODO: add inf graph
  return {fwd_workspace, bwd_workspace, 0};
}

WorkspaceReq CuDNNEngine::query_avgpool_graph(void* backend_handle, const AvgPool2DStats& stats,
                                              DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);

  GraphCacheKey fwd_key{
      .op_type = OpType::AVG_POOL_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"pool_h", stats.pool_h},
                     {"pool_w", stats.pool_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w}},
  };
  auto it_fwd = graph_cache_.find(fwd_key);
  if (it_fwd == graph_cache_.end()) {
    it_fwd = graph_cache_.emplace(fwd_key, avgpool2d_fwd_graph(handle, stats, type_desc)).first;
  }
  auto& fwd_graph = std::any_cast<avgpool2d_fwd_graph&>(it_fwd->second);

  return {fwd_graph.workspace_size, 0, 0};
}

WorkspaceReq CuDNNEngine::query_maxpool2d_graph(void* backend_handle, const MaxPool2DStats& stats,
                                                DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);

  GraphCacheKey fwd_key{
      .op_type = OpType::MAXPOOL2D_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"pool_h", stats.pool_h},
                     {"pool_w", stats.pool_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w}},
  };
  auto it_fwd = graph_cache_.find(fwd_key);
  if (it_fwd == graph_cache_.end()) {
    it_fwd = graph_cache_.emplace(fwd_key, maxpool2d_fwd_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey inf_key{
      .op_type = OpType::MAXPOOL2D_INFER,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"pool_h", stats.pool_h},
                     {"pool_w", stats.pool_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w}},
  };
  auto it_inf = graph_cache_.find(inf_key);
  if (it_inf == graph_cache_.end()) {
    it_inf = graph_cache_.emplace(inf_key, maxpool2d_inf_graph(handle, stats, type_desc)).first;
  }

  auto& fwd_graph = std::any_cast<maxpool2d_fwd_graph&>(it_fwd->second);
  auto& inf_graph = std::any_cast<maxpool2d_inf_graph&>(it_inf->second);

  return {fwd_graph.workspace_size, 0, inf_graph.workspace_size};
}

WorkspaceReq CuDNNEngine::query_class_token_graph(void* backend_handle,
                                                  const ClassTokenStats& stats,
                                                  DTypeDesc type_desc) {
  size_t temp = stats.embed_dim * get_dtype_size(type_desc.param_dtype);
  return {0, temp, 0};
}

WorkspaceReq CuDNNEngine::query_dropout_graph(void* backend_handle, const DropoutStats& stats,
                                              DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CuDNNEngine::query_embedding_graph(void* backend_handle, const EmbeddingStats& stats,
                                                DTypeDesc type_desc) {
  size_t temp = stats.vocab_size * stats.embed_dim * get_dtype_size(type_desc.param_dtype);
  return {0, temp, 0};
}

WorkspaceReq CuDNNEngine::query_relu_graph(void* backend_handle, const ReLUStats& stats,
                                           DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CuDNNEngine::query_batchnorm_graph(void* backend_handle, const BatchNormStats& stats,
                                                DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);

  GraphCacheKey fwd_key{
      .op_type = OpType::BATCHNORM_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"epsilon", stats.epsilon},
                     {"momentum", stats.momentum},
                     {"use_relu", stats.use_relu}},
  };
  auto it_fwd = graph_cache_.find(fwd_key);
  if (it_fwd == graph_cache_.end()) {
    it_fwd = graph_cache_.emplace(fwd_key, batchnorm_fwd_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey inf_key{
      .op_type = OpType::BATCHNORM_INFER,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"epsilon", stats.epsilon},
                     {"momentum", stats.momentum},
                     {"use_relu", stats.use_relu}},
  };
  auto it_inf = graph_cache_.find(inf_key);
  if (it_inf == graph_cache_.end()) {
    it_inf = graph_cache_.emplace(inf_key, batchnorm_inf_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey bwd_key{
      .op_type = OpType::BATCHNORM_BWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"epsilon", stats.epsilon},
                     {"momentum", stats.momentum},
                     {"use_relu", stats.use_relu}},
  };
  auto it_bwd = graph_cache_.find(bwd_key);
  if (it_bwd == graph_cache_.end()) {
    it_bwd = graph_cache_.emplace(bwd_key, batchnorm_bwd_graph(handle, stats, type_desc)).first;
  }

  auto& fwd_graph = std::any_cast<batchnorm_fwd_graph&>(it_fwd->second);
  auto& inf_graph = std::any_cast<batchnorm_inf_graph&>(it_inf->second);
  auto& bwd_graph = std::any_cast<batchnorm_bwd_graph&>(it_bwd->second);

  size_t temp = 2 * stats.channels * get_dtype_size(type_desc.param_dtype);
  return {fwd_graph.workspace_size, bwd_graph.workspace_size + temp, inf_graph.workspace_size};
}

WorkspaceReq CuDNNEngine::query_conv2d_graph(void* backend_handle, const Conv2DStats& stats,
                                             DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);

  GraphCacheKey fwd_key{
      .op_type = OpType::CONV2D_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
               stats.input_w},
      .attributes = {{"kernel_h", stats.kernel_h},
                     {"kernel_w", stats.kernel_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w},
                     {"use_bias", stats.use_bias}},
  };
  auto it_fwd = graph_cache_.find(fwd_key);
  if (it_fwd == graph_cache_.end()) {
    it_fwd = graph_cache_.emplace(fwd_key, conv2d_fwd_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey dgrad_key{
      .op_type = OpType::CONV2D_DGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
               stats.input_w},
      .attributes = {{"kernel_h", stats.kernel_h},
                     {"kernel_w", stats.kernel_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w},
                     {"use_bias", stats.use_bias}},
  };
  auto it_dgrad = graph_cache_.find(dgrad_key);
  if (it_dgrad == graph_cache_.end()) {
    it_dgrad = graph_cache_.emplace(dgrad_key, conv2d_dgrad_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey wgrad_key{
      .op_type = OpType::CONV2D_WGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
               stats.input_w},
      .attributes = {{"kernel_h", stats.kernel_h},
                     {"kernel_w", stats.kernel_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w},
                     {"use_bias", stats.use_bias}},
  };
  auto it_wgrad = graph_cache_.find(wgrad_key);
  if (it_wgrad == graph_cache_.end()) {
    it_wgrad = graph_cache_.emplace(wgrad_key, conv2d_wgrad_graph(handle, stats, type_desc)).first;
  }

  auto& fwd_graph = std::any_cast<conv2d_fwd_graph&>(it_fwd->second);
  auto& dgrad_graph = std::any_cast<conv2d_dgrad_graph&>(it_dgrad->second);
  auto& wgrad_graph = std::any_cast<conv2d_wgrad_graph&>(it_wgrad->second);
  size_t wgrad_temp_size = stats.out_channels * stats.in_channels * stats.kernel_h *
                           stats.kernel_w * get_dtype_size(type_desc.param_dtype);

  size_t fwd_ws = fwd_graph.workspace_size;
  size_t bwd_ws =
      std::max(dgrad_graph.workspace_size, wgrad_graph.workspace_size + wgrad_temp_size);

  if (stats.use_bias) {
    GraphCacheKey bgrad_key{
        .op_type = OpType::CONV2D_BGRAD,
        .dtype_desc = type_desc,
        .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
                 stats.input_w},
        .attributes = {{"kernel_h", stats.kernel_h},
                       {"kernel_w", stats.kernel_w},
                       {"stride_h", stats.stride_h},
                       {"stride_w", stats.stride_w},
                       {"pad_h", stats.pad_h},
                       {"pad_w", stats.pad_w},
                       {"use_bias", stats.use_bias}},
    };
    auto it_bgrad = graph_cache_.find(bgrad_key);
    if (it_bgrad == graph_cache_.end()) {
      it_bgrad =
          graph_cache_.emplace(bgrad_key, conv2d_bgrad_graph(handle, stats, type_desc)).first;
    }
    auto& bgrad_graph = std::any_cast<conv2d_bgrad_graph&>(it_bgrad->second);
    size_t bgrad_temp_size = stats.out_channels * get_dtype_size(type_desc.param_dtype);
    bwd_ws = std::max(bwd_ws, bgrad_graph.workspace_size + bgrad_temp_size);
  }

  return {fwd_ws, bwd_ws, 0};
}

WorkspaceReq CuDNNEngine::query_sdpa_graph(void* backend_handle, const AttentionStats& stats,
                                           DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);

  GraphCacheKey fwd_key{
      .op_type = OpType::SDPA_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.num_heads, stats.seq_len, stats.head_dim},
      .attributes = {{"is_causal", stats.is_causal}, {"attn_scale", stats.attn_scale}},
  };
  auto it_fwd = graph_cache_.find(fwd_key);
  if (it_fwd == graph_cache_.end()) {
    it_fwd = graph_cache_.emplace(fwd_key, sdpa_fwd_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey bwd_key{
      .op_type = OpType::SDPA_BWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.num_heads, stats.seq_len, stats.head_dim},
      .attributes = {{"is_causal", stats.is_causal}, {"attn_scale", stats.attn_scale}},
  };
  auto it_bwd = graph_cache_.find(bwd_key);
  if (it_bwd == graph_cache_.end()) {
    it_bwd = graph_cache_.emplace(bwd_key, sdpa_bwd_graph(handle, stats, type_desc)).first;
  }

  auto& fwd_graph = std::any_cast<sdpa_fwd_graph&>(it_fwd->second);
  auto& bwd_graph = std::any_cast<sdpa_bwd_graph&>(it_bwd->second);

  int64_t fwd_workspace_size = 0;
  int64_t bwd_workspace_size = 0;
  ensure_ok(fwd_graph.graph->get_workspace_size(fwd_workspace_size), "sdpa fwd workspace");
  ensure_ok(bwd_graph.graph->get_workspace_size(bwd_workspace_size), "sdpa bwd workspace");

  return WorkspaceReq{static_cast<size_t>(fwd_workspace_size),
                      static_cast<size_t>(bwd_workspace_size), 0};
}

void CuDNNEngine::sdpa_fwd(void* backend_handle, const AttentionStats& stats, const void* q_data,
                           const void* k_data, const void* v_data, void* o_data, void* stats_data,
                           void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);

  GraphCacheKey key{
      .op_type = OpType::SDPA_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.num_heads, stats.seq_len, stats.head_dim},
      .attributes = {{"is_causal", stats.is_causal}, {"attn_scale", stats.attn_scale}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error(
        "cuDNN Graph not found for sdpa fwd. Please call query_sdpa_graph first.");
  }
  auto& graph_struct = std::any_cast<sdpa_fwd_graph&>(it->second);

  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
      {1, const_cast<void*>(q_data)},
      {2, const_cast<void*>(k_data)},
      {3, const_cast<void*>(v_data)},
      {4, o_data},
      {5, stats_data}};

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "sdpa_fwd execute");
}

void CuDNNEngine::sdpa_bwd(void* backend_handle, const AttentionStats& stats, const void* q_data,
                           const void* k_data, const void* v_data, const void* o_data,
                           const void* dO_data, const void* stats_data, void* dQ_data,
                           void* dK_data, void* dV_data, void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);

  GraphCacheKey key{
      .op_type = OpType::SDPA_BWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.num_heads, stats.seq_len, stats.head_dim},
      .attributes = {{"is_causal", stats.is_causal}, {"attn_scale", stats.attn_scale}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error(
        "cuDNN Graph not found for sdpa bwd. Please call query_sdpa_graph first.");
  }
  auto& graph_struct = std::any_cast<sdpa_bwd_graph&>(it->second);

  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
      {1, const_cast<void*>(q_data)},
      {2, const_cast<void*>(k_data)},
      {3, const_cast<void*>(v_data)},
      {4, const_cast<void*>(o_data)},
      {5, const_cast<void*>(dO_data)},
      {6, const_cast<void*>(stats_data)},
      {7, dQ_data},
      {8, dK_data},
      {9, dV_data}};

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "sdpa_bwd execute");
}

WorkspaceReq CuDNNEngine::query_layernorm_graph(void* backend_handle, const LayerNormStats& stats,
                                                DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);

  GraphCacheKey fwd_key{
      .op_type = OpType::LAYERNORM_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels},
      .attributes = {{"seq_len", stats.seq_len}, {"epsilon", stats.epsilon}},
  };
  auto it_fwd = graph_cache_.find(fwd_key);
  if (it_fwd == graph_cache_.end()) {
    it_fwd = graph_cache_.emplace(fwd_key, layernorm_fwd_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey inf_key{
      .op_type = OpType::LAYERNORM_INFER,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels},
      .attributes = {{"seq_len", stats.seq_len}, {"epsilon", stats.epsilon}},
  };
  auto it_inf = graph_cache_.find(inf_key);
  if (it_inf == graph_cache_.end()) {
    it_inf = graph_cache_.emplace(inf_key, layernorm_inf_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey bwd_key{
      .op_type = OpType::LAYERNORM_BWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels},
      .attributes = {{"seq_len", stats.seq_len}, {"epsilon", stats.epsilon}},
  };
  auto it_bwd = graph_cache_.find(bwd_key);
  if (it_bwd == graph_cache_.end()) {
    it_bwd = graph_cache_.emplace(bwd_key, layernorm_bwd_graph(handle, stats, type_desc)).first;
  }

  auto& fwd_graph = std::any_cast<layernorm_fwd_graph&>(it_fwd->second);
  auto& inf_graph = std::any_cast<layernorm_inf_graph&>(it_inf->second);
  auto& bwd_graph = std::any_cast<layernorm_bwd_graph&>(it_bwd->second);

  size_t temp = 2 * stats.channels * get_dtype_size(type_desc.param_dtype);
  return {fwd_graph.workspace_size, bwd_graph.workspace_size + temp, inf_graph.workspace_size};
}

void CuDNNEngine::dense_fwd(void* backend_handle, const DenseStats& stats, const void* input,
                            const void* weight, const void* bias, void* output, void* workspace,
                            DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::DENSE_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_features, stats.out_features},
      .attributes = {{"use_bias", stats.use_bias}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error(
        "cuDNN Graph not found for dense fwd. Please call query_dense_graph first.");
  }
  auto& graph_struct = std::any_cast<dense_fwd_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.w, const_cast<void*>(weight)},
      {graph_struct.b, const_cast<void*>(bias)},
      {graph_struct.y, output},
  };
  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "dense_fwd execute");
}

void CuDNNEngine::dense_wgrad(void* backend_handle, const DenseStats& stats,
                              const void* grad_output, const void* input, void* grad_weight,
                              void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::DENSE_WGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_features, stats.out_features},
      .attributes = {{"use_bias", stats.use_bias}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error(
        "cuDNN Graph not found for dense wgrad. Please call query_dense_graph first.");
  }
  auto& graph_struct = std::any_cast<dense_wgrad_graph&>(it->second);
  // for cudnn gemm, in-place accumulation is possible.
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.dy, const_cast<void*>(grad_output)},
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.dw, const_cast<void*>(grad_weight)},
      {graph_struct.dw_temp, grad_weight},
  };
  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "dense_wgrad execute");
}

void CuDNNEngine::dense_dgrad(void* backend_handle, const DenseStats& stats,
                              const void* grad_output, const void* weight, void* grad_input,
                              void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::DENSE_DGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_features, stats.out_features},
      .attributes = {{"use_bias", stats.use_bias}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error(
        "cuDNN Graph not found for dense dgrad. Please call query_dense_graph first.");
  }
  auto& graph_struct = std::any_cast<dense_dgrad_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.dy, const_cast<void*>(grad_output)},
      {graph_struct.w, const_cast<void*>(weight)},
      {graph_struct.dx, grad_input}};
  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "dense_dgrad execute");
}

void CuDNNEngine::dense_bgrad(void* backend_handle, const DenseStats& stats,
                              const void* grad_output, void* grad_bias, void* workspace,
                              DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  cudaStream_t stream = nullptr;
  cudnnGetStream(handle, &stream);

  size_t out_features = stats.out_features;
  int threads_per_block = 128;
  int warps_per_block = threads_per_block / 32;
  int num_blocks = (out_features + warps_per_block - 1) / warps_per_block;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    bgrad_reduce_accumulate_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_bias),
        static_cast<int>(stats.batch_size), static_cast<int>(stats.out_features));
  });

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to launch bgrad custom kernel: ") +
                             cudaGetErrorString(err));
  }
}

void CuDNNEngine::avgpool_fwd(void* backend_handle, const AvgPool2DStats& stats, const void* input,
                              void* output, void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::AVG_POOL_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"pool_h", stats.pool_h},
                     {"pool_w", stats.pool_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for avgpool fwd.");
  }
  auto& graph_struct = std::any_cast<avgpool2d_fwd_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)}, {graph_struct.y, output}};
  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "avgpool fwd execute");
}

void CuDNNEngine::avgpool_bwd(void* backend_handle, const AvgPool2DStats& stats,
                              const void* grad_output, void* grad_input, void* workspace,
                              DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;

  size_t total_outputs = stats.batch_size * output_h * output_w * stats.channels;
  int threads = 256;
  int blocks = (total_outputs + threads - 1) / threads;

  cudaMemsetAsync(grad_input, 0, total_outputs * get_dtype_size(type_desc.io_dtype), stream);
  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    avgpool_bwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_input), stats.batch_size,
        stats.channels, stats.height, stats.width, output_h, output_w, stats.pool_h, stats.pool_w,
        stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w);
  });
}

void CuDNNEngine::maxpool2d_fwd(void* backend_handle, const MaxPool2DStats& stats,
                                const void* input, void* output, void* mask, void* workspace,
                                DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::MAXPOOL2D_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"pool_h", stats.pool_h},
                     {"pool_w", stats.pool_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for maxpool2d fwd.");
  }
  auto& graph_struct = std::any_cast<maxpool2d_fwd_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.y, output},
      {graph_struct.mask, mask},
  };
  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "avgpool fwd execute");
}

void CuDNNEngine::maxpool2d_infer(void* backend_handle, const MaxPool2DStats& stats,
                                  const void* input, void* output, void* workspace,
                                  DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::MAXPOOL2D_INFER,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"pool_h", stats.pool_h},
                     {"pool_w", stats.pool_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error(
        "cuDNN Graph not found for maxpool2d infer. Please call query_maxpool2d_graph first.");
  }
  auto& graph_struct = std::any_cast<maxpool2d_inf_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.y, output},
  };
  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "maxpool2d infer execute");
}

void CuDNNEngine::maxpool2d_bwd(void* backend_handle, const MaxPool2DStats& stats,
                                const void* grad_output, void* grad_input, const void* mask,
                                void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream;
  cudnnGetStream(static_cast<cudnnHandle_t>(backend_handle), &stream);

  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  size_t total_outputs = stats.batch_size * output_h * output_w * stats.channels;
  int threads = 256;
  int blocks = (total_outputs + threads - 1) / threads;

  size_t total_inputs = stats.batch_size * stats.height * stats.width * stats.channels;

  cudaMemsetAsync(grad_input, 0, total_inputs * get_dtype_size(type_desc.io_dtype), stream);
  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    maxpool2d_bwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
        static_cast<const int32*>(mask), stats.batch_size, stats.channels, output_h, output_w,
        stats.pool_h, stats.pool_w, stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w,
        stats.height, stats.width);
  });
}

void CuDNNEngine::class_token_fwd(void* backend_handle, const ClassTokenStats& stats,
                                  const void* input, const void* token, void* output,
                                  void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t output_seq_len = stats.seq_len + 1;
  size_t total_elements = stats.batch_size * output_seq_len * stats.embed_dim;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    class_token_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(token), static_cast<T*>(output),
        stats.seq_len, stats.embed_dim, total_elements);
  });
}

void CuDNNEngine::class_token_bwd(void* backend_handle, const ClassTokenStats& stats,
                                  const void* grad_output, void* grad_input, void* grad_token,
                                  void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t output_seq_len = stats.seq_len + 1;
  size_t total_elements = stats.batch_size * output_seq_len * stats.embed_dim;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    class_token_bwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
        static_cast<T*>(grad_token), stats.batch_size, stats.seq_len, stats.embed_dim);
  });
}

void CuDNNEngine::dropout_fwd(void* backend_handle, const DropoutStats& stats, const void* input,
                              void* output, bool* mask, void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.batch_size * stats.channels * stats.spatial_size;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  float scale = 1.0f / (1.0f - static_cast<float>(stats.dropout_rate));

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    dropout_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), mask,
        static_cast<float>(stats.dropout_rate), scale, 0x12345678ULL, total_elements);
  });
}

void CuDNNEngine::dropout_bwd(void* backend_handle, const DropoutStats& stats,
                              const void* grad_output, void* grad_input, const bool* mask,
                              double scale, void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.batch_size * stats.channels * stats.spatial_size;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    dropout_bwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_input), mask,
        static_cast<float>(scale), total_elements);
  });
}

void CuDNNEngine::relu_fwd(void* backend_handle, const ReLUStats& stats, const void* input,
                           void* output, bool* mask, void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.batch_size * stats.spatial_size;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    relu_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), mask, total_elements);
  });
}

void CuDNNEngine::relu_bwd(void* backend_handle, const ReLUStats& stats, const void* grad_output,
                           void* grad_input, const bool* mask, void* workspace,
                           DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.batch_size * stats.spatial_size;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    relu_bwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_input), mask, total_elements);
  });
}

void CuDNNEngine::embedding_fwd(void* backend_handle, const EmbeddingStats& stats,
                                const void* input, const void* weight, void* output,
                                void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.num_indices * stats.embed_dim;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    embedding_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(weight), static_cast<T*>(output),
        stats.num_indices, stats.vocab_size, stats.embed_dim, stats.padding_idx);
  });
}

void CuDNNEngine::embedding_bwd(void* backend_handle, const EmbeddingStats& stats,
                                const void* grad_output, const void* input, void* grad_weight,
                                void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.num_indices * stats.embed_dim;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    embedding_bwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(grad_output),
        static_cast<T*>(grad_weight), stats.num_indices, stats.vocab_size, stats.embed_dim,
        stats.padding_idx);
  });
}

void CuDNNEngine::batchnorm_fwd(void* backend_handle, const BatchNormStats& stats,
                                const void* input, const void* gamma, const void* beta,
                                void* output, void* prev_running_mean, void* prev_running_var,
                                void* next_running_mean, void* next_running_var, void* batch_mean,
                                void* batch_invar, void* relu_mask, void* workspace,
                                DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::BATCHNORM_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"epsilon", stats.epsilon},
                     {"momentum", stats.momentum},
                     {"use_relu", stats.use_relu}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for batchnorm fwd.");
  }
  auto& graph_struct = std::any_cast<batchnorm_fwd_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.scale, const_cast<void*>(gamma)},
      {graph_struct.bias, const_cast<void*>(beta)},
      {graph_struct.y, output},
      {graph_struct.mean, batch_mean},
      {graph_struct.invar, batch_invar},
      {graph_struct.prev_mean, prev_running_mean},
      {graph_struct.prev_var, prev_running_var},
      {graph_struct.next_mean, next_running_mean},
      {graph_struct.next_var, next_running_var}};

  if (stats.use_relu) {
    variant_pack[graph_struct.relu_mask] = relu_mask;
  }

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "batchnorm fwd execute");
}

void CuDNNEngine::batchnorm_infer(void* backend_handle, const BatchNormStats& stats,
                                  const void* input, const void* gamma, const void* beta,
                                  const void* saved_mean, const void* saved_var, void* output,
                                  void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::BATCHNORM_INFER,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"epsilon", stats.epsilon},
                     {"momentum", stats.momentum},
                     {"use_relu", stats.use_relu}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for batchnorm infer.");
  }
  auto& graph_struct = std::any_cast<batchnorm_inf_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.scale, const_cast<void*>(gamma)},
      {graph_struct.bias, const_cast<void*>(beta)},
      {graph_struct.y, output},
      {graph_struct.saved_mean, const_cast<void*>(saved_mean)},
      {graph_struct.saved_var, const_cast<void*>(saved_var)}};

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "batchnorm infer execute");
}

void CuDNNEngine::batchnorm_bwd(void* backend_handle, const BatchNormStats& stats,
                                const void* grad_output, const void* input, const void* relu_mask,
                                const void* gamma, void* grad_input, void* grad_gamma,
                                void* grad_beta, const void* batch_mean, const void* batch_invar,
                                void* workspace, DTypeDesc type_desc) {
  size_t grad_gamma_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  size_t grad_beta_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  void* grad_gamma_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_gamma_temp_size;
  void* grad_beta_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_beta_temp_size;

  assert(grad_gamma != grad_gamma_temp && "grad_gamma should be different from grad_gamma_temp");
  assert(grad_beta != grad_beta_temp && "grad_beta should be different from grad_beta_temp");
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::BATCHNORM_BWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"epsilon", stats.epsilon},
                     {"momentum", stats.momentum},
                     {"use_relu", stats.use_relu}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for batchnorm bwd.");
  }
  auto& graph_struct = std::any_cast<batchnorm_bwd_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.dy, const_cast<void*>(grad_output)},
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.scale, const_cast<void*>(gamma)},
      {graph_struct.mean, const_cast<void*>(batch_mean)},
      {graph_struct.invar, const_cast<void*>(batch_invar)},
      {graph_struct.dx, grad_input},
      {graph_struct.dscale, grad_gamma_temp},
      {graph_struct.dbias, grad_beta_temp}};

  if (stats.use_relu) {
    variant_pack[graph_struct.mask] = const_cast<void*>(relu_mask);
  }

  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "batchnorm backward execute");

  size_t num_elements = stats.channels;
  tunx::cuda::axpy(grad_gamma_temp, grad_gamma, num_elements, type_desc.compute_dtype, stream);
  tunx::cuda::axpy(grad_beta_temp, grad_beta, num_elements, type_desc.compute_dtype, stream);
}

void CuDNNEngine::conv2d_fwd(void* backend_handle, const Conv2DStats& stats, const void* input,
                             const void* weight, const void* bias, void* output, void* workspace,
                             DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::CONV2D_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
               stats.input_w},
      .attributes = {{"kernel_h", stats.kernel_h},
                     {"kernel_w", stats.kernel_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w},
                     {"use_bias", stats.use_bias}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for conv2d fwd.");
  }
  auto& graph_struct = std::any_cast<conv2d_fwd_graph&>(it->second);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.w, const_cast<void*>(weight)},
      {graph_struct.y, output}};

  if (stats.use_bias && bias && graph_struct.b) {
    variant_pack[graph_struct.b] = const_cast<void*>(bias);
  }

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "conv2d fwd execute");
}

void CuDNNEngine::conv2d_dgrad(void* backend_handle, const Conv2DStats& stats,
                               const void* grad_output, const void* weight, void* grad_input,
                               void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::CONV2D_DGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
               stats.input_w},
      .attributes = {{"kernel_h", stats.kernel_h},
                     {"kernel_w", stats.kernel_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w},
                     {"use_bias", stats.use_bias}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for conv2d dgrad.");
  }
  auto& graph_struct = std::any_cast<conv2d_dgrad_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.dy, const_cast<void*>(grad_output)},
      {graph_struct.w, const_cast<void*>(weight)},
      {graph_struct.dx, grad_input}};

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "conv2d dgrad execute");
}

void CuDNNEngine::conv2d_wgrad(void* backend_handle, const Conv2DStats& stats,
                               const void* grad_output, const void* input, void* grad_weight,
                               void* workspace, DTypeDesc type_desc) {
  size_t grad_weight_temp_size = stats.out_channels * stats.kernel_h * stats.kernel_w *
                                 stats.in_channels * get_dtype_size(type_desc.param_dtype);
  void* grad_weight_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_weight_temp_size;

  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::CONV2D_WGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
               stats.input_w},
      .attributes = {{"kernel_h", stats.kernel_h},
                     {"kernel_w", stats.kernel_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w},
                     {"use_bias", stats.use_bias}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for conv2d wgrad.");
  }
  auto& graph_struct = std::any_cast<conv2d_wgrad_graph&>(it->second);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.dy, const_cast<void*>(grad_output)},
      {graph_struct.dw, grad_weight_temp}};
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "conv2d wgrad execute");

  size_t num_elements =
      static_cast<size_t>(stats.out_channels) * stats.in_channels * stats.kernel_h * stats.kernel_w;
  tunx::cuda::axpy(grad_weight_temp, grad_weight, num_elements, type_desc.param_dtype, stream);
}

void CuDNNEngine::conv2d_bgrad(void* backend_handle, const Conv2DStats& stats,
                               const void* grad_output, void* grad_bias, void* workspace,
                               DTypeDesc type_desc) {
  size_t grad_bias_temp_size = stats.out_channels * get_dtype_size(type_desc.param_dtype);
  void* grad_bias_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_bias_temp_size;

  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::CONV2D_BGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
               stats.input_w},
      .attributes = {{"kernel_h", stats.kernel_h},
                     {"kernel_w", stats.kernel_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w},
                     {"use_bias", stats.use_bias}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for conv2d bgrad.");
  }
  auto& graph_struct = std::any_cast<conv2d_bgrad_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.dy, const_cast<void*>(grad_output)},
      {graph_struct.db, grad_bias_temp},
  };

  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "conv2d bgrad execute");

  size_t num_elements = stats.out_channels;

  tunx::cuda::axpy(grad_bias_temp, grad_bias, num_elements, type_desc.param_dtype, stream);
}

void CuDNNEngine::layernorm_fwd(void* backend_handle, const LayerNormStats& stats,
                                const void* input, const void* gamma, const void* beta,
                                void* output, void* mean, void* inv_variance, void* workspace,
                                DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::LAYERNORM_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels},
      .attributes = {{"seq_len", stats.seq_len}, {"epsilon", stats.epsilon}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for layernorm fwd.");
  }
  auto& graph_struct = std::any_cast<layernorm_fwd_graph&>(it->second);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
  variant_pack[graph_struct.x] = const_cast<void*>(input);
  variant_pack[graph_struct.scale] = const_cast<void*>(gamma);
  variant_pack[graph_struct.bias] = const_cast<void*>(beta);
  variant_pack[graph_struct.y] = output;
  variant_pack[graph_struct.mean] = mean;
  variant_pack[graph_struct.inv_variance] = inv_variance;

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "layernorm fwd execute");
}

void CuDNNEngine::layernorm_infer(void* backend_handle, const LayerNormStats& stats,
                                  const void* input, const void* gamma, const void* beta,
                                  void* output, void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::LAYERNORM_INFER,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels},
      .attributes = {{"seq_len", stats.seq_len}, {"epsilon", stats.epsilon}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error(
        "cuDNN Graph not found for layernorm infer. Please call query_layernorm_graph first.");
  }
  auto& graph_struct = std::any_cast<layernorm_inf_graph&>(it->second);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
  variant_pack[graph_struct.x] = const_cast<void*>(input);
  variant_pack[graph_struct.scale] = const_cast<void*>(gamma);
  variant_pack[graph_struct.bias] = const_cast<void*>(beta);
  variant_pack[graph_struct.y] = output;

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "layernorm infer execute");
}

void CuDNNEngine::layernorm_bwd(void* backend_handle, const LayerNormStats& stats,
                                const void* grad_output, const void* input, const void* gamma,
                                const void* mean, const void* inv_variance, void* grad_input,
                                void* grad_gamma, void* grad_beta, void* workspace,
                                DTypeDesc type_desc) {
  size_t grad_gamma_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  size_t grad_beta_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  void* grad_gamma_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_gamma_temp_size;
  void* grad_beta_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_beta_temp_size;

  assert(grad_gamma != grad_gamma_temp && "grad_gamma should be different from grad_gamma_temp");
  assert(grad_beta != grad_beta_temp && "grad_beta should be different from grad_beta_temp");
  cudnnHandle_t handle = static_cast<cudnnHandle_t>(backend_handle);
  GraphCacheKey key{
      .op_type = OpType::LAYERNORM_BWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels},
      .attributes = {{"seq_len", stats.seq_len}, {"epsilon", stats.epsilon}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for layernorm bwd.");
  }
  auto& graph_struct = std::any_cast<layernorm_bwd_graph&>(it->second);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
  variant_pack[graph_struct.dy] = const_cast<void*>(grad_output);
  variant_pack[graph_struct.x] = const_cast<void*>(input);
  variant_pack[graph_struct.mean] = const_cast<void*>(mean);
  variant_pack[graph_struct.inv_variance] = const_cast<void*>(inv_variance);
  variant_pack[graph_struct.scale] = const_cast<void*>(gamma);
  variant_pack[graph_struct.dx] = grad_input;
  variant_pack[graph_struct.dscale] = grad_gamma_temp;
  variant_pack[graph_struct.dbias] = grad_beta_temp;

  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "layernorm bwd execute");

  size_t num_elements = stats.channels;
  tunx::cuda::axpy(grad_gamma_temp, grad_gamma, num_elements, type_desc.compute_dtype, stream);
  tunx::cuda::axpy(grad_beta_temp, grad_beta, num_elements, type_desc.compute_dtype, stream);
}

}  // end namespace tunx
