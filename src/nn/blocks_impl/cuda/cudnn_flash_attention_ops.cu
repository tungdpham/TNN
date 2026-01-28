
#include "nn/blocks_impl/cuda/cudnn_flash_attention_ops.hpp"

#ifdef USE_CUDNN

#include "nn/blocks_impl/common/flash_attention.hpp"
#include <cudnn_frontend.h>
#include <stdexcept>
#include <string>

namespace tnn {
namespace cuda {
namespace cudnn_flash_attention {

namespace fe = cudnn_frontend;

constexpr fe::graph::Tensor_attributes::uid_t Q_UID = 1;
constexpr fe::graph::Tensor_attributes::uid_t K_UID = 2;
constexpr fe::graph::Tensor_attributes::uid_t V_UID = 3;
constexpr fe::graph::Tensor_attributes::uid_t O_UID = 4;

struct feHandle_t {
  cudnnHandle_t cudnn_handle = nullptr;
  cudnnDataType_t io_data_type;
  cudnnDataType_t compute_data_type;

  std::shared_ptr<fe::graph::Graph> fwd_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_q;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_k;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_v;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_o;
};

static void ensure_fe_ok(const fe::error_t &status, const std::string &stage) {
  if (status.is_bad()) {
    throw std::runtime_error("cuDNN SDPA error at " + stage + ": " + status.err_msg);
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
    throw std::runtime_error("Unsupported CUDNN data type");
  }
}

static fe::DataType_t to_fe_compute_type(cudnnDataType_t data_type) {
  if (data_type == CUDNN_DATA_HALF || data_type == CUDNN_DATA_BFLOAT16) {
    return fe::DataType_t::FLOAT;
  }
  return to_fe_data_type(data_type);
}

static void build_fwd_graph(feHandle_t *handle, AttentionStats &stats) {
  const int64_t b = static_cast<int64_t>(stats.batch_size);
  const int64_t h = static_cast<int64_t>(stats.num_heads);
  const int64_t s = static_cast<int64_t>(stats.seq_len);
  const int64_t d = static_cast<int64_t>(stats.head_dim);

  auto io_type = to_fe_data_type(handle->io_data_type);
  auto compute_type = to_fe_compute_type(handle->compute_data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type);

  auto Q = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("Q")
                             .set_uid(Q_UID)
                             .set_dim({b, h, s, d})
                             .set_stride({h * s * d, s * d, d, 1}));

  auto K = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("K")
                             .set_uid(K_UID)
                             .set_dim({b, h, s, d})
                             .set_stride({h * s * d, s * d, d, 1}));

  auto V = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("V")
                             .set_uid(V_UID)
                             .set_dim({b, h, s, d})
                             .set_stride({h * s * d, s * d, d, 1}));

  auto sdpa_options =
      fe::graph::SDPA_attributes().set_name("flash_attention").set_attn_scale(stats.attn_scale);

  if (stats.is_causal) {
    sdpa_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
        .set_diagonal_band_right_bound(0);
  }

  auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

  O->set_output(true).set_dim({b, h, s, d}).set_stride({h * s * d, s * d, d, 1}).set_uid(O_UID);

  (void)Stats;

  ensure_fe_ok(graph->validate(), "sdpa validate");
  ensure_fe_ok(graph->build_operation_graph(handle->cudnn_handle), "sdpa build op graph");
  ensure_fe_ok(graph->create_execution_plans({fe::HeurMode_t::A}), "sdpa create plans");
  ensure_fe_ok(graph->check_support(), "sdpa check support");
  ensure_fe_ok(graph->build_plans(), "sdpa build plans");

  int64_t workspace_size = 0;
  ensure_fe_ok(graph->get_workspace_size(workspace_size), "sdpa workspace");

  handle->fwd_graph = graph;
  handle->fwd_q = Q;
  handle->fwd_k = K;
  handle->fwd_v = V;
  handle->fwd_o = O;
  stats.fwd_workspace_size = static_cast<size_t>(workspace_size);
}

static void rebuild_all_graphs(feHandle_t *handle, AttentionStats &stats) {
  build_fwd_graph(handle, stats);
}

feHandle_t *initialize_fe_handle(cudnnHandle_t cudnn_handle, cudnnDataType_t io_data_type,
                                 cudnnDataType_t compute_data_type, AttentionStats &stats) {
  feHandle_t *handle = new feHandle_t();
  handle->cudnn_handle = cudnn_handle;
  handle->io_data_type = io_data_type;
  handle->compute_data_type = compute_data_type;

  rebuild_all_graphs(handle, stats);

  return handle;
}

void destroy_fe_handle(feHandle_t *handle) {
  if (handle) {
    delete handle;
  }
}

void run_forward(feHandle_t *handle, const AttentionStats &stats, const void *q_data,
                 const void *k_data, const void *v_data, void *o_data, void *workspace,
                 cudaStream_t stream) {
  cudnnSetStream(handle->cudnn_handle, stream);
  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> variant_pack = {
      {Q_UID, const_cast<void *>(q_data)},
      {K_UID, const_cast<void *>(k_data)},
      {V_UID, const_cast<void *>(v_data)},
      {O_UID, o_data}};

  auto status = handle->fwd_graph->execute(handle->cudnn_handle, variant_pack, workspace);
  ensure_fe_ok(status, "sdpa execute");
}

} // namespace cudnn_flash_attention
} // namespace cuda
} // namespace tnn

#endif
