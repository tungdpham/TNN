
#include "nn/blocks_impl/cuda/cudnn_flash_attention_ops.hpp"

#ifdef USE_CUDNN

#include <cudnn_frontend.h>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace tnn {
namespace cuda {
namespace cudnn_flash_attention {

namespace fe = cudnn_frontend;

constexpr fe::graph::Tensor_attributes::uid_t Q_UID = 1;
constexpr fe::graph::Tensor_attributes::uid_t K_UID = 2;
constexpr fe::graph::Tensor_attributes::uid_t V_UID = 3;
constexpr fe::graph::Tensor_attributes::uid_t O_UID = 4;

static void ensure_fe_ok(const fe::error_t &status, const std::string &stage) {
  if (status.is_bad()) {
    throw std::runtime_error("cuDNN SDPA error at " + stage + ": " + status.err_msg);
  }
}

GraphPtr create_sdpa_forward_graph(int64_t b, int64_t h, int64_t s, int64_t d, float attn_scale,
                                   bool is_causal) {
  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

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
      fe::graph::SDPA_attributes().set_name("flash_attention").set_attn_scale(attn_scale);

  if (is_causal) {
    sdpa_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
        .set_diagonal_band_right_bound(0);
  }

  auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

  O->set_output(true).set_dim({b, h, s, d}).set_stride({h * s * d, s * d, d, 1}).set_uid(O_UID);

  (void)Stats;
  return graph;
}

void build_sdpa_forward_graph(const GraphPtr &graph, cudnnHandle_t handle) {
  ensure_fe_ok(graph->build(handle, {fe::HeurMode_t::A}), "sdpa build");
}

size_t get_sdpa_forward_workspace_bytes(const GraphPtr &graph) {
  int64_t workspace_size = 0;
  ensure_fe_ok(graph->get_workspace_size(workspace_size), "sdpa workspace size");
  return static_cast<size_t>(workspace_size);
}

void run_sdpa_forward(const GraphPtr &graph, cudnnHandle_t handle, void *q, void *k, void *v,
                      void *o, void *workspace, cudaStream_t stream) {
  cudnnSetStream(handle, stream);
  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> variant_pack = {
      {Q_UID, q}, {K_UID, k}, {V_UID, v}, {O_UID, o}};

  auto status = graph->execute(handle, variant_pack, workspace);
  ensure_fe_ok(status, "sdpa execute");
}

} // namespace cudnn_flash_attention
} // namespace cuda
} // namespace tnn

#endif
