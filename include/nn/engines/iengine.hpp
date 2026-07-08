#pragma once

#include <functional>
#include <vector>

#include "nn/stats/stats.hpp"
#include "type/type.hpp"

namespace tunx {

struct DTypeDesc {
  DType_t io_dtype;
  DType_t param_dtype;
  DType_t compute_dtype;
};

inline bool operator==(const DTypeDesc& lhs, const DTypeDesc& rhs) {
  return lhs.io_dtype == rhs.io_dtype && lhs.param_dtype == rhs.param_dtype &&
         lhs.compute_dtype == rhs.compute_dtype;
}

enum OpType {
  DENSE_FWD,
  DENSE_WGRAD,
  DENSE_DGRAD,
  DENSE_BGRAD,
  DENSE_ADD_BIAS,
  AVG_POOL_FWD,
  AVG_POOL_BWD,
  MAXPOOL2D_FWD,
  MAXPOOL2D_INFER,
  MAXPOOL2D_BWD,
  CLASS_TOKEN_FWD,
  CLASS_TOKEN_BWD,
  DROPOUT_FWD,
  DROPOUT_BWD,
  RELU_FWD,
  RELU_BWD,
  EMBEDDING_FWD,
  EMBEDDING_BWD,
  BATCHNORM_FWD,
  BATCHNORM_INFER,
  BATCHNORM_BWD,
  CONV2D_FWD,
  CONV2D_DGRAD,
  CONV2D_WGRAD,
  CONV2D_BGRAD,
  LAYERNORM_FWD,
  LAYERNORM_INFER,
  LAYERNORM_BWD,
};

struct GraphCacheKey {
  OpType op_type;
  DTypeDesc dtype_desc;
  std::vector<size_t> dims;
  std::unordered_map<std::string, float> attributes;
};

inline bool operator==(const GraphCacheKey& lhs, const GraphCacheKey& rhs) {
  return lhs.op_type == rhs.op_type && lhs.dtype_desc == rhs.dtype_desc && lhs.dims == rhs.dims &&
         lhs.attributes == rhs.attributes;
}

}  // namespace tunx

namespace std {

template <>
struct hash<tunx::GraphCacheKey> {
  size_t operator()(const tunx::GraphCacheKey& key) const {
    size_t h = hash<int>()(static_cast<int>(key.op_type));
    auto hash_combine = [](size_t& seed, size_t val) {
      seed ^= val + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    };
    hash_combine(h, hash<tunx::DType_t>()(key.dtype_desc.io_dtype));
    hash_combine(h, hash<tunx::DType_t>()(key.dtype_desc.param_dtype));
    hash_combine(h, hash<tunx::DType_t>()(key.dtype_desc.compute_dtype));
    for (size_t dim : key.dims) {
      hash_combine(h, hash<size_t>()(dim));
    }
    return h;
  }
};
}  // namespace std

namespace tunx {

class IEngine {
public:
  virtual ~IEngine() = default;

  // create a backend handle
  virtual void* create_backend_handle() = 0;

  // Query graph will build the graph from stats and type_desc if not yet exists.
  // In both cases, will return workspace requirement.

  /**
   * @brief Queries the workspace memory requirement for Dense graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Dense layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  virtual WorkspaceReq query_dense_graph(void* backend_handle, const DenseStats& stats,
                                         DTypeDesc type_desc) = 0;

  /**
   * @brief Queries the workspace memory requirement for AvgPool graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats AvgPool layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  virtual WorkspaceReq query_avgpool_graph(void* backend_handle, const AvgPool2DStats& stats,
                                           DTypeDesc type_desc) = 0;

  /**
   * @brief Queries the workspace memory requirement for MaxPool2D graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats MaxPool layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  virtual WorkspaceReq query_maxpool2d_graph(void* backend_handle, const MaxPool2DStats& stats,
                                             DTypeDesc type_desc) = 0;

  /**
   * @brief Queries the workspace memory requirement for ClassToken graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats ClassToken layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  virtual WorkspaceReq query_class_token_graph(void* backend_handle, const ClassTokenStats& stats,
                                               DTypeDesc type_desc) = 0;

  /**
   * @brief Queries the workspace memory requirement for Dropout graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Dropout layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  virtual WorkspaceReq query_dropout_graph(void* backend_handle, const DropoutStats& stats,
                                           DTypeDesc type_desc) = 0;

  /**
   * @brief Queries the workspace memory requirement for Embedding graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Embedding layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  virtual WorkspaceReq query_embedding_graph(void* backend_handle, const EmbeddingStats& stats,
                                             DTypeDesc type_desc) = 0;

  /**
   * @brief Queries the workspace memory requirement for ReLU graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats ReLU layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  virtual WorkspaceReq query_relu_graph(void* backend_handle, const ReLUStats& stats,
                                        DTypeDesc type_desc) = 0;

  /**
   * @brief Queries the workspace memory requirement for BatchNorm graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats BatchNorm layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward, backward, and inference workspace size in bytes.
   */
  virtual WorkspaceReq query_batchnorm_graph(void* backend_handle, const BatchNormStats& stats,
                                             DTypeDesc type_desc) = 0;

  /**
   * @brief Queries the workspace memory requirement for Conv2D graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Conv2D layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  virtual WorkspaceReq query_conv2d_graph(void* backend_handle, const Conv2DStats& stats,
                                          DTypeDesc type_desc) = 0;

  /**
   * @brief Queries the workspace memory requirement for LayerNorm graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats LayerNorm layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  virtual WorkspaceReq query_layernorm_graph(void* backend_handle, const LayerNormStats& stats,
                                             DTypeDesc type_desc) = 0;

  /**
   * @brief Forward pass for a Dense (Linear) layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Dense layer configuration.
   * @param input Input tensor. Shape: [batch_size, in_features], DType: io_dtype.
   * @param weight Weight tensor. Shape: [out_features, in_features], DType: param_dtype.
   * @param bias Bias tensor. Shape: [1, out_features], DType: param_dtype.
   * @param output Output tensor. Shape: [batch_size, out_features], DType: io_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void dense_fwd(void* backend_handle, const DenseStats& stats, const void* input,
                         const void* weight, const void* bias, void* output, void* workspace,
                         DTypeDesc type_desc) = 0;

  /**
   * @brief Computes weight gradients for a Dense layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Dense layer configuration.
   * @param grad_output Gradient w.r.t output. Shape: [batch_size, out_features], DType: io_dtype.
   * @param input Input tensor. Shape: [batch_size, in_features], DType: io_dtype.
   * @param grad_weight Gradient from previous steps (for accumulation).
   * @param grad_weight_temp Accumulated weight gradient output. Shape: [out_features, in_features],
   * DType: param_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void dense_wgrad(void* backend_handle, const DenseStats& stats, const void* grad_output,
                           const void* input, void* grad_weight, void* workspace,
                           DTypeDesc type_desc) = 0;

  /**
   * @brief Computes data gradients for a Dense layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Dense layer configuration.
   * @param grad_output Gradient w.r.t output. Shape: [batch_size, out_features], DType: io_dtype.
   * @param weight Weight tensor. Shape: [out_features, in_features], DType: param_dtype.
   * @param grad_input Computed gradient w.r.t input. Shape: [batch_size, in_features], DType:
   * io_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void dense_dgrad(void* backend_handle, const DenseStats& stats, const void* grad_output,
                           const void* weight, void* grad_input, void* workspace,
                           DTypeDesc type_desc) = 0;

  /**
   * @brief Computes bias gradients for a Dense layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Dense layer configuration.
   * @param grad_output Gradient w.r.t output. Shape: [batch_size, out_features], DType: io_dtype.
   * @param grad_bias Gradient from previous steps.
   * @param grad_bias_temp Accumulated bias gradient output. Shape: [1, out_features], DType:
   * param_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void dense_bgrad(void* backend_handle, const DenseStats& stats, const void* grad_output,
                           void* grad_bias, void* workspace, DTypeDesc type_desc) = 0;

  /**
   * @brief Forward pass for an AvgPool layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats AvgPool layer configuration.
   * @param input Input tensor. NHWC Shape: [batch_size, height, width, channels], DType: io_dtype.
   * @param output Output tensor. NHWC Shape: [batch_size, output_h, output_w, channels], DType:
   * io_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void avgpool_fwd(void* backend_handle, const AvgPool2DStats& stats, const void* input,
                           void* output, void* workspace, DTypeDesc type_desc) = 0;

  /**
   * @brief Backward pass for an AvgPool layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats AvgPool layer configuration.
   * @param grad_output Gradient w.r.t output. NHWC Shape: [batch_size, output_h, output_w,
   * channels], DType: io_dtype.
   * @param grad_input Computed gradient w.r.t input. NHWC Shape: [batch_size, height, width,
   * channels], DType: io_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void avgpool_bwd(void* backend_handle, const AvgPool2DStats& stats,
                           const void* grad_output, void* grad_input, void* workspace,
                           DTypeDesc type_desc) = 0;

  /**
   * @brief Forward pass for a MaxPool2D layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats MaxPool layer configuration.
   * @param input Input tensor. NHWC Shape: [batch_size, height, width, channels], DType: io_dtype.
   * @param output Output tensor. NHWC Shape: [batch_size, output_h, output_w, channels], DType:
   * io_dtype.
   * @param mask Mask tensor (index) output. NHWC Shape: [batch_size, output_h, output_w, channels],
   * DType: int8. Represents max position relative to pool window.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void maxpool2d_fwd(void* backend_handle, const MaxPool2DStats& stats, const void* input,
                             void* output, void* mask, void* workspace, DTypeDesc type_desc) = 0;

  /**
   * @brief Forward pass for a MaxPool2D layer (inference mode, no mask output).
   * @param backend_handle Opaque handle to the backend context.
   * @param stats MaxPool layer configuration.
   * @param input Input tensor. NHWC Shape: [batch_size, height, width, channels], DType: io_dtype.
   * @param output Output tensor. NHWC Shape: [batch_size, output_h, output_w, channels], DType:
   * io_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void maxpool2d_infer(void* backend_handle, const MaxPool2DStats& stats, const void* input,
                               void* output, void* workspace, DTypeDesc type_desc) = 0;

  /**
   * @brief Backward pass for a MaxPool2D layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats MaxPool layer configuration.
   * @param grad_output Gradient w.r.t output. NHWC Shape: [batch_size, output_h, output_w,
   * channels], DType: io_dtype.
   * @param grad_input Computed gradient w.r.t input. NHWC Shape: [batch_size, height, width,
   * channels], DType: io_dtype.
   * @param mask Mask tensor from forward pass. NHWC Shape: [batch_size, output_h, output_w,
   * channels], DType: int8. Represents max position relative to pool window.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void maxpool2d_bwd(void* backend_handle, const MaxPool2DStats& stats,
                             const void* grad_output, void* grad_input, const void* mask,
                             void* workspace, DTypeDesc type_desc) = 0;

  /**
   * @brief Forward pass for prepending a Class Token.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats ClassToken layer configuration.
   * @param input Input sequence. Shape: [batch_size, seq_len, embed_dim], DType: io_dtype.
   * @param token Class token tensor. Shape: [embed_dim], DType: param_dtype.
   * @param output Output sequence. Shape: [batch_size, seq_len + 1, embed_dim], DType: io_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void class_token_fwd(void* backend_handle, const ClassTokenStats& stats,
                               const void* input, const void* token, void* output, void* workspace,
                               DTypeDesc type_desc) = 0;

  /**
   * @brief Backward pass for a Class Token layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats ClassToken layer configuration.
   * @param grad_output Gradient w.r.t output sequence. Shape: [batch_size, seq_len + 1, embed_dim],
   * DType: io_dtype.
   * @param grad_input Computed gradient w.r.t input sequence. Shape: [batch_size, seq_len,
   * embed_dim], DType: io_dtype.
   * @param grad_token Gradient from previous steps.
   * @param grad_token_temp Accumulated token gradient output. Shape: [embed_dim], DType:
   * param_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void class_token_bwd(void* backend_handle, const ClassTokenStats& stats,
                               const void* grad_output, void* grad_input, void* grad_token,
                               void* workspace, DTypeDesc type_desc) = 0;

  /**
   * @brief Forward pass for a Dropout layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Dropout layer configuration.
   * @param input Input tensor. Shape: Total elements flattened, DType: io_dtype.
   * @param output Output tensor. Shape: Total elements flattened, DType: io_dtype.
   * @param mask Mask tensor indicating dropped elements. Shape: Total elements, DType: boolean.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void dropout_fwd(void* backend_handle, const DropoutStats& stats, const void* input,
                           void* output, bool* mask, void* workspace, DTypeDesc type_desc) = 0;

  /**
   * @brief Backward pass for a Dropout layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Dropout layer configuration.
   * @param grad_output Gradient w.r.t output. Shape: Total elements flattened, DType: io_dtype.
   * @param grad_input Computed gradient w.r.t input. Shape: Total elements flattened, DType:
   * io_dtype.
   * @param mask Mask tensor from forward pass. Shape: Total elements, DType: boolean.
   * @param scale Scaling factor applied to gradients.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void dropout_bwd(void* backend_handle, const DropoutStats& stats, const void* grad_output,
                           void* grad_input, const bool* mask, double scale, void* workspace,
                           DTypeDesc type_desc) = 0;

  /**
   * @brief Forward pass for an Embedding layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Embedding layer configuration.
   * @param input Input indices tensor. Shape: [num_indices], DType: io_dtype.
   * @param weight Embedding lookup table. Shape: [vocab_size, embed_dim], DType: param_dtype.
   * @param output Output embeddings. Shape: [num_indices, embed_dim], DType: io_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void embedding_fwd(void* backend_handle, const EmbeddingStats& stats, const void* input,
                             const void* weight, void* output, void* workspace,
                             DTypeDesc type_desc) = 0;

  /**
   * @brief Backward pass for an Embedding layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Embedding layer configuration.
   * @param grad_output Gradient w.r.t output embeddings. Shape: [num_indices, embed_dim], DType:
   * io_dtype.
   * @param input Input indices tensor. Shape: [num_indices], DType: io_dtype.
   * @param grad_weight Gradient from previous steps.
   * @param grad_weight_temp Accumulated weight gradient output. Shape: [vocab_size, embed_dim],
   * DType: param_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void embedding_bwd(void* backend_handle, const EmbeddingStats& stats,
                             const void* grad_output, const void* input, void* grad_weight,
                             void* workspace, DTypeDesc type_desc) = 0;

  /**
   * @brief Forward pass for a ReLU activation.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats ReLU layer configuration.
   * @param input Input tensor. Shape: Total elements flattened, DType: io_dtype.
   * @param output Output tensor. Shape: Total elements flattened, DType: io_dtype.
   * @param mask Mask tensor (bitmask or bool) for backward pass.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void relu_fwd(void* backend_handle, const ReLUStats& stats, const void* input,
                        void* output, bool* mask, void* workspace, DTypeDesc type_desc) = 0;

  /**
   * @brief Backward pass for a ReLU activation.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats ReLU layer configuration.
   * @param grad_output Gradient w.r.t output. Shape: Total elements flattened, DType: io_dtype.
   * @param grad_input Computed gradient w.r.t input. Shape: Total elements flattened, DType:
   * io_dtype.
   * @param mask Mask tensor from forward pass.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void relu_bwd(void* backend_handle, const ReLUStats& stats, const void* grad_output,
                        void* grad_input, const bool* mask, void* workspace,
                        DTypeDesc type_desc) = 0;

  /**
   * @brief Forward pass for a BatchNorm layer (training mode).
   * @param backend_handle Opaque handle to the backend context.
   * @param stats BatchNorm layer configuration.
   * @param input Input tensor. NHWC Shape: [batch_size, height, width, channels], DType: io_dtype.
   * @param gamma Scale parameter tensor. Shape: [1, 1, 1, channels], DType: param_dtype.
   * @param beta Bias parameter tensor. Shape: [1, 1, 1, channels], DType: param_dtype.
   * @param output Output tensor. NHWC Shape: [batch_size, height, width, channels], DType:
   * io_dtype.
   * @param prev_running_mean Previous running mean. Shape: [1, 1, 1, channels], DType:
   * compute_dtype.
   * @param prev_running_var Previous running variance. Shape: [1, 1, 1, channels], DType:
   * compute_dtype.
   * @param next_running_mean Updated running mean. Shape: [1, 1, 1, channels], DType:
   * compute_dtype.
   * @param next_running_var Updated running variance. Shape: [1, 1, 1, channels], DType:
   * compute_dtype.
   * @param batch_mean Computed batch mean (for backward pass). Shape: [1, 1, 1, channels], DType:
   * compute_dtype.
   * @param batch_invar Computed batch inverse variance (for backward pass). Shape: [1, 1, 1,
   * channels], DType: compute_dtype.
   * @param relu_mask Optional bitmask if fused with ReLU.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void batchnorm_fwd(void* backend_handle, const BatchNormStats& stats, const void* input,
                             const void* gamma, const void* beta, void* output,
                             void* prev_running_mean, void* prev_running_var,
                             void* next_running_mean, void* next_running_var, void* batch_mean,
                             void* batch_invar, void* relu_mask, void* workspace,
                             DTypeDesc type_desc) = 0;

  /**
   * @brief Forward pass for a BatchNorm layer (inference mode).
   * @param backend_handle Opaque handle to the backend context.
   * @param stats BatchNorm layer configuration.
   * @param input Input tensor. NHWC Shape: [batch_size, height, width, channels], DType: io_dtype.
   * @param gamma Scale parameter tensor. Shape: [1, 1, 1, channels], DType: param_dtype.
   * @param beta Bias parameter tensor. Shape: [1, 1, 1, channels], DType: param_dtype.
   * @param saved_mean Saved running mean. Shape: [1, 1, 1, channels], DType: compute_dtype.
   * @param saved_var Saved running variance. Shape: [1, 1, 1, channels], DType: compute_dtype.
   * @param output Output tensor. NHWC Shape: [batch_size, height, width, channels], DType:
   * io_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void batchnorm_infer(void* backend_handle, const BatchNormStats& stats, const void* input,
                               const void* gamma, const void* beta, const void* saved_mean,
                               const void* saved_var, void* output, void* workspace,
                               DTypeDesc type_desc) = 0;

  /**
   * @brief Backward pass for a BatchNorm layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats BatchNorm layer configuration.
   * @param grad_output Gradient w.r.t output. NHWC Shape: [batch_size, height, width, channels],
   * DType: io_dtype.
   * @param input Input tensor. NHWC Shape: [batch_size, height, width, channels], DType: io_dtype.
   * @param relu_mask Optional bitmask if fused with ReLU.
   * @param gamma Scale parameter tensor. Shape: [1, 1, 1, channels], DType: param_dtype.
   * @param grad_input Computed gradient w.r.t input. NHWC Shape: [batch_size, height, width,
   * channels], DType: io_dtype.
   * @param grad_gamma Gradient of gamma from previous steps.
   * @param grad_gamma_temp Accumulated gradient w.r.t gamma. Shape: [1, 1, 1, channels], DType:
   * param_dtype.
   * @param grad_beta Gradient of beta from previous steps.
   * @param grad_beta_temp Accumulated gradient w.r.t beta. Shape: [1, 1, 1, channels], DType:
   * param_dtype.
   * @param batch_mean Saved batch mean from forward pass. Shape: [1, 1, 1, channels], DType:
   * compute_dtype.
   * @param batch_invar Saved batch inverse variance from forward pass. Shape: [1, 1, 1, channels],
   * DType: compute_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void batchnorm_bwd(void* backend_handle, const BatchNormStats& stats,
                             const void* grad_output, const void* input, const void* relu_mask,
                             const void* gamma, void* grad_input, void* grad_gamma, void* grad_beta,
                             const void* batch_mean, const void* batch_invar, void* workspace,
                             DTypeDesc type_desc) = 0;

  /**
   * @brief Forward pass for a Conv2D layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Conv2D layer configuration.
   * @param input Input tensor. NHWC Shape: [batch_size, input_h, input_w, in_channels], DType:
   * io_dtype.
   * @param weight Weight tensor. NHWC Shape: [out_channels, kernel_h, kernel_w, in_channels],
   * DType: param_dtype.
   * @param bias Bias tensor. NHWC Shape: [1, 1, 1, out_channels], DType: param_dtype.
   * @param output Output tensor. NHWC Shape: [batch_size, output_h, output_w, out_channels], DType:
   * io_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void conv2d_fwd(void* backend_handle, const Conv2DStats& stats, const void* input,
                          const void* weight, const void* bias, void* output, void* workspace,
                          DTypeDesc type_desc) = 0;

  /**
   * @brief Computes data gradients for a Conv2D layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Conv2D layer configuration.
   * @param grad_output Gradient w.r.t output. NHWC Shape: [batch_size, output_h, output_w,
   * out_channels], DType: io_dtype.
   * @param weight Weight tensor. NHWC Shape: [out_channels, kernel_h, kernel_w, in_channels],
   * DType: param_dtype.
   * @param grad_input Computed gradient w.r.t input. NHWC Shape: [batch_size, input_h, input_w,
   * in_channels], DType: io_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void conv2d_dgrad(void* backend_handle, const Conv2DStats& stats, const void* grad_output,
                            const void* weight, void* grad_input, void* workspace,
                            DTypeDesc type_desc) = 0;

  /**
   * @brief Computes weight gradients for a Conv2D layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Conv2D layer configuration.
   * @param grad_output Gradient w.r.t output. NHWC Shape: [batch_size, output_h, output_w,
   * out_channels], DType: io_dtype.
   * @param input Input tensor. NHWC Shape: [batch_size, input_h, input_w, in_channels], DType:
   * io_dtype.
   * @param grad_weight Gradient from previous steps.
   * @param grad_weight_temp Accumulated weight gradient output. NHWC Shape: [out_channels,
   * kernel_h, kernel_w, in_channels], DType: param_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void conv2d_wgrad(void* backend_handle, const Conv2DStats& stats, const void* grad_output,
                            const void* input, void* grad_weight, void* workspace,
                            DTypeDesc type_desc) = 0;

  /**
   * @brief Computes bias gradients for a Conv2D layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Conv2D layer configuration.
   * @param grad_output Gradient w.r.t output. NHWC Shape: [batch_size, output_h, output_w,
   * out_channels], DType: io_dtype.
   * @param grad_bias Gradient from previous steps.
   * @param grad_bias_temp Accumulated bias gradient output. NHWC Shape: [1, 1, 1, out_channels],
   * DType: param_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void conv2d_bgrad(void* backend_handle, const Conv2DStats& stats, const void* grad_output,
                            void* grad_bias, void* workspace, DTypeDesc type_desc) = 0;

  /**
   * @brief Forward pass for a LayerNorm layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats LayerNorm layer configuration.
   * @param input Input tensor. Shape: [batch_size, channels], DType: io_dtype.
   * @param gamma Scale parameter tensor. Shape: [1, channels], DType: param_dtype.
   * @param beta Bias parameter tensor. Shape: [1, channels], DType: param_dtype.
   * @param output Output tensor. Shape: [batch_size, channels], DType: io_dtype.
   * @param mean Computed mean (for backward pass). Shape: [batch_size, 1], DType: compute_dtype.
   * @param inv_variance Computed inverse variance (for backward pass). Shape: [batch_size, 1],
   * DType: compute_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void layernorm_fwd(void* backend_handle, const LayerNormStats& stats, const void* input,
                             const void* gamma, const void* beta, void* output, void* mean,
                             void* inv_variance, void* workspace, DTypeDesc type_desc) = 0;

  /**
   * @brief Forward pass for a LayerNorm layer (inference mode).
   * @param backend_handle Opaque handle to the backend context.
   * @param stats LayerNorm layer configuration.
   * @param input Input tensor. Shape: [batch_size, channels], DType: io_dtype.
   * @param gamma Scale parameter tensor. Shape: [1, channels], DType: param_dtype.
   * @param beta Bias parameter tensor. Shape: [1, channels], DType: param_dtype.
   * @param output Output tensor. Shape: [batch_size, channels], DType: io_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void layernorm_infer(void* backend_handle, const LayerNormStats& stats, const void* input,
                               const void* gamma, const void* beta, void* output, void* workspace,
                               DTypeDesc type_desc) = 0;

  /**
   * @brief Backward pass for a LayerNorm layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats LayerNorm layer configuration.
   * @param grad_output Gradient w.r.t output. Shape: [batch_size, channels], DType: io_dtype.
   * @param input Input tensor. Shape: [batch_size, channels], DType: io_dtype.
   * @param gamma Scale parameter tensor. Shape: [1, channels], DType: param_dtype.
   * @param mean Saved mean from forward pass. Shape: [batch_size, 1], DType: compute_dtype.
   * @param inv_variance Saved inverse variance from forward pass. Shape: [batch_size, 1], DType:
   * compute_dtype.
   * @param grad_input Computed gradient w.r.t input. Shape: [batch_size, channels], DType:
   * io_dtype.
   * @param grad_gamma Gradient of gamma from previous steps.
   * @param grad_gamma_temp Accumulated gradient w.r.t gamma. Shape: [1, channels], DType:
   * param_dtype.
   * @param grad_beta Gradient of beta from previous steps.
   * @param grad_beta_temp Accumulated gradient w.r.t beta. Shape: [1, channels], DType:
   * param_dtype.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  virtual void layernorm_bwd(void* backend_handle, const LayerNormStats& stats,
                             const void* grad_output, const void* input, const void* gamma,
                             const void* mean, const void* inv_variance, void* grad_input,
                             void* grad_gamma, void* grad_beta, void* workspace,
                             DTypeDesc type_desc) = 0;

  // --- Legacy APIs ---

  virtual void legacy_dense_fwd(void* backend_handle, const void* input, const void* weight,
                                void* output, size_t batch_size, size_t in_features,
                                size_t out_features, DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_dense_wgrad(void* backend_handle, const void* input, const void* grad_output,
                                  void* grad_weight, size_t batch_size, size_t in_features,
                                  size_t out_features, DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_dense_dgrad(void* backend_handle, const void* grad_output, const void* weight,
                                  void* grad_input, size_t batch_size, size_t in_features,
                                  size_t out_features, DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_dense_bgrad(void* backend_handle, const void* grad_output, void* grad_bias,
                                  size_t batch_size, size_t out_features, DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_dense_add_bias(void* backend_handle, void* output, const void* bias,
                                     size_t batch_size, size_t out_features, DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_avgpool2d_fwd(void* backend_handle, const void* input, void* output,
                                    size_t batch_size, size_t channels, size_t input_h,
                                    size_t input_w, size_t output_h, size_t output_w, size_t pool_h,
                                    size_t pool_w, size_t stride_h, size_t stride_w, size_t pad_h,
                                    size_t pad_w, DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_avgpool2d_bwd(void* backend_handle, const void* grad_output, void* grad_input,
                                    size_t batch_size, size_t channels, size_t input_h,
                                    size_t input_w, size_t output_h, size_t output_w, size_t pool_h,
                                    size_t pool_w, size_t stride_h, size_t stride_w, size_t pad_h,
                                    size_t pad_w, DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_maxpool2d_fwd(void* backend_handle, const void* input, void* output,
                                    size_t batch_size, size_t channels, size_t input_h,
                                    size_t input_w, size_t output_h, size_t output_w, size_t pool_h,
                                    size_t pool_w, size_t stride_h, size_t stride_w, size_t pad_h,
                                    size_t pad_w, void* mask_indices, DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_maxpool2d_bwd(void* backend_handle, const void* grad_output, void* grad_input,
                                    size_t batch_size, size_t channels, size_t output_h,
                                    size_t output_w, const void* mask_indices,
                                    DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_batchnorm_fwd(void* backend_handle, const void* input, void* batch_mean,
                                    void* batch_inv_std, void* running_mean, void* running_var,
                                    const void* gamma, const void* beta, void* output, void* norm,
                                    size_t batch_size, size_t channels, size_t spatial_size,
                                    float momentum, float epsilon, bool affine,
                                    DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_batchnorm_infer(void* backend_handle, const void* input,
                                      const void* running_mean, const void* running_var,
                                      const void* gamma, const void* beta, void* output,
                                      size_t batch_size, size_t channels, size_t spatial_size,
                                      float epsilon, bool affine, DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_batchnorm_bwd(void* backend_handle, const void* grad_output,
                                    const void* norm_input, const void* inv_std, const void* gamma,
                                    void* d_gamma, void* d_beta, void* grad_input,
                                    size_t batch_size, size_t channels, size_t spatial_size,
                                    bool affine, DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_conv2d_fwd(void* backend_handle, const void* col_data,
                                 const void* weight_data, void* output_data, size_t output_size,
                                 size_t kernel_size, size_t out_channels, DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_conv2d_wgrad(void* backend_handle, const void* col_data,
                                   const void* gradient_data, void* grad_weight_data,
                                   size_t output_size, size_t kernel_size, size_t out_channels,
                                   DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_conv2d_dgrad(void* backend_handle, const void* gradient_data,
                                   const void* weight_data, void* col_grad_data, size_t output_size,
                                   size_t kernel_size, size_t out_channels, DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_conv2d_bgrad(void* backend_handle, const void* gradient_data,
                                   void* grad_bias_data, size_t batch_size, size_t output_h,
                                   size_t output_w, size_t out_channels, DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }

  virtual void legacy_conv2d_add_bias(void* backend_handle, void* output_data,
                                      const void* bias_data, size_t batch_size, size_t output_h,
                                      size_t output_w, size_t out_channels, DTypeDesc type_desc) {
    throw std::runtime_error("Not implemented");
  }
};

}  // namespace tunx