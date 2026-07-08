#pragma once

#include <any>
#include <unordered_map>

#include "nn/engines/iengine.hpp"
#include "nn/stats/stats.hpp"

namespace tunx {

class CuDNNEngine : public IEngine {
public:
  CuDNNEngine() = default;
  ~CuDNNEngine() override = default;
  CuDNNEngine(const CuDNNEngine& other) = delete;
  CuDNNEngine& operator=(const CuDNNEngine& other) = delete;
  CuDNNEngine(CuDNNEngine&& other) = delete;
  CuDNNEngine& operator=(CuDNNEngine&& other) = delete;

  void* create_backend_handle() override;

  /**
   * @brief Queries the workspace memory requirement for Dense graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Dense layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  WorkspaceReq query_dense_graph(void* backend_handle, const DenseStats& stats,
                                 DTypeDesc type_desc) override;

  /**
   * @brief Queries the workspace memory requirement for AvgPool graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats AvgPool layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  WorkspaceReq query_avgpool_graph(void* backend_handle, const AvgPool2DStats& stats,
                                   DTypeDesc type_desc) override;

  /**
   * @brief Queries the workspace memory requirement for MaxPool graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats MaxPool layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  WorkspaceReq query_maxpool2d_graph(void* backend_handle, const MaxPool2DStats& stats,
                                     DTypeDesc type_desc) override;

  /**
   * @brief Queries the workspace memory requirement for ClassToken graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats ClassToken layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  WorkspaceReq query_class_token_graph(void* backend_handle, const ClassTokenStats& stats,
                                       DTypeDesc type_desc) override;

  /**
   * @brief Queries the workspace memory requirement for Dropout graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Dropout layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  WorkspaceReq query_dropout_graph(void* backend_handle, const DropoutStats& stats,
                                   DTypeDesc type_desc) override;

  /**
   * @brief Queries the workspace memory requirement for Embedding graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Embedding layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  WorkspaceReq query_embedding_graph(void* backend_handle, const EmbeddingStats& stats,
                                     DTypeDesc type_desc) override;

  /**
   * @brief Queries the workspace memory requirement for ReLU graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats ReLU layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  WorkspaceReq query_relu_graph(void* backend_handle, const ReLUStats& stats,
                                DTypeDesc type_desc) override;

  /**
   * @brief Queries the workspace memory requirement for BatchNorm graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats BatchNorm layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward, backward, and inference workspace size in bytes.
   */
  WorkspaceReq query_batchnorm_graph(void* backend_handle, const BatchNormStats& stats,
                                     DTypeDesc type_desc) override;

  /**
   * @brief Queries the workspace memory requirement for Conv2D graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats Conv2D layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  WorkspaceReq query_conv2d_graph(void* backend_handle, const Conv2DStats& stats,
                                  DTypeDesc type_desc) override;

  /**
   * @brief Queries the workspace memory requirement for LayerNorm graphs.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats LayerNorm layer configuration.
   * @param type_desc Data type descriptors.
   * @return WorkspaceReq specifying forward and backward workspace size in bytes.
   */
  WorkspaceReq query_layernorm_graph(void* backend_handle, const LayerNormStats& stats,
                                     DTypeDesc type_desc) override;

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
  void dense_fwd(void* backend_handle, const DenseStats& stats, const void* input,
                 const void* weight, const void* bias, void* output, void* workspace,
                 DTypeDesc type_desc) override;

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
  void dense_wgrad(void* backend_handle, const DenseStats& stats, const void* grad_output,
                   const void* input, void* grad_weight, void* workspace,
                   DTypeDesc type_desc) override;

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
  void dense_dgrad(void* backend_handle, const DenseStats& stats, const void* grad_output,
                   const void* weight, void* grad_input, void* workspace,
                   DTypeDesc type_desc) override;

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
  void dense_bgrad(void* backend_handle, const DenseStats& stats, const void* grad_output,
                   void* grad_bias, void* workspace, DTypeDesc type_desc) override;

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
  void avgpool_fwd(void* backend_handle, const AvgPool2DStats& stats, const void* input,
                   void* output, void* workspace, DTypeDesc type_desc) override;

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
  void avgpool_bwd(void* backend_handle, const AvgPool2DStats& stats, const void* grad_output,
                   void* grad_input, void* workspace, DTypeDesc type_desc) override;

  /**
   * @brief Forward pass for a MaxPool2D layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats MaxPool layer configuration.
   * @param input Input tensor. NHWC Shape: [batch_size, height, width, channels], DType: io_dtype.
   * @param output Output tensor. NHWC Shape: [batch_size, output_h, output_w, channels], DType:
   * io_dtype.
   * @param mask Mask tensor (index) output. NHWC Shape: [batch_size, output_h, output_w, channels],
   * DType: int8.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  void maxpool2d_fwd(void* backend_handle, const MaxPool2DStats& stats, const void* input,
                     void* output, void* mask, void* workspace, DTypeDesc type_desc) override;

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
  void maxpool2d_infer(void* backend_handle, const MaxPool2DStats& stats, const void* input,
                       void* output, void* workspace, DTypeDesc type_desc) override;

  /**
   * @brief Backward pass for a MaxPool2D layer.
   * @param backend_handle Opaque handle to the backend context.
   * @param stats MaxPool layer configuration.
   * @param grad_output Gradient w.r.t output. NHWC Shape: [batch_size, output_h, output_w,
   * channels], DType: io_dtype.
   * @param grad_input Computed gradient w.r.t input. NHWC Shape: [batch_size, height, width,
   * channels], DType: io_dtype.
   * @param mask Mask tensor from forward pass. NHWC Shape: [batch_size, output_h, output_w,
   * channels], DType: int8.
   * @param workspace Workspace buffer.
   * @param type_desc Data type descriptors.
   */
  void maxpool2d_bwd(void* backend_handle, const MaxPool2DStats& stats, const void* grad_output,
                     void* grad_input, const void* mask, void* workspace,
                     DTypeDesc type_desc) override;

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
  void class_token_fwd(void* backend_handle, const ClassTokenStats& stats, const void* input,
                       const void* token, void* output, void* workspace,
                       DTypeDesc type_desc) override;

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
  void class_token_bwd(void* backend_handle, const ClassTokenStats& stats, const void* grad_output,
                       void* grad_input, void* grad_token, void* workspace,
                       DTypeDesc type_desc) override;

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
  void dropout_fwd(void* backend_handle, const DropoutStats& stats, const void* input, void* output,
                   bool* mask, void* workspace, DTypeDesc type_desc) override;

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
  void dropout_bwd(void* backend_handle, const DropoutStats& stats, const void* grad_output,
                   void* grad_input, const bool* mask, double scale, void* workspace,
                   DTypeDesc type_desc) override;

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
  void relu_fwd(void* backend_handle, const ReLUStats& stats, const void* input, void* output,
                bool* mask, void* workspace, DTypeDesc type_desc) override;

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
  void relu_bwd(void* backend_handle, const ReLUStats& stats, const void* grad_output,
                void* grad_input, const bool* mask, void* workspace, DTypeDesc type_desc) override;

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
  void embedding_fwd(void* backend_handle, const EmbeddingStats& stats, const void* input,
                     const void* weight, void* output, void* workspace,
                     DTypeDesc type_desc) override;

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
  void embedding_bwd(void* backend_handle, const EmbeddingStats& stats, const void* grad_output,
                     const void* input, void* grad_weight, void* workspace,
                     DTypeDesc type_desc) override;

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
  void batchnorm_fwd(void* backend_handle, const BatchNormStats& stats, const void* input,
                     const void* gamma, const void* beta, void* output, void* prev_running_mean,
                     void* prev_running_var, void* next_running_mean, void* next_running_var,
                     void* batch_mean, void* batch_invar, void* relu_mask, void* workspace,
                     DTypeDesc type_desc) override;

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
  void batchnorm_infer(void* backend_handle, const BatchNormStats& stats, const void* input,
                       const void* gamma, const void* beta, const void* saved_mean,
                       const void* saved_var, void* output, void* workspace,
                       DTypeDesc type_desc) override;

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
  void batchnorm_bwd(void* backend_handle, const BatchNormStats& stats, const void* grad_output,
                     const void* input, const void* relu_mask, const void* gamma, void* grad_input,
                     void* grad_gamma, void* grad_beta, const void* batch_mean,
                     const void* batch_invar, void* workspace, DTypeDesc type_desc) override;

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
  void conv2d_fwd(void* backend_handle, const Conv2DStats& stats, const void* input,
                  const void* weight, const void* bias, void* output, void* workspace,
                  DTypeDesc type_desc) override;

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
  void conv2d_dgrad(void* backend_handle, const Conv2DStats& stats, const void* grad_output,
                    const void* weight, void* grad_input, void* workspace,
                    DTypeDesc type_desc) override;

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
  void conv2d_wgrad(void* backend_handle, const Conv2DStats& stats, const void* grad_output,
                    const void* input, void* grad_weight, void* workspace,
                    DTypeDesc type_desc) override;

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
  void conv2d_bgrad(void* backend_handle, const Conv2DStats& stats, const void* grad_output,
                    void* grad_bias, void* workspace, DTypeDesc type_desc) override;

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
  void layernorm_fwd(void* backend_handle, const LayerNormStats& stats, const void* input,
                     const void* gamma, const void* beta, void* output, void* mean,
                     void* inv_variance, void* workspace, DTypeDesc type_desc) override;

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
  void layernorm_infer(void* backend_handle, const LayerNormStats& stats, const void* input,
                       const void* gamma, const void* beta, void* output, void* workspace,
                       DTypeDesc type_desc) override;

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
  void layernorm_bwd(void* backend_handle, const LayerNormStats& stats, const void* grad_output,
                     const void* input, const void* gamma, const void* mean,
                     const void* inv_variance, void* grad_input, void* grad_gamma, void* grad_beta,
                     void* workspace, DTypeDesc type_desc) override;

private:
  std::unordered_map<GraphCacheKey, std::any> graph_cache_;
};

}  // namespace tunx