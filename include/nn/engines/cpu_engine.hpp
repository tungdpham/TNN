#pragma once

#include "nn/engines/iengine.hpp"

namespace tunx {

class CPUEngine : public IEngine {
public:
  CPUEngine() = default;
  ~CPUEngine() override = default;

  void* create_backend_handle() override;

  WorkspaceReq query_dense_graph(void* backend_handle, const DenseStats& stats,
                                 DTypeDesc type_desc) override;
  WorkspaceReq query_avgpool_graph(void* backend_handle, const AvgPool2DStats& stats,
                                   DTypeDesc type_desc) override;
  WorkspaceReq query_maxpool2d_graph(void* backend_handle, const MaxPool2DStats& stats,
                                     DTypeDesc type_desc) override;
  WorkspaceReq query_class_token_graph(void* backend_handle, const ClassTokenStats& stats,
                                       DTypeDesc type_desc) override;
  WorkspaceReq query_dropout_graph(void* backend_handle, const DropoutStats& stats,
                                   DTypeDesc type_desc) override;
  WorkspaceReq query_embedding_graph(void* backend_handle, const EmbeddingStats& stats,
                                     DTypeDesc type_desc) override;
  WorkspaceReq query_relu_graph(void* backend_handle, const ReLUStats& stats,
                                DTypeDesc type_desc) override;
  WorkspaceReq query_batchnorm_graph(void* backend_handle, const BatchNormStats& stats,
                                     DTypeDesc type_desc) override;
  WorkspaceReq query_conv2d_graph(void* backend_handle, const Conv2DStats& stats,
                                  DTypeDesc type_desc) override;
  WorkspaceReq query_layernorm_graph(void* backend_handle, const LayerNormStats& stats,
                                     DTypeDesc type_desc) override;

  void dense_fwd(void* backend_handle, const DenseStats& stats, const void* input,
                 const void* weight, const void* bias, void* output, void* workspace,
                 DTypeDesc type_desc) override;

  void dense_wgrad(void* backend_handle, const DenseStats& stats, const void* grad_output,
                   const void* input, void* grad_weight, void* workspace,
                   DTypeDesc type_desc) override;

  void dense_dgrad(void* backend_handle, const DenseStats& stats, const void* grad_output,
                   const void* weight, void* grad_input, void* workspace,
                   DTypeDesc type_desc) override;

  void dense_bgrad(void* backend_handle, const DenseStats& stats, const void* grad_output,
                   void* grad_bias, void* workspace, DTypeDesc type_desc) override;

  void avgpool_fwd(void* backend_handle, const AvgPool2DStats& stats, const void* input,
                   void* output, void* workspace, DTypeDesc type_desc) override;

  void avgpool_bwd(void* backend_handle, const AvgPool2DStats& stats, const void* grad_output,
                   void* grad_input, void* workspace, DTypeDesc type_desc) override;

  void maxpool2d_fwd(void* backend_handle, const MaxPool2DStats& stats, const void* input,
                     void* output, void* mask, void* workspace, DTypeDesc type_desc) override;

  void maxpool2d_infer(void* backend_handle, const MaxPool2DStats& stats, const void* input,
                       void* output, void* workspace, DTypeDesc type_desc) override;

  void maxpool2d_bwd(void* backend_handle, const MaxPool2DStats& stats, const void* grad_output,
                     void* grad_input, const void* mask, void* workspace,
                     DTypeDesc type_desc) override;

  void class_token_fwd(void* backend_handle, const ClassTokenStats& stats, const void* input,
                       const void* token, void* output, void* workspace,
                       DTypeDesc type_desc) override;

  void class_token_bwd(void* backend_handle, const ClassTokenStats& stats, const void* grad_output,
                       void* grad_input, void* grad_token, void* workspace,
                       DTypeDesc type_desc) override;

  void dropout_fwd(void* backend_handle, const DropoutStats& stats, const void* input, void* output,
                   bool* mask, void* workspace, DTypeDesc type_desc) override;

  void dropout_bwd(void* backend_handle, const DropoutStats& stats, const void* grad_output,
                   void* grad_input, const bool* mask, double scale, void* workspace,
                   DTypeDesc type_desc) override;

  void relu_fwd(void* backend_handle, const ReLUStats& stats, const void* input, void* output,
                bool* mask, void* workspace, DTypeDesc type_desc) override;

  void relu_bwd(void* backend_handle, const ReLUStats& stats, const void* grad_output,
                void* grad_input, const bool* mask, void* workspace, DTypeDesc type_desc) override;

  void embedding_fwd(void* backend_handle, const EmbeddingStats& stats, const void* input,
                     const void* weight, void* output, void* workspace,
                     DTypeDesc type_desc) override;

  void embedding_bwd(void* backend_handle, const EmbeddingStats& stats, const void* grad_output,
                     const void* input, void* grad_weight, void* workspace,
                     DTypeDesc type_desc) override;

  void batchnorm_fwd(void* backend_handle, const BatchNormStats& stats, const void* input,
                     const void* gamma, const void* beta, void* output, void* prev_running_mean,
                     void* prev_running_var, void* next_running_mean, void* next_running_var,
                     void* batch_mean, void* batch_invar, void* relu_mask, void* workspace,
                     DTypeDesc type_desc) override;

  void batchnorm_infer(void* backend_handle, const BatchNormStats& stats, const void* input,
                       const void* gamma, const void* beta, const void* saved_mean,
                       const void* saved_var, void* output, void* workspace,
                       DTypeDesc type_desc) override;

  void batchnorm_bwd(void* backend_handle, const BatchNormStats& stats, const void* grad_output,
                     const void* input, const void* relu_mask, const void* gamma, void* grad_input,
                     void* grad_gamma, void* grad_beta, const void* batch_mean,
                     const void* batch_invar, void* workspace, DTypeDesc type_desc) override;

  void conv2d_fwd(void* backend_handle, const Conv2DStats& stats, const void* input,
                  const void* weight, const void* bias, void* output, void* workspace,
                  DTypeDesc type_desc) override;

  void conv2d_dgrad(void* backend_handle, const Conv2DStats& stats, const void* grad_output,
                    const void* weight, void* grad_input, void* workspace,
                    DTypeDesc type_desc) override;

  void conv2d_wgrad(void* backend_handle, const Conv2DStats& stats, const void* grad_output,
                    const void* input, void* grad_weight, void* workspace,
                    DTypeDesc type_desc) override;

  void conv2d_bgrad(void* backend_handle, const Conv2DStats& stats, const void* grad_output,
                    void* grad_bias, void* workspace, DTypeDesc type_desc) override;

  void layernorm_fwd(void* backend_handle, const LayerNormStats& stats, const void* input,
                     const void* gamma, const void* beta, void* output, void* mean,
                     void* inv_variance, void* workspace, DTypeDesc type_desc) override;

  void layernorm_infer(void* backend_handle, const LayerNormStats& stats, const void* input,
                       const void* gamma, const void* beta, void* output, void* workspace,
                       DTypeDesc type_desc) override;

  void layernorm_bwd(void* backend_handle, const LayerNormStats& stats, const void* grad_output,
                     const void* input, const void* gamma, const void* mean,
                     const void* inv_variance, void* grad_input, void* grad_gamma, void* grad_beta,
                     void* workspace, DTypeDesc type_desc) override;

  // --- Legacy APIs ---

  void legacy_dense_fwd(void* backend_handle, const void* input, const void* weight, void* output,
                        size_t batch_size, size_t in_features, size_t out_features,
                        DTypeDesc type_desc) override;

  void legacy_dense_wgrad(void* backend_handle, const void* input, const void* grad_output,
                          void* grad_weight, size_t batch_size, size_t in_features,
                          size_t out_features, DTypeDesc type_desc) override;

  void legacy_dense_dgrad(void* backend_handle, const void* grad_output, const void* weight,
                          void* grad_input, size_t batch_size, size_t in_features,
                          size_t out_features, DTypeDesc type_desc) override;

  void legacy_dense_bgrad(void* backend_handle, const void* grad_output, void* grad_bias,
                          size_t batch_size, size_t out_features, DTypeDesc type_desc) override;

  void legacy_dense_add_bias(void* backend_handle, void* output, const void* bias,
                             size_t batch_size, size_t out_features, DTypeDesc type_desc) override;

  void legacy_avgpool2d_fwd(void* backend_handle, const void* input, void* output,
                            size_t batch_size, size_t channels, size_t input_h, size_t input_w,
                            size_t output_h, size_t output_w, size_t pool_h, size_t pool_w,
                            size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                            DTypeDesc type_desc) override;

  void legacy_avgpool2d_bwd(void* backend_handle, const void* grad_output, void* grad_input,
                            size_t batch_size, size_t channels, size_t input_h, size_t input_w,
                            size_t output_h, size_t output_w, size_t pool_h, size_t pool_w,
                            size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                            DTypeDesc type_desc) override;

  void legacy_maxpool2d_fwd(void* backend_handle, const void* input, void* output,
                            size_t batch_size, size_t channels, size_t input_h, size_t input_w,
                            size_t output_h, size_t output_w, size_t pool_h, size_t pool_w,
                            size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                            void* mask_indices, DTypeDesc type_desc) override;

  void legacy_maxpool2d_bwd(void* backend_handle, const void* grad_output, void* grad_input,
                            size_t batch_size, size_t channels, size_t output_h, size_t output_w,
                            const void* mask_indices, DTypeDesc type_desc) override;

  void legacy_batchnorm_fwd(void* backend_handle, const void* input, void* batch_mean,
                            void* batch_inv_std, void* running_mean, void* running_var,
                            const void* gamma, const void* beta, void* output, void* norm,
                            size_t batch_size, size_t channels, size_t spatial_size, float momentum,
                            float epsilon, bool affine, DTypeDesc type_desc) override;

  void legacy_batchnorm_infer(void* backend_handle, const void* input, const void* running_mean,
                              const void* running_var, const void* gamma, const void* beta,
                              void* output, size_t batch_size, size_t channels, size_t spatial_size,
                              float epsilon, bool affine, DTypeDesc type_desc) override;

  void legacy_batchnorm_bwd(void* backend_handle, const void* grad_output, const void* norm_input,
                            const void* inv_std, const void* gamma, void* d_gamma, void* d_beta,
                            void* grad_input, size_t batch_size, size_t channels,
                            size_t spatial_size, bool affine, DTypeDesc type_desc) override;

  void legacy_conv2d_fwd(void* backend_handle, const void* col_data, const void* weight_data,
                         void* output_data, size_t output_size, size_t kernel_size,
                         size_t out_channels, DTypeDesc type_desc) override;

  void legacy_conv2d_wgrad(void* backend_handle, const void* col_data, const void* gradient_data,
                           void* grad_weight_data, size_t output_size, size_t kernel_size,
                           size_t out_channels, DTypeDesc type_desc) override;

  void legacy_conv2d_dgrad(void* backend_handle, const void* gradient_data, const void* weight_data,
                           void* col_grad_data, size_t output_size, size_t kernel_size,
                           size_t out_channels, DTypeDesc type_desc) override;

  void legacy_conv2d_bgrad(void* backend_handle, const void* gradient_data, void* grad_bias_data,
                           size_t batch_size, size_t output_h, size_t output_w, size_t out_channels,
                           DTypeDesc type_desc) override;

  void legacy_conv2d_add_bias(void* backend_handle, void* output_data, const void* bias_data,
                              size_t batch_size, size_t output_h, size_t output_w,
                              size_t out_channels, DTypeDesc type_desc) override;
};

}  // namespace tunx