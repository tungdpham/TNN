/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

class LayerNormLayerImpl : public SISOLayerImpl {
private:
  size_t normalized_shape_;  // Size of C (channels)
  float epsilon_;
  bool affine_;  // Whether to use learnable affine parameters

  Tensor gamma_;
  Tensor beta_;
  Tensor grad_gamma_;
  Tensor grad_beta_;

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  explicit LayerNormLayerImpl(size_t normalized_shape, float epsilon = 1e-5f, bool affine = true,
                              const std::string &name = "layer_norm");

  ~LayerNormLayerImpl();

  static constexpr const char *TYPE_NAME = "layer_norm";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override {
    return input_shape;
  }

  Vec<ParamDescriptor> param_descriptors() override {
    Vec<ParamDescriptor> descriptors;
    if (affine_) {
      auto gamma_desc = ParamDescriptor{
          param_dtype_,
          {normalized_shape_},
          &gamma_,
          &grad_gamma_,
      };
      descriptors.push_back(gamma_desc);
      auto beta_desc = ParamDescriptor{
          param_dtype_,
          {normalized_shape_},
          &beta_,
          &grad_beta_,
      };
      descriptors.push_back(beta_desc);
    }
    return descriptors;
  }

  static std::shared_ptr<LayerNormLayerImpl> create_from_config(const LayerConfig &config);
};

class LayerNormLayer : public LayerRef<LayerNormLayerImpl> {
public:
  explicit LayerNormLayer(size_t normalized_shape, float epsilon = 1e-5f, bool affine = true,
                          const std::string &name = "layer_norm")
      : LayerRef(std::make_shared<LayerNormLayerImpl>(normalized_shape, epsilon, affine, name)) {}

  using LayerRef<LayerNormLayerImpl>::LayerRef;
};

}  // namespace tunx
