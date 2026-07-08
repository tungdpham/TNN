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

namespace tunx {

class LegacyDenseLayerImpl : public SISOLayerImpl {
private:
  size_t input_features_;
  size_t output_features_;
  bool use_bias_;

  Tensor weights_;
  Tensor bias_;
  Tensor grad_weights_;
  Tensor grad_bias_;

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  LegacyDenseLayerImpl(size_t input_features, size_t output_features, bool use_bias = true,
                       const std::string &name = "legacy_dense");

  static constexpr const char *TYPE_NAME = "legacy_dense";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;

  Vec<ParamDescriptor> param_descriptors() override {
    Vec<ParamDescriptor> descriptors;
    auto weight_desc = ParamDescriptor{
        param_dtype_,
        {output_features_, input_features_},
        &weights_,
        &grad_weights_,
    };
    descriptors.push_back(weight_desc);
    if (use_bias_) {
      auto bias_desc = ParamDescriptor{
          param_dtype_,
          {output_features_},
          &bias_,
          &grad_bias_,
      };
      descriptors.push_back(bias_desc);
    }
    return descriptors;
  }

  static std::shared_ptr<LegacyDenseLayerImpl> create_from_config(const LayerConfig &config);
};

class LegacyDenseLayer : public LayerRef<LegacyDenseLayerImpl> {
public:
  LegacyDenseLayer(size_t input_features, size_t output_features, bool use_bias = true,
                   const std::string &name = "legacy_dense")
      : LayerRef(std::make_shared<LegacyDenseLayerImpl>(input_features, output_features, use_bias,
                                                        name)) {}

  using LayerRef<LegacyDenseLayerImpl>::LayerRef;
};

}  // namespace tunx
