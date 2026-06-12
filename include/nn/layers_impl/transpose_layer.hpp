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

namespace synet {

class TransposeLayerImpl : public SISOLayerImpl {
private:
  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> permute(const ConstTensor &input, const Tensor &output, size_t B, size_t L,
                                size_t H, size_t D, flowHandle_t handle) const;

  Tensor forward_impl(const ConstTensor &input, size_t mb_id = 0) override;
  Tensor backward_impl(const ConstTensor &grad_output, size_t mb_id = 0) override;

public:
  TransposeLayerImpl(const std::string &name = "transpose");

  static constexpr const char *TYPE_NAME = "transpose";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;

  static std::shared_ptr<TransposeLayerImpl> create_from_config(const LayerConfig &config);
};

class TransposeLayer : public LayerRef<TransposeLayerImpl> {
public:
  explicit TransposeLayer(const std::string &name = "transpose")
      : LayerRef(std::make_shared<TransposeLayerImpl>(name)) {}

  using LayerRef<TransposeLayerImpl>::LayerRef;
};

}  // namespace synet
