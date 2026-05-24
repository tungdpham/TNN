/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "device/task.hpp"
#include "nn/layer.hpp"
#include "nn/layers_impl/common/n_ary.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

// Base class for N-ary element-wise operations
// TODO: dependent caching. Probably should move impl into individual layers for better control
class NAryOpLayer : public virtual Layer {
public:
  explicit NAryOpLayer(NAryOp op_type, const std::string &name = "")
      : Layer(name),
        op_type_(op_type) {}

  Vec<Tensor> forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id = 0) override;
  Vec<Tensor> backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id = 0) override;

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;
  LayerConfig get_config() const override;
  Vec<ParamDescriptor> param_descriptors() override { return {}; }
  std::string type() const override = 0;

protected:
  NAryOp op_type_;

private:
  template <typename Compute_T>
  std::unique_ptr<Task> compute_nary_forward_impl(const Vec<ConstTensor> &inputs,
                                                  const Tensor &output, const Vec<size_t> &shape,
                                                  flowHandle_t handle);
  template <typename Compute_T>
  std::unique_ptr<Task> compute_nary_backward_impl(const ConstTensor &grad_output,
                                                   const Vec<Tensor> &grad_inputs,
                                                   const Vec<ConstTensor> &fwd_inputs,
                                                   const Vec<size_t> &shape, flowHandle_t handle);
};

class AddLayer : public NAryOpLayer {
public:
  static constexpr const char *TYPE_NAME = "add";

  explicit AddLayer(const std::string &name = "add")
      : NAryOpLayer(NAryOp::ADD, name) {}

  std::string type() const override { return TYPE_NAME; }
  static std::unique_ptr<AddLayer> create_from_config(const LayerConfig &config);
};

class SubLayer : public NAryOpLayer {
public:
  static constexpr const char *TYPE_NAME = "sub";

  explicit SubLayer(const std::string &name = "sub")
      : NAryOpLayer(NAryOp::SUB, name) {}

  std::string type() const override { return TYPE_NAME; }
  static std::unique_ptr<SubLayer> create_from_config(const LayerConfig &config);
};

class MulLayer : public NAryOpLayer {
public:
  static constexpr const char *TYPE_NAME = "mul";

  explicit MulLayer(const std::string &name = "mul")
      : NAryOpLayer(NAryOp::MUL, name) {}

  std::string type() const override { return TYPE_NAME; }
  static std::unique_ptr<MulLayer> create_from_config(const LayerConfig &config);
};

class DivLayer : public NAryOpLayer {
public:
  static constexpr const char *TYPE_NAME = "div";

  explicit DivLayer(const std::string &name = "div")
      : NAryOpLayer(NAryOp::DIV, name) {}

  std::string type() const override { return TYPE_NAME; }
  static std::unique_ptr<DivLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
