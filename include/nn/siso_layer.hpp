#pragma once

#include "nn/layer.hpp"

namespace tnn {
class SISOLayer : virtual public Layer {
public:
  SISOLayer() = default;

  void forward(const std::vector<ConstTensor> &inputs, const std::vector<Tensor> &outputs,
               size_t mb_id = 0) override;
  void backward(const std::vector<ConstTensor> &gradients, const std::vector<Tensor> &grad_inputs,
                size_t mb_id = 0) override;
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;

protected:
  virtual void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) = 0;
  virtual void backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                             size_t mb_id = 0) = 0;
  virtual Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const = 0;
};

}  // namespace tnn