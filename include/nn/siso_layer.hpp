#pragma once

#include "nn/layer.hpp"

namespace tnn {
class SISOLayer : virtual public Layer {
public:
  SISOLayer() = default;
  void forward_impl(const Vec<ConstTensor> &inputs, const Vec<Tensor> &outputs,
                    size_t mb_id) override {
    if (inputs.size() != 1 || outputs.size() != 1) {
      throw std::runtime_error("SISOLayer only supports single input and single output");
    }
    forward_impl(inputs[0], outputs[0], mb_id);
  }
  void backward_impl(const Vec<ConstTensor> &grad_outputs, const Vec<Tensor> &grad_inputs,
                     size_t mb_id) override {
    if (grad_outputs.size() != 1 || grad_inputs.size() != 1) {
      throw std::runtime_error("SISOLayer only supports single grad output and single grad input");
    }
    backward_impl(grad_outputs[0], grad_inputs[0], mb_id);
  }

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;

protected:
  virtual void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) = 0;
  virtual void backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                             size_t mb_id = 0) = 0;
  virtual Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const = 0;
};

}  // namespace tnn