#include "nn/siso_layer.hpp"

namespace tnn {
void SISOLayer::forward(const std::vector<ConstTensor> &inputs, const std::vector<Tensor> &outputs,
                        size_t mb_id) {
  if (!initialized_) {
    std::cerr << "Warning: Layer " << name_ << " is not initialized. Call init() before forward."
              << std::endl;
    return;
  }
  if (inputs.empty() || outputs.empty()) {
    throw std::runtime_error("Layer " + name_ + " received empty IO tensors.");
  }
  if (inputs[0]->data_type() != io_dtype_) {
    throw std::runtime_error("Layer " + name_ +
                             " input tensor dtype does not match layer io_dtype.");
  }
  if (outputs[0]->data_type() != io_dtype_) {
    throw std::runtime_error("Layer " + name_ +
                             " output tensor dtype does not match layer io_dtype.");
  }
  ConstTensor current = inputs[0];
  Tensor device_input;
  if (inputs[0]->device() != this->device()) {
    device_input = this->get_act(inputs[0]->shape());
    inputs[0]->copy_to(device_input);
    current = device_input;
  }
  if (outputs[0]->device() != this->device()) {
    throw std::runtime_error("Layer " + name_ +
                             " output tensor device does not match layer device.");
  }
  forward_impl(current, outputs[0], mb_id);
#ifndef NDEBUG
  this->device().getFlow(this->flow_handle_)->synchronize();
#endif
}

void SISOLayer::backward(const std::vector<ConstTensor> &gradients,
                         const std::vector<Tensor> &grad_inputs, size_t mb_id) {
  if (!initialized_) {
    std::cerr << "Warning: Layer " << name_ << " is not initialized. Call init() before backward."
              << std::endl;
    return;
  }
  if (gradients.empty() || grad_inputs.empty()) {
    throw std::runtime_error("Layer " + name_ +
                             " received empty gradients or grad_inputs tensors.");
  }
  if (gradients[0]->data_type() != io_dtype_) {
    throw std::runtime_error("Layer " + name_ +
                             " grad_output tensor dtype does not match layer io_dtype.");
  }
  if (grad_inputs[0]->data_type() != io_dtype_) {
    throw std::runtime_error("Layer " + name_ +
                             " grad_input tensor dtype does not match layer io_dtype.");
  }
  ConstTensor current_gradient = gradients[0];
  Tensor device_gradient;
  if (gradients[0]->device() != this->device()) {
    device_gradient = this->get_workspace(gradients[0]->shape(), gradients[0]->data_type());
    gradients[0]->copy_to(device_gradient);
    current_gradient = device_gradient;
  }
  if (grad_inputs[0]->device() != this->device()) {
    throw std::runtime_error("Layer " + name_ +
                             " grad_input tensor device does not match layer device.");
  }
  backward_impl(current_gradient, grad_inputs[0], mb_id);
#ifndef NDEBUG
  this->device().getFlow(this->flow_handle_)->synchronize();
#endif
  clear_cache(mb_id);
}

Vec<Vec<size_t>> SISOLayer::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (input_shapes.size() != 1) {
    throw std::runtime_error("Only single input supported in output_shape for SISO layers.");
  }
  return {compute_output_shape(input_shapes[0])};
}

}  // namespace tnn