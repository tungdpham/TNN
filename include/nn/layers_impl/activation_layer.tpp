#include "activation_layer.hpp"

#include <stdexcept>
#include <cassert>

namespace tnn {

template <typename T>
ActivationLayer<T>::ActivationLayer(std::unique_ptr<ActivationFunction<T>> activation,
                                    const std::string &name)
    : StatelessLayer<T>(name), activation_(std::move(activation)) {
  if (!activation_) {
    throw std::invalid_argument("Activation function cannot be null");
  }
}

template <typename T>
Tensor<T> ActivationLayer<T>::forward(const Tensor<T> &input, int micro_batch_id) {
  micro_batch_inputs_[micro_batch_id] = input.clone();
  Tensor<T> output = input.clone();
  activation_->apply(output);
  return output;
}

template <typename T>
Tensor<T> ActivationLayer<T>::backward(const Tensor<T> &grad_output,
                                       int micro_batch_id) {
  auto it = micro_batch_inputs_.find(micro_batch_id);
  assert(it != micro_batch_inputs_.end() && "No stored input for given micro_batch_id");
  const Tensor<T> &last_input = it->second;
  Tensor<T> grad = activation_->compute_gradient(last_input, &grad_output);
  return grad;
}


template <typename T>
std::string ActivationLayer<T>::type() const {
  return "activation";
}


template <typename T>
LayerConfig ActivationLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["activation"] = activation_->name();
  return config;
}

template <typename T>
std::unique_ptr<Layer<T>> ActivationLayer<T>::clone() const {
  return std::make_unique<ActivationLayer<T>>(activation_->clone(),
                                              this->name_);
}

template <typename T>
std::vector<size_t>
ActivationLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape; // Activation doesn't change shape
}

} // namespace tnn
