/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/blocks_impl/residual_block.hpp"

#include <alloca.h>

#include <cstddef>

#include "nn/activations.hpp"
#include "nn/layer.hpp"
#include "nn/layers.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"

namespace synet {

ResidualBlockImpl::ResidualBlockImpl(Vec<Layer> main_path, Vec<Layer> shortcut_path,
                                     const std::string &final_activation, const std::string &name)
    : activation_type_(final_activation) {
  if (main_path.empty()) {
    throw std::runtime_error("Main path of ResidualBlock cannot be empty.");
  }
  main_path_ = Sequential(std::move(main_path), name + "_main_path");
  if (!shortcut_path.empty()) {
    shortcut_path_ = Sequential(std::move(shortcut_path), name + "_shortcut_path");
  }
  this->name_ = name;

  if (final_activation != "none" && final_activation != "linear") {
    auto factory = ActivationFactory();
    factory.register_defaults();
    final_activation_ = factory.create(final_activation);
  }
}

ResidualBlockImpl::ResidualBlockImpl(Sequential main_path, Sequential shortcut_path,
                                     const std::string &final_activation, const std::string &name) {
  if (!main_path) {
    throw std::runtime_error("Main path of ResidualBlock cannot be null.");
  }
  this->main_path_ = std::move(main_path);
  if (shortcut_path) {
    this->shortcut_path_ = std::move(shortcut_path);
  }
  this->activation_type_ = final_activation;
  this->name_ = name;

  if (final_activation != "none" && final_activation != "linear") {
    auto factory = ActivationFactory();
    factory.register_defaults();
    final_activation_ = factory.create(final_activation);
  }
}

Vec<Tensor> ResidualBlockImpl::forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id) {
  // Forward through main path
  Vec<Tensor> main_outputs = main_path_->forward(inputs, mb_id);

  // Forward through shortcut path
  Vec<ConstTensor> shortcut_outputs = inputs;
  if (shortcut_path_) {
    Vec<Tensor> shortcut_outputs_vec = shortcut_path_->forward(inputs, mb_id);
    for (size_t i = 0; i < shortcut_outputs_vec.size(); ++i) {
      shortcut_outputs[i] = shortcut_outputs_vec[i];
    }
  }

  Vec<Tensor> outputs = main_outputs;  // reuse main path outputs for final output to save memory

  // Add outputs and apply final activation
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (final_activation_) {
      std::string pre_act_key = "pre_activation_" + std::to_string(i);
      Tensor pre_act = get_tensor(main_outputs[i]->shape(), io_dtype_);
      DISPATCH_IO_DTYPE(ops::add, main_outputs[i]->data_ptr(), shortcut_outputs[i]->data_ptr(),
                        pre_act->data_ptr(), outputs[i]->size());
      set_mutable_cache(mb_id, pre_act_key, pre_act);
      final_activation_->apply(pre_act, outputs[i]);
    } else {
      DISPATCH_IO_DTYPE(ops::add, main_outputs[i]->data_ptr(), shortcut_outputs[i]->data_ptr(),
                        outputs[i]->data_ptr(), outputs[i]->size());
    }
  }
  return outputs;
}

Vec<Tensor> ResidualBlockImpl::backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id) {
  // Compute gradients through final activation if present
  Vec<ConstTensor> grads_to_propagate = grad_outputs;
  if (final_activation_) {
    for (size_t i = 0; i < grad_outputs.size(); ++i) {
      std::string pre_act_key = "pre_activation_" + std::to_string(i);
      Tensor &pre_act = this->get_mutable_cache(mb_id, pre_act_key);
      Tensor grad_pre_act = this->get_tensor(pre_act->shape(), pre_act->data_type());
      final_activation_->compute_gradient(pre_act, grad_outputs[i], grad_pre_act);
      pre_act = nullptr;  // free pre-activation cache after backward
      grads_to_propagate[i] = grad_pre_act;
    }
    allocator_->flip();  // flip workspace allocator between main and shortcut backward
  }

  // Backward through main path
  Vec<Tensor> main_grad_inputs = main_path_->backward(grads_to_propagate, mb_id);

  // Backward through shortcut path
  Vec<ConstTensor> shortcut_grad_inputs = grads_to_propagate;
  if (shortcut_path_) {
    auto temp = shortcut_path_->backward(grads_to_propagate, mb_id);
    shortcut_grad_inputs = Vec<ConstTensor>(temp.begin(), temp.end());
  }

  Vec<Tensor> grad_inputs(main_grad_inputs.size());
  for (size_t i = 0; i < grad_inputs.size(); ++i) {
    grad_inputs[i] =
        this->get_tensor(main_grad_inputs[i]->shape(), main_grad_inputs[i]->data_type());
    DISPATCH_IO_DTYPE(ops::add, main_grad_inputs[i]->data_ptr(),
                      shortcut_grad_inputs[i]->data_ptr(), grad_inputs[i]->data_ptr(),
                      grad_inputs[i]->size(), defaultFlowHandle);
  }
  return grad_inputs;
}

Vec<Vec<size_t>> ResidualBlockImpl::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  return main_path_->output_shapes(input_shapes);
}

LayerConfig ResidualBlockImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("activation", activation_type_);

  LayerConfig main_config = main_path_->get_config();
  config.set("main_path", main_config.to_json());
  if (shortcut_path_) {
    LayerConfig shortcut_config = shortcut_path_->get_config();
    config.set("shortcut_path", shortcut_config.to_json());
  } else {
    config.set("shortcut_path", nlohmann::json::object());
  }
  return config;
}

std::shared_ptr<ResidualBlockImpl> ResidualBlockImpl::create_from_config(
    const LayerConfig &config) {
  std::shared_ptr<SequentialImpl> main_path, shortcut_path;
  nlohmann::json main_json = config.get<nlohmann::json>("main_path", nlohmann::json::object());
  LayerFactory::register_defaults();
  main_path = SequentialImpl::create_from_config(LayerConfig::from_json(main_json));

  shortcut_path = nullptr;
  nlohmann::json shortcut_json =
      config.get<nlohmann::json>("shortcut_path", nlohmann::json::object());
  if (!shortcut_json.is_null() && !shortcut_json.empty()) {
    LayerConfig shortcut_config = LayerConfig::from_json(shortcut_json);
    nlohmann::json layers_json =
        shortcut_config.get<nlohmann::json>("layers", nlohmann::json::array());
    if (layers_json.is_array() && !layers_json.empty()) {
      shortcut_path = SequentialImpl::create_from_config(shortcut_config);
    }
  }

  std::string activation = config.get<std::string>("activation", "relu");
  return std::make_shared<ResidualBlockImpl>(std::move(main_path), std::move(shortcut_path),
                                             activation, config.name);
}

}  // namespace synet
