/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/blocks_impl/residual_block.hpp"

#include <alloca.h>

#include <algorithm>
#include <cstddef>
#include <numeric>

#include "nn/activations.hpp"
#include "nn/layer.hpp"
#include "nn/layers.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

ResidualBlock::ResidualBlock(Vec<std::unique_ptr<Layer>> main_path,
                             Vec<std::unique_ptr<Layer>> shortcut_path,
                             const std::string &final_activation, const std::string &name)
    : activation_type_(final_activation) {
  if (main_path.empty()) {
    throw std::runtime_error("Main path of ResidualBlock cannot be empty.");
  }
  main_path_ = std::make_unique<Sequential>(std::move(main_path), name + "_main_path");
  if (!shortcut_path.empty()) {
    shortcut_path_ =
        std::make_unique<Sequential>(std::move(shortcut_path), name + "_shortcut_path");
  }
  this->name_ = name;

  if (final_activation != "none" && final_activation != "linear") {
    auto factory = ActivationFactory();
    factory.register_defaults();
    final_activation_ = factory.create(final_activation);
  }
}

ResidualBlock::ResidualBlock(std::unique_ptr<Sequential> main_path,
                             std::unique_ptr<Sequential> shortcut_path,
                             const std::string &final_activation, const std::string &name)
    : main_path_(std::move(main_path)),
      shortcut_path_(std::move(shortcut_path)),
      activation_type_(final_activation) {
  if (!main_path_) {
    throw std::runtime_error("Main path of ResidualBlock cannot be null.");
  }
  this->name_ = name;

  if (final_activation != "none" && final_activation != "linear") {
    auto factory = ActivationFactory();
    factory.register_defaults();
    final_activation_ = factory.create(final_activation);
  }
}

Vec<Tensor> ResidualBlock::forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id) {
  // Cache input shapes
  Vec<Vec<size_t>> input_shapes(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    input_shapes[i] = inputs[i]->shape();
  }
  input_shape_cache_[mb_id] = input_shapes;

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
      Tensor pre_act = get_cache_tensor(main_outputs[i]->shape(), io_dtype_);
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

Vec<Tensor> ResidualBlock::backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id) {
  auto it_input_shapes = input_shape_cache_.find(mb_id);
  if (it_input_shapes == input_shape_cache_.end()) {
    throw std::runtime_error("No cached input shapes found for micro-batch ID: " +
                             std::to_string(mb_id));
  }
  Vec<Vec<size_t>> input_shapes = it_input_shapes->second;

  // Compute gradients through final activation if present
  Vec<ConstTensor> grads_to_propagate = grad_outputs;
  if (final_activation_) {
    for (size_t i = 0; i < grad_outputs.size(); ++i) {
      std::string pre_act_key = "pre_activation_" + std::to_string(i);
      Tensor &pre_act = this->get_mutable_cache(mb_id, pre_act_key);
      Tensor grad_pre_act = this->get_workspace(pre_act->shape());
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
    grad_inputs[i] = this->get_output_tensor(input_shapes[i]);
    DISPATCH_IO_DTYPE(ops::add, main_grad_inputs[i]->data_ptr(),
                      shortcut_grad_inputs[i]->data_ptr(), grad_inputs[i]->data_ptr(),
                      grad_inputs[i]->size(), defaultFlowHandle);
  }
  return grad_inputs;
}

Vec<Vec<size_t>> ResidualBlock::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  return main_path_->output_shapes(input_shapes);
}

LayerConfig ResidualBlock::get_config() const {
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

std::unique_ptr<ResidualBlock> ResidualBlock::create_from_config(const LayerConfig &config) {
  std::unique_ptr<Sequential> main_path, shortcut_path;
  nlohmann::json main_json = config.get<nlohmann::json>("main_path", nlohmann::json::object());
  LayerFactory::register_defaults();
  main_path = Sequential::create_from_config(LayerConfig::from_json(main_json));

  shortcut_path = nullptr;
  nlohmann::json shortcut_json =
      config.get<nlohmann::json>("shortcut_path", nlohmann::json::object());
  if (!shortcut_json.is_null()) {
    shortcut_path = Sequential::create_from_config(LayerConfig::from_json(shortcut_json));
  }

  std::string activation = config.get<std::string>("activation", "relu");
  return std::make_unique<ResidualBlock>(std::move(main_path), std::move(shortcut_path), activation,
                                         config.name);
}

size_t ResidualBlock::fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const {
  size_t total_cache = 0;
  total_cache += main_path_->fwd_cache_bytes(input_shapes);
  if (shortcut_path_) {
    total_cache += shortcut_path_->fwd_cache_bytes(input_shapes);
  }
  return total_cache;
}

size_t ResidualBlock::fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  size_t total_ws = 0;

  auto output_shapes = main_path_->output_shapes(input_shapes);
  size_t dtype_size = get_dtype_size(io_dtype_);
  size_t output_bytes = 0;
  for (const auto &shape : output_shapes) {
    output_bytes +=
        std::accumulate(shape.begin(), shape.end(), dtype_size, std::multiplies<size_t>());
  }
  total_ws = std::max(
      total_ws,
      output_bytes + output_bytes);  // need to keep both main and shortcut outputs for the join

  // Main path workspace
  total_ws = std::max(total_ws, main_path_->fwd_workspace(input_shapes));

  // Shortcut path workspace
  if (shortcut_path_) {
    total_ws = std::max(total_ws, shortcut_path_->fwd_workspace(input_shapes));
  }

  return total_ws;
}

size_t ResidualBlock::inf_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  size_t dtype_size = get_dtype_size(io_dtype_);

  auto output_shapes = main_path_->output_shapes(input_shapes);
  size_t main_output_bytes = 0;
  for (const auto &shape : output_shapes) {
    main_output_bytes +=
        std::accumulate(shape.begin(), shape.end(), dtype_size, std::multiplies<size_t>());
  }

  size_t shortcut_output_bytes = 0;
  if (shortcut_path_) {
    auto shortcut_output_shapes = shortcut_path_->output_shapes(input_shapes);
    for (const auto &shape : shortcut_output_shapes) {
      shortcut_output_bytes +=
          std::accumulate(shape.begin(), shape.end(), dtype_size, std::multiplies<size_t>());
    }
  }

  // Compute M_b,i (buffer cycling workspace) for each path
  size_t main_workspace = main_path_->inf_workspace(input_shapes);
  size_t shortcut_workspace = 0;
  if (shortcut_path_) {
    shortcut_workspace = shortcut_path_->inf_workspace(input_shapes);
  }

  // Apply Algorithm 2 from paper (MISO joining)
  // We have 2 sequences with (a_i, b_i) = (M_b,i, O_i)
  // Sort by (a_i - b_i) descending to find optimal order
  struct PathInfo {
    size_t workspace;     // a_i
    size_t output_bytes;  // b_i
    size_t priority;      // a_i - b_i
  };

  PathInfo main_info{main_workspace + main_output_bytes, main_output_bytes, main_workspace};
  PathInfo shortcut_info{shortcut_workspace + shortcut_output_bytes, shortcut_output_bytes,
                         shortcut_workspace};

  // Execute in optimal order and compute peak memory
  size_t k = 0;
  if (main_info.priority >= shortcut_info.priority) {
    // Execute main first, then shortcut
    k = std::max(main_info.workspace, shortcut_info.workspace + main_info.output_bytes);
  } else {
    // Execute shortcut first, then main
    k = std::max(shortcut_info.workspace, main_info.workspace + shortcut_info.output_bytes);
  }

  // Add space for the join operation (add + optional activation)
  size_t join_output_bytes = 0;
  for (const auto &shape : output_shapes) {
    join_output_bytes +=
        std::accumulate(shape.begin(), shape.end(), dtype_size, std::multiplies<size_t>());
  }

  k = std::max(k, main_output_bytes + shortcut_output_bytes + join_output_bytes);

  return k;
}

size_t ResidualBlock::bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  size_t total_ws = 0;

  total_ws = std::max(total_ws, main_path_->bwd_workspace(input_shapes));

  if (shortcut_path_) {
    total_ws = std::max(total_ws, shortcut_path_->bwd_workspace(input_shapes));
  }

  return total_ws;
}

}  // namespace tnn
