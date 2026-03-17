/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/blocks_impl/residual_block.hpp"

#include <algorithm>
#include <numeric>

#include "nn/activations.hpp"
#include "nn/layers.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

ResidualBlock::ResidualBlock(std::vector<std::unique_ptr<Layer>> main_path,
                             std::vector<std::unique_ptr<Layer>> shortcut_path,
                             const std::string &final_activation, const std::string &name)
    : main_path_(std::move(main_path)),
      shortcut_path_(std::move(shortcut_path)),
      activation_type_(final_activation) {
  if (main_path_.empty()) {
    throw std::runtime_error("Main path of ResidualBlock cannot be empty.");
  }
  this->name_ = name;

  if (final_activation != "none" && final_activation != "linear") {
    auto factory = ActivationFactory();
    factory.register_defaults();
    final_activation_ = factory.create(final_activation);
  }
}

static size_t compute_path_max_size(const std::vector<std::unique_ptr<Layer>> &path,
                                    const Vec<Vec<size_t>> &input_shapes, DType_t dtype) {
  size_t max_size = 0;
  Vec<Vec<size_t>> current_shapes = input_shapes;
  for (const auto &layer : path) {
    current_shapes = layer->output_shapes(current_shapes);
    for (const auto &shape : current_shapes) {
      size_t layer_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
      max_size = std::max(max_size, layer_size);
    }
  }
  return max_size;
}

void ResidualBlock::forward_impl(const Vec<ConstTensor> &inputs, const Vec<Tensor> &outputs,
                                 size_t mb_id) {
  // Cache input shapes
  Vec<Vec<size_t>> input_shapes(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    input_shapes[i] = inputs[i]->shape();
  }
  input_shape_cache_[mb_id] = input_shapes;

  // Compute output shapes for both paths
  Vec<Vec<size_t>> main_out_shapes = input_shapes;
  for (const auto &layer : main_path_) {
    main_out_shapes = layer->output_shapes(main_out_shapes);
  }
  Vec<Vec<size_t>> shortcut_out_shapes = input_shapes;
  for (const auto &layer : shortcut_path_) {
    shortcut_out_shapes = layer->output_shapes(shortcut_out_shapes);
  }

  // Forward through main path
  Vec<ConstTensor> main_current = inputs;
  for (size_t i = 0; i < main_path_.size(); ++i) {
    Vec<Vec<size_t>> current_out_shapes = main_path_[i]->output_shapes(
        Vec<Vec<size_t>>(main_current.size(), main_current[0]->shape()));
    // Properly handle input shapes
    Vec<Vec<size_t>> actual_input_shapes;
    for (const auto &t : main_current) {
      actual_input_shapes.push_back(t->shape());
    }
    current_out_shapes = main_path_[i]->output_shapes(actual_input_shapes);

    Vec<Tensor> temp_outputs(current_out_shapes.size());
    for (size_t j = 0; j < temp_outputs.size(); ++j) {
      if (is_training_) {
        temp_outputs[j] = this->get_act(current_out_shapes[j]);
      } else {
        temp_outputs[j] = this->get_workspace(current_out_shapes[j], io_dtype_);
      }
    }
    main_path_[i]->forward(main_current, temp_outputs, mb_id);
    main_current = Vec<ConstTensor>(temp_outputs.begin(), temp_outputs.end());
  }

  // Forward through shortcut path
  Vec<ConstTensor> shortcut_current = inputs;
  for (size_t i = 0; i < shortcut_path_.size(); ++i) {
    Vec<Vec<size_t>> actual_input_shapes;
    for (const auto &t : shortcut_current) {
      actual_input_shapes.push_back(t->shape());
    }
    Vec<Vec<size_t>> current_out_shapes = shortcut_path_[i]->output_shapes(actual_input_shapes);

    Vec<Tensor> temp_outputs(current_out_shapes.size());
    for (size_t j = 0; j < temp_outputs.size(); ++j) {
      if (is_training_) {
        temp_outputs[j] = this->get_act(current_out_shapes[j]);
      } else {
        temp_outputs[j] = this->get_workspace(current_out_shapes[j], io_dtype_);
      }
    }
    shortcut_path_[i]->forward(shortcut_current, temp_outputs, mb_id);
    shortcut_current = Vec<ConstTensor>(temp_outputs.begin(), temp_outputs.end());
  }

  // Add outputs and apply final activation
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (final_activation_) {
      std::string pre_act_key = "pre_activation_" + std::to_string(i);
      Tensor &pre_act = this->get_mutable_tensor(mb_id, pre_act_key);
      if (!pre_act)
        pre_act = this->make_io_tensor(main_current[i]->shape());
      else
        pre_act->ensure(main_current[i]->shape());
      DISPATCH_IO_DTYPE(ops::add, main_current[i]->data_ptr(), shortcut_current[i]->data_ptr(),
                        pre_act->data_ptr(), pre_act->size());

      outputs[i]->ensure(main_current[i]->shape());
      final_activation_->apply(pre_act, outputs[i]);
    } else {
      outputs[i]->ensure(main_current[i]->shape());
      DISPATCH_IO_DTYPE(ops::add, main_current[i]->data_ptr(), shortcut_current[i]->data_ptr(),
                        outputs[i]->data_ptr(), outputs[i]->size());
    }
  }
}

void ResidualBlock::backward_impl(const Vec<ConstTensor> &grad_outputs,
                                  const Vec<Tensor> &grad_inputs, size_t mb_id) {
  auto it_input_shapes = input_shape_cache_.find(mb_id);
  if (it_input_shapes == input_shape_cache_.end()) {
    throw std::runtime_error("No cached input shapes found for micro-batch ID: " +
                             std::to_string(mb_id));
  }
  Vec<Vec<size_t>> input_shapes = it_input_shapes->second;

  // Compute gradients through final activation if present
  Vec<ConstTensor> grads_to_propagate(grad_outputs.size());
  for (size_t i = 0; i < grad_outputs.size(); ++i) {
    if (final_activation_) {
      std::string pre_act_key = "pre_activation_" + std::to_string(i);
      const Tensor &pre_act = this->get_mutable_tensor(mb_id, pre_act_key);
      if (!pre_act) {
        throw std::runtime_error("No cached pre-activation output found for micro-batch ID: " +
                                 std::to_string(mb_id));
      }
      Tensor dpre_act = this->make_io_tensor(pre_act->shape());
      final_activation_->compute_gradient(pre_act, grad_outputs[i], dpre_act);
      grads_to_propagate[i] = dpre_act;
    } else {
      grads_to_propagate[i] = grad_outputs[i];
    }
  }

  size_t main_path_max_size =
      compute_path_max_size(main_path_, input_shapes, grad_outputs[0]->data_type());
  size_t shortcut_path_max_size =
      compute_path_max_size(shortcut_path_, input_shapes, grad_outputs[0]->data_type());

  // Backward through main path
  Vec<ConstTensor> main_grad_current = grads_to_propagate;
  for (int i = static_cast<int>(main_path_.size()) - 1; i >= 0; --i) {
    Vec<Vec<size_t>> grad_input_shapes;
    if (i == 0) {
      grad_input_shapes = input_shapes;
    } else {
      Vec<Vec<size_t>> prev_input_shapes = input_shapes;
      for (int j = 0; j < i; ++j) {
        prev_input_shapes = main_path_[j]->output_shapes(prev_input_shapes);
      }
      grad_input_shapes = prev_input_shapes;
    }

    Vec<Tensor> temp_grads(grad_input_shapes.size());
    for (size_t j = 0; j < temp_grads.size(); ++j) {
      temp_grads[j] = this->get_workspace({main_path_max_size}, grad_outputs[0]->data_type());
    }
    main_path_[i]->backward(main_grad_current, temp_grads, mb_id);
    main_grad_current = Vec<ConstTensor>(temp_grads.begin(), temp_grads.end());
  }

  // Backward through shortcut path
  Vec<ConstTensor> shortcut_grad_current = grads_to_propagate;
  if (!shortcut_path_.empty()) {
    for (int i = static_cast<int>(shortcut_path_.size()) - 1; i >= 0; --i) {
      Vec<Vec<size_t>> grad_input_shapes;
      if (i == 0) {
        grad_input_shapes = input_shapes;
      } else {
        Vec<Vec<size_t>> prev_input_shapes = input_shapes;
        for (int j = 0; j < i; ++j) {
          prev_input_shapes = shortcut_path_[j]->output_shapes(prev_input_shapes);
        }
        grad_input_shapes = prev_input_shapes;
      }

      Vec<Tensor> temp_grads(grad_input_shapes.size());
      for (size_t j = 0; j < temp_grads.size(); ++j) {
        temp_grads[j] = this->get_workspace({shortcut_path_max_size}, grad_outputs[0]->data_type());
      }
      shortcut_path_[i]->backward(shortcut_grad_current, temp_grads, mb_id);
      shortcut_grad_current = Vec<ConstTensor>(temp_grads.begin(), temp_grads.end());
    }
  }

  // Add gradients from both paths
  for (size_t i = 0; i < grad_inputs.size(); ++i) {
    grad_inputs[i]->ensure(main_grad_current[i]->shape());
    DISPATCH_IO_DTYPE(ops::add, main_grad_current[i]->data_ptr(),
                      shortcut_grad_current[i]->data_ptr(), grad_inputs[i]->data_ptr(),
                      grad_inputs[i]->size());
  }
}

Vec<Vec<size_t>> ResidualBlock::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  Vec<Vec<size_t>> current_shapes = input_shapes;
  for (const auto &layer : main_path_) {
    current_shapes = layer->output_shapes(current_shapes);
  }
  return current_shapes;
}

LayerConfig ResidualBlock::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("activation", activation_type_);

  nlohmann::json main_array = nlohmann::json::array();
  for (const auto &layer : main_path_) {
    LayerConfig sub_cfg = layer->get_config();
    nlohmann::json sub_json = sub_cfg.to_json();
    main_array.push_back(sub_json);
  }

  nlohmann::json shortcut_array = nlohmann::json::array();
  for (const auto &layer : shortcut_path_) {
    LayerConfig sub_cfg = layer->get_config();
    nlohmann::json sub_json = sub_cfg.to_json();
    shortcut_array.push_back(sub_json);
  }

  config.set("main_path", main_array);
  config.set("shortcut_path", shortcut_array);

  return config;
}

std::unique_ptr<ResidualBlock> ResidualBlock::create_from_config(const LayerConfig &config) {
  std::vector<std::unique_ptr<Layer>> main_path;
  std::vector<std::unique_ptr<Layer>> shortcut_path;
  nlohmann::json main_json = config.get<nlohmann::json>("main_path", nlohmann::json::array());
  LayerFactory::register_defaults();
  for (const auto &layer_json : main_json) {
    LayerConfig layer_config = LayerConfig::from_json(layer_json);
    auto layer = LayerFactory::create(layer_config);
    main_path.push_back(std::move(layer));
  }
  nlohmann::json shortcut_json =
      config.get<nlohmann::json>("shortcut_path", nlohmann::json::array());
  for (const auto &layer_json : shortcut_json) {
    LayerConfig layer_config = LayerConfig::from_json(layer_json);
    auto layer = LayerFactory::create(layer_config);
    shortcut_path.push_back(std::move(layer));
  }

  std::string activation = config.get<std::string>("activation", "relu");
  return std::make_unique<ResidualBlock>(std::move(main_path), std::move(shortcut_path), activation,
                                         config.name);
}

size_t ResidualBlock::fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  if (main_path_.empty()) return 0;
  const auto &input_shape = input_shapes[0];
  size_t total_ws = 0;
  size_t dtype_size = get_dtype_size(io_dtype_);

  // Main path workspace
  Vec<size_t> main_shape = input_shape;
  for (const auto &layer : main_path_) {
    total_ws += layer->fwd_workspace({{main_shape}});
    total_ws += std::accumulate(main_shape.begin(), main_shape.end(), dtype_size,
                                std::multiplies<size_t>());
    main_shape = layer->output_shapes({main_shape})[0];
  }

  // Shortcut path workspace
  Vec<size_t> shortcut_shape = input_shape;
  for (const auto &layer : shortcut_path_) {
    total_ws += layer->fwd_workspace({{shortcut_shape}});
    total_ws += std::accumulate(shortcut_shape.begin(), shortcut_shape.end(), dtype_size,
                                std::multiplies<size_t>());
    shortcut_shape = layer->output_shapes({shortcut_shape})[0];
  }

  // Pre-activation buffer if final activation exists
  if (final_activation_) {
    total_ws += std::accumulate(main_shape.begin(), main_shape.end(), dtype_size,
                                std::multiplies<size_t>());
  }

  return total_ws;
}

size_t ResidualBlock::inf_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  if (main_path_.empty()) return 0;
  const auto &input_shape = input_shapes[0];
  size_t dtype_size = get_dtype_size(io_dtype_);

  // Helper lambda to compute buffer cycling workspace for a path (SISO sequence)
  auto compute_path_workspace = [dtype_size](const std::vector<std::unique_ptr<Layer>> &path,
                                             const Vec<size_t> &input_shape) -> size_t {
    if (path.empty()) return 0;
    Vec<size_t> out_bytes;
    Vec<size_t> sub_ws;
    Vec<size_t> cur = input_shape;
    for (const auto &layer : path) {
      Vec<size_t> out = layer->output_shapes({cur})[0];
      size_t bytes = std::accumulate(out.begin(), out.end(), dtype_size, std::multiplies<size_t>());
      out_bytes.push_back(bytes);
      sub_ws.push_back(layer->inf_workspace({{cur}}));
      cur = out;
    }
    size_t m_b = 0;
    for (size_t i = 0; i < out_bytes.size() - 1; ++i) {
      m_b = std::max(m_b, out_bytes[i] + out_bytes[i + 1] + sub_ws[i]);
    }
    if (!out_bytes.empty() && !sub_ws.empty()) {
      m_b = std::max(m_b, out_bytes.back() + sub_ws.back());
    }
    return m_b;
  };

  // Compute O_i (terminal output size) for each path
  Vec<size_t> main_output_shape = input_shape;
  for (const auto &layer : main_path_) {
    main_output_shape = layer->output_shapes({main_output_shape})[0];
  }
  size_t main_output_bytes = std::accumulate(main_output_shape.begin(), main_output_shape.end(),
                                             dtype_size, std::multiplies<size_t>());

  Vec<size_t> shortcut_output_shape = input_shape;
  for (const auto &layer : shortcut_path_) {
    shortcut_output_shape = layer->output_shapes({shortcut_output_shape})[0];
  }
  size_t shortcut_output_bytes =
      std::accumulate(shortcut_output_shape.begin(), shortcut_output_shape.end(), dtype_size,
                      std::multiplies<size_t>());

  // Compute M_b,i (buffer cycling workspace) for each path
  size_t main_workspace = compute_path_workspace(main_path_, input_shape);
  size_t shortcut_workspace = compute_path_workspace(shortcut_path_, input_shape);

  // Apply Algorithm 2 from paper (MISO joining)
  // We have 2 sequences with (a_i, b_i) = (M_b,i, O_i)
  // Sort by (a_i - b_i) descending to find optimal order
  struct PathInfo {
    size_t workspace;     // a_i
    size_t output_bytes;  // b_i
    size_t priority;      // a_i - b_i
  };

  PathInfo main_info{main_workspace, main_output_bytes, main_workspace - main_output_bytes};
  PathInfo shortcut_info{shortcut_workspace, shortcut_output_bytes,
                         shortcut_workspace - shortcut_output_bytes};

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
  size_t join_output_bytes = std::accumulate(main_output_shape.begin(), main_output_shape.end(),
                                             dtype_size, std::multiplies<size_t>());
  k = std::max(k, main_output_bytes + shortcut_output_bytes + join_output_bytes);

  return k;
}

size_t ResidualBlock::bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  if (main_path_.empty()) return 0;
  const auto &input_shape = input_shapes[0];
  size_t dtype_size = get_dtype_size(io_dtype_);

  // Helper lambda to compute buffer cycling workspace for a path (SISO sequence)
  auto compute_path_workspace = [dtype_size](const std::vector<std::unique_ptr<Layer>> &path,
                                             const Vec<size_t> &input_shape) -> size_t {
    if (path.empty()) return 0;
    Vec<size_t> out_bytes;
    Vec<size_t> sub_ws;
    Vec<size_t> cur = input_shape;
    for (const auto &layer : path) {
      Vec<size_t> out = layer->output_shapes({cur})[0];
      size_t bytes = std::accumulate(out.begin(), out.end(), dtype_size, std::multiplies<size_t>());
      out_bytes.push_back(bytes);
      sub_ws.push_back(layer->bwd_workspace({{cur}}));
      cur = out;
    }
    size_t m_b = 0;
    for (size_t i = 0; i < out_bytes.size() - 1; ++i) {
      m_b = std::max(m_b, out_bytes[i] + out_bytes[i + 1] + sub_ws[i]);
    }
    if (!out_bytes.empty() && !sub_ws.empty()) {
      m_b = std::max(m_b, out_bytes.back() + sub_ws.back());
    }
    return m_b;
  };

  // Compute O_i (terminal output size) for each path
  Vec<size_t> main_output_shape = input_shape;
  for (const auto &layer : main_path_) {
    main_output_shape = layer->output_shapes({main_output_shape})[0];
  }
  size_t main_output_bytes = std::accumulate(main_output_shape.begin(), main_output_shape.end(),
                                             dtype_size, std::multiplies<size_t>());

  Vec<size_t> shortcut_output_shape = input_shape;
  for (const auto &layer : shortcut_path_) {
    shortcut_output_shape = layer->output_shapes({shortcut_output_shape})[0];
  }
  size_t shortcut_output_bytes =
      std::accumulate(shortcut_output_shape.begin(), shortcut_output_shape.end(), dtype_size,
                      std::multiplies<size_t>());

  // Compute M_b,i (buffer cycling workspace) for each path
  size_t main_workspace = compute_path_workspace(main_path_, input_shape);
  size_t shortcut_workspace = compute_path_workspace(shortcut_path_, input_shape);

  // Apply Algorithm 2 from paper (MISO joining)
  // We have 2 sequences with (a_i, b_i) = (M_b,i, O_i)
  // Sort by (a_i - b_i) descending to find optimal order
  struct PathInfo {
    size_t workspace;     // a_i
    size_t output_bytes;  // b_i
    size_t priority;      // a_i - b_i
  };

  PathInfo main_info{main_workspace, main_output_bytes, main_workspace - main_output_bytes};
  PathInfo shortcut_info{shortcut_workspace, shortcut_output_bytes,
                         shortcut_workspace - shortcut_output_bytes};

  // Execute in optimal order and compute peak memory
  size_t k = 0;
  if (main_info.priority >= shortcut_info.priority) {
    // Execute main first, then shortcut
    k = std::max(main_info.workspace, shortcut_info.workspace + main_info.output_bytes);
  } else {
    // Execute shortcut first, then main
    k = std::max(shortcut_info.workspace, main_info.workspace + shortcut_info.output_bytes);
  }

  // Add space for the backward through join operation
  size_t grad_input_bytes = std::accumulate(input_shape.begin(), input_shape.end(), dtype_size,
                                            std::multiplies<size_t>());
  k = std::max(k, main_output_bytes + shortcut_output_bytes + grad_input_bytes);

  return k;
}

}  // namespace tnn
