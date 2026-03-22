/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/blocks_impl/msequential.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "nlohmann/json_fwd.hpp"
#include "nn/block.hpp"
#include "nn/layers.hpp"
#include "type/type.hpp"

namespace tnn {

MSequential::MSequential(std::vector<std::unique_ptr<Sequential>> sequences,
                         std::unique_ptr<Layer> join_layer, const std::string &name)
    : Block(name),
      sequences_(std::move(sequences)),
      join_layer_(std::move(join_layer)) {
  if (sequences_.empty()) {
    throw std::runtime_error("MSequential requires at least one sequence");
  }
  if (!join_layer_) {
    throw std::runtime_error("MSequential requires a join layer");
  }
}

MSequential::SequenceMemInfo MSequential::compute_sequence_memory(
    size_t seq_idx, const Vec<size_t> &input_shapes) const {
  const auto &seq = sequences_[seq_idx];
  size_t dtype_size = get_dtype_size(io_dtype_);

  size_t cycling_cost = seq->inf_workspace({input_shapes});

  Vec<Vec<size_t>> output_shape = seq->output_shapes({input_shapes});
  size_t output_size = 0;
  for (const auto &shape : output_shape) {
    output_size +=
        std::accumulate(shape.begin(), shape.end(), dtype_size, std::multiplies<size_t>());
  }

  SequenceMemInfo info;
  info.cycling_cost = cycling_cost;
  info.output_size = output_size;

  info.priority = static_cast<int>(cycling_cost) - static_cast<int>(output_size);
  info.index = seq_idx;

  return info;
}

std::vector<size_t> MSequential::compute_execution_order(
    const Vec<Vec<size_t>> &input_shapes) const {
  if (input_shapes.size() != sequences_.size()) {
    throw std::runtime_error(fmt::format("MSequential: Expected {} inputs, got {}",
                                         sequences_.size(), input_shapes.size()));
  }

  std::vector<SequenceMemInfo> mem_infos;
  mem_infos.reserve(sequences_.size());

  for (size_t i = 0; i < sequences_.size(); ++i) {
    mem_infos.push_back(compute_sequence_memory(i, input_shapes[i]));
  }

  std::sort(
      mem_infos.begin(), mem_infos.end(),
      [](const SequenceMemInfo &a, const SequenceMemInfo &b) { return a.priority > b.priority; });

  std::vector<size_t> order;
  order.reserve(sequences_.size());
  for (const auto &info : mem_infos) {
    order.push_back(info.index);
  }

  return order;
}

/**
 * @brief Forward pass for the MSequential block
 *
 * @param inputs M input tensor corresponding to each sequence's input
 * @param outputs Output tensors from the join layer
 * @param mb_id Micro-batch ID
 */
void MSequential::forward_impl(const Vec<ConstTensor> &inputs, const Vec<Tensor> &outputs,
                               size_t mb_id) {
  if (sequences_.empty()) {
    throw std::runtime_error("Cannot forward through empty MSequential model");
  }

  if (inputs.size() != sequences_.size()) {
    throw std::runtime_error(
        fmt::format("MSequential: Expected {} inputs, got {}", sequences_.size(), inputs.size()));
  }

  Vec<Vec<size_t>> input_shapes(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    input_shapes[i] = inputs[i]->shape();
  }
  input_shapes_cache_[mb_id] = input_shapes;

  if (!execution_order_cached_) {
    execution_order_ = compute_execution_order(input_shapes);
    execution_order_cached_ = true;
  }

  Vec<Vec<Tensor>> sequence_outputs(sequences_.size());
  Vec<Vec<Vec<size_t>>> output_shapes(sequences_.size());

  for (size_t i = 0; i < sequences_.size(); ++i) {
    output_shapes[i] = sequences_[i]->output_shapes({{input_shapes[i]}});
  }

  for (size_t order_idx = 0; order_idx < execution_order_.size(); ++order_idx) {
    size_t seq_idx = execution_order_[order_idx];
    const auto &seq = sequences_[seq_idx];

    ConstTensor input = inputs[seq_idx];

    Vec<Tensor> seq_output(output_shapes[seq_idx].size());
    for (size_t j = 0; j < output_shapes[seq_idx].size(); ++j) {
      seq_output[j] = this->get_workspace(output_shapes[seq_idx][j], io_dtype_);
      if (is_training_) {
        allocator_->flip();
      }
    }

    seq->forward({input}, seq_output, mb_id);

    sequence_outputs[seq_idx] = seq_output;
  }

  Vec<ConstTensor> join_inputs;
  join_inputs.reserve(sequence_outputs.size());
  for (auto &out : sequence_outputs) {
    join_inputs.insert(join_inputs.end(), out.begin(), out.end());
  }

  join_layer_->forward(join_inputs, outputs, mb_id);

  this->device().getFlow(this->flow_handle_)->synchronize();
}

void MSequential::backward_impl(const Vec<ConstTensor> &grad_outputs,
                                const Vec<Tensor> &grad_inputs, size_t mb_id) {
  if (sequences_.empty()) {
    throw std::runtime_error("Cannot backward through empty MSequential model");
  }

  auto it_in_shapes = input_shapes_cache_.find(mb_id);
  if (it_in_shapes == input_shapes_cache_.end()) {
    throw std::runtime_error("No cached input shape found for micro-batch ID: " +
                             std::to_string(mb_id));
  }
  const Vec<Vec<size_t>> &input_shapes = it_in_shapes->second;

  Vec<Vec<Vec<size_t>>> output_shapes(sequences_.size());
  for (size_t i = 0; i < sequences_.size(); ++i) {
    output_shapes[i] = sequences_[i]->output_shapes({input_shapes[i]});
  }

  Vec<Vec<Tensor>> seq_grad_outputs(sequences_.size());
  for (size_t i = 0; i < sequences_.size(); ++i) {
    seq_grad_outputs[i].resize(output_shapes[i].size());
    for (size_t j = 0; j < output_shapes[i].size(); ++j) {
      seq_grad_outputs[i][j] = this->get_workspace(output_shapes[i][j], io_dtype_);
    }
  }

  Vec<Tensor> current_grad;
  for (size_t i = 0; i < sequences_.size(); ++i) {
    current_grad.insert(current_grad.end(), seq_grad_outputs[i].begin(), seq_grad_outputs[i].end());
  }

  join_layer_->backward(grad_outputs, current_grad, mb_id);

  for (int i = static_cast<int>(sequences_.size()) - 1; i >= 0; --i) {
    Vec<ConstTensor> path_grad_outputs(seq_grad_outputs[i].begin(), seq_grad_outputs[i].end());
    sequences_[i]->backward(path_grad_outputs, {grad_inputs[i]}, mb_id);
  }

  this->device().getFlow(this->flow_handle_)->synchronize();
}

Vec<Vec<size_t>> MSequential::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (input_shapes.size() != sequences_.size()) {
    throw std::runtime_error(fmt::format("MSequential: Expected {} inputs, got {}",
                                         sequences_.size(), input_shapes.size()));
  }

  Vec<Vec<size_t>> sequence_output_shapes;
  sequence_output_shapes.reserve(sequences_.size());

  for (size_t i = 0; i < sequences_.size(); ++i) {
    Vec<Vec<size_t>> seq_out = sequences_[i]->output_shapes({{input_shapes[i]}});
    sequence_output_shapes.push_back(seq_out[0]);
  }

  return join_layer_->output_shapes(sequence_output_shapes);
}

size_t MSequential::fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const {
  if (sequences_.empty()) return 0;

  size_t total_cache = 0;
  Vec<Vec<size_t>> current_shapes = input_shapes;

  for (auto it = sequences_.begin(); it != sequences_.end(); ++it) {
    const auto &layer = *it;
    total_cache += layer->fwd_cache_bytes(current_shapes);
    current_shapes = layer->output_shapes(current_shapes);
  }

  return total_cache;
}

size_t MSequential::fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  if (sequences_.empty()) return 0;

  size_t total_ws = 0;

  size_t total_seq_ws = 0;
  for (size_t i = 0; i < sequences_.size(); ++i) {
    size_t seq_ws = sequences_[i]->fwd_workspace({input_shapes[i]});
    total_seq_ws = std::max(total_seq_ws, seq_ws);
  }
  total_ws = std::max(total_ws, total_seq_ws);

  Vec<Vec<size_t>> seq_output_shapes;
  for (size_t i = 0; i < sequences_.size(); ++i) {
    auto seq_outs = sequences_[i]->output_shapes({{input_shapes[i]}});
    seq_output_shapes.push_back(seq_outs[0]);
  }

  size_t total_join_ws = join_layer_->fwd_workspace(seq_output_shapes);

  total_ws = std::max(total_ws, total_join_ws);

  return total_ws;
}

size_t MSequential::inf_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  if (sequences_.empty()) return 0;

  size_t dtype_size = get_dtype_size(io_dtype_);

  std::vector<size_t> order = compute_execution_order(input_shapes);

  size_t k = 0;
  size_t accumulated = 0;

  for (size_t order_idx = 0; order_idx < order.size(); ++order_idx) {
    size_t seq_idx = order[order_idx];
    SequenceMemInfo info = compute_sequence_memory(seq_idx, input_shapes[seq_idx]);
    k = std::max(k, info.cycling_cost + accumulated);
    accumulated += info.output_size;
  }

  Vec<Vec<size_t>> seq_output_shapes;
  for (size_t i = 0; i < sequences_.size(); ++i) {
    auto seq_outs = sequences_[i]->output_shapes({{input_shapes[i]}});
    seq_output_shapes.push_back(seq_outs[0]);
  }

  size_t join_ws = join_layer_->inf_workspace(seq_output_shapes);
  Vec<size_t> join_output = join_layer_->output_shapes(seq_output_shapes)[0];
  size_t join_output_bytes = std::accumulate(join_output.begin(), join_output.end(), dtype_size,
                                             std::multiplies<size_t>());

  k = std::max(k, accumulated + join_ws + join_output_bytes);

  return k;
}

size_t MSequential::bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  if (sequences_.empty()) return 0;

  size_t dtype_size = get_dtype_size(io_dtype_);

  size_t total_ws = 0;

  for (size_t i = 0; i < sequences_.size(); ++i) {
    Vec<size_t> seq_output = sequences_[i]->output_shapes({{input_shapes[i]}})[0];
    size_t grad_bytes = std::accumulate(seq_output.begin(), seq_output.end(), dtype_size,
                                        std::multiplies<size_t>());
    total_ws += grad_bytes;

    total_ws += sequences_[i]->bwd_workspace({{input_shapes[i]}});
  }

  Vec<Vec<size_t>> seq_output_shapes;
  for (size_t i = 0; i < sequences_.size(); ++i) {
    seq_output_shapes.push_back(sequences_[i]->output_shapes({{input_shapes[i]}})[0]);
  }
  total_ws += join_layer_->bwd_workspace(seq_output_shapes);

  return total_ws;
}

void MSequential::print_summary(const Vec<Vec<size_t>> &input_shapes) const {
  if (sequences_.empty()) {
    std::cout << "Empty MSequential model.\n";
    return;
  }

  auto format_shape = [](const std::vector<size_t> &shape) {
    std::string shape_str = "(";
    for (size_t j = 0; j < shape.size(); ++j) {
      if (j > 0) shape_str += ",";
      shape_str += std::to_string(shape[j]);
    }
    shape_str += ")";
    return shape_str;
  };

  std::cout << std::string(100, '=') << "\n";
  std::cout << "MSequential Model Summary: " << name_ << "\n";
  std::cout << "Number of branches: " << sequences_.size() << "\n";
  std::cout << std::string(100, '=') << "\n";

  for (size_t i = 0; i < sequences_.size(); ++i) {
    std::cout << "\n--- Branch " << i << " (Input Shape: " << format_shape(input_shapes[i])
              << ") ---\n";
    sequences_[i]->print_summary(input_shapes[i]);
  }

  std::cout << "\n--- Join Layer ---\n";
  std::cout << "Type: " << join_layer_->type() << "\n";
  std::cout << "Name: " << (join_layer_->name().empty() ? "<unnamed>" : join_layer_->name())
            << "\n";

  Vec<Vec<size_t>> seq_outputs;
  for (size_t i = 0; i < sequences_.size(); ++i) {
    seq_outputs.push_back(sequences_[i]->output_shapes({{input_shapes[i]}})[0]);
  }

  std::cout << "Input shapes: ";
  for (size_t i = 0; i < seq_outputs.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << format_shape(seq_outputs[i]);
  }
  std::cout << "\n";

  Vec<Vec<size_t>> final_output = join_layer_->output_shapes(seq_outputs);
  std::cout << "Output shape: " << format_shape(final_output[0]) << "\n";

  std::cout << "\n--- Memory Statistics ---\n";
  std::cout << "Inference workspace: " << inf_workspace(input_shapes) / (1024.0 * 1024.0)
            << " MB\n";
  std::cout << "Forward workspace: " << fwd_workspace(input_shapes) / (1024.0 * 1024.0) << " MB\n";
  std::cout << "Backward workspace: " << bwd_workspace(input_shapes) / (1024.0 * 1024.0) << " MB\n";

  std::vector<size_t> order = compute_execution_order(input_shapes);
  std::cout << "\nOptimal execution order (by priority M_b,i - O_i): ";
  for (size_t i = 0; i < order.size(); ++i) {
    if (i > 0) std::cout << " -> ";
    std::cout << "Branch_" << order[i];
  }
  std::cout << " -> Join\n";

  std::cout << std::string(100, '-') << "\n";
}

std::vector<Sequential *> MSequential::get_sequences() {
  std::vector<Sequential *> seqs;
  seqs.reserve(sequences_.size());
  for (auto &seq : sequences_) {
    seqs.push_back(seq.get());
  }
  return seqs;
}

Layer *MSequential::get_join_layer() { return join_layer_.get(); }

LayerConfig MSequential::get_config() const {
  LayerConfig config;
  config.name = name_;
  config.type = TYPE_NAME;

  nlohmann::json sequences_config = nlohmann::json::array();
  for (const auto &seq : sequences_) {
    auto seq_config = seq->get_config();
    sequences_config.push_back(seq_config.to_json());
  }
  config.set("sequences", sequences_config);

  if (join_layer_) {
    auto join_config = join_layer_->get_config();
    config.set("join_layer", join_config.to_json());
  }

  return config;
}

std::unique_ptr<MSequential> MSequential::create_from_config(const LayerConfig &config) {
  nlohmann::json sequences_json = config.get<nlohmann::json>("sequences", nlohmann::json::array());
  if (!sequences_json.is_array()) {
    throw std::runtime_error("MSequential config 'sequences' parameter must be an array");
  }

  std::vector<std::unique_ptr<Sequential>> sequences;
  LayerFactory::register_defaults();

  for (const auto &seq_json : sequences_json) {
    LayerConfig seq_config = LayerConfig::from_json(seq_json);
    auto seq = Sequential::create_from_config(seq_config);
    sequences.push_back(std::move(seq));
  }

  nlohmann::json join_json = config.get<nlohmann::json>("join_layer");
  LayerConfig join_config = LayerConfig::from_json(join_json);
  auto join_layer = LayerFactory::create(join_config);

  return std::make_unique<MSequential>(std::move(sequences), std::move(join_layer), config.name);
}

}  // namespace tnn
