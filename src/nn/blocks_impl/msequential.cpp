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
#include <stdexcept>

#include "nlohmann/json_fwd.hpp"
#include "nn/block.hpp"
#include "nn/layers.hpp"
#include "type/type.hpp"

namespace tnn {

MSequential::MSequential(Vec<std::unique_ptr<Sequential>> sequences,
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

MSequential::SequenceMemInfo MSequential::measure_sequence_memory(size_t seq_idx, ConstTensor input,
                                                                  size_t mb_id) {
  const auto &seq = sequences_[seq_idx];

  size_t m_prev = allocator_->total_allocated();
  size_t m_max = m_prev;

  size_t hook_id = allocator_->add_allocation_hook([&m_max](size_t total_allocated) {
    if (total_allocated > m_max) m_max = total_allocated;
  });

  Vec<Tensor> trial_output = seq->forward({input}, mb_id);
  this->device().getFlow(defaultFlowHandle)->synchronize();
  allocator_->remove_allocation_hook(hook_id);

  // Capture retained cost before cleanup: b_i = O_i + R_i
  size_t m_after = allocator_->total_allocated();

  // Clear side effects: cached activations in each sub-layer and the sequence itself
  for (auto *layer : seq->get_layers()) {
    layer->clear_cache(mb_id);
  }
  seq->clear_cache(mb_id);
  trial_output.clear();  // release output tensors back to the allocator

  SequenceMemInfo info;
  info.cycling_cost = m_max - m_prev;   // W_i: peak memory pressure during execution
  info.output_size = m_after - m_prev;  // b_i = O_i + R_i: retained cost before cleanup
  info.priority = static_cast<int>(info.cycling_cost) - static_cast<int>(info.output_size);
  info.index = seq_idx;

  return info;
}

Vec<size_t> MSequential::compute_execution_order(const Vec<ConstTensor> &inputs, size_t mb_id) {
  if (inputs.size() != sequences_.size()) {
    throw std::runtime_error(
        fmt::format("MSequential: Expected {} inputs, got {}", sequences_.size(), inputs.size()));
  }

  Vec<SequenceMemInfo> mem_infos;
  mem_infos.reserve(sequences_.size());

  for (size_t i = 0; i < sequences_.size(); ++i) {
    mem_infos.push_back(measure_sequence_memory(i, inputs[i], mb_id));
  }

  std::sort(
      mem_infos.begin(), mem_infos.end(),
      [](const SequenceMemInfo &a, const SequenceMemInfo &b) { return a.priority > b.priority; });

  Vec<size_t> order;
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
Vec<Tensor> MSequential::forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id) {
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
    execution_order_ = compute_execution_order(inputs, mb_id);
    execution_order_cached_ = true;
  }

  Vec<Vec<Tensor>> sequence_outputs(sequences_.size());

  for (size_t order_idx = 0; order_idx < execution_order_.size(); ++order_idx) {
    size_t seq_idx = execution_order_[order_idx];
    const auto &seq = sequences_[seq_idx];

    ConstTensor input = inputs[seq_idx];
    sequence_outputs[seq_idx] = seq->forward({input}, mb_id);
  }

  Vec<ConstTensor> join_inputs;
  join_inputs.reserve(sequence_outputs.size());
  for (auto &out : sequence_outputs) {
    join_inputs.insert(join_inputs.end(), out.begin(), out.end());
  }

  Vec<Tensor> join_outputs = join_layer_->forward(join_inputs, mb_id);

  return join_outputs;
}

Vec<Tensor> MSequential::backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id) {
  if (sequences_.empty()) {
    throw std::runtime_error("Cannot backward through empty MSequential model");
  }

  Vec<Tensor> current_grads = join_layer_->backward(grad_outputs, mb_id);

  Vec<Tensor> grad_inputs(sequences_.size());

  for (int i = static_cast<int>(sequences_.size()) - 1; i >= 0; --i) {
    grad_inputs[i] = sequences_[i]->backward({current_grads[i]}, mb_id)[0];
  }

  return grad_inputs;
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

void MSequential::print_summary(const Vec<Vec<size_t>> &input_shapes) const {
  if (sequences_.empty()) {
    std::cout << "Empty MSequential model.\n";
    return;
  }

  auto format_shape = [](const Vec<size_t> &shape) {
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
}

Vec<Sequential *> MSequential::get_sequences() {
  Vec<Sequential *> seqs;
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

  Vec<std::unique_ptr<Sequential>> sequences;
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
