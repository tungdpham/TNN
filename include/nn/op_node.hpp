#pragma once

#include "nn/io_node.hpp"
#include "nn/layer.hpp"
#include "nn/node.hpp"

namespace tnn {

// Simple op node. 1 Input, 1 Output.
class OpNode : public INode {
public:
  OpNode(std::string uid, std::unique_ptr<Layer> layer)
      : INode(uid),
        layer_(std::move(layer)) {}

  void init() { layer_->init(); }

  Vec<ParamDescriptor> param_descriptors() const { return layer_->param_descriptors(); }

  Vec<Tensor> forward(const Vec<ConstTensor> &inputs, size_t mb_id = 0) {
    return layer_->forward(inputs, mb_id);
  }

  Vec<Tensor> backward(const Vec<ConstTensor> &gradients, size_t mb_id = 0) {
    return layer_->backward(gradients, mb_id);
  }

  void set_seed(size_t seed) { layer_->set_seed(seed); }

  void set_training(bool training) { layer_->set_training(training); }

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
    return layer_->output_shapes(input_shapes);
  }

  Vec<Tensor> parameters() { return layer_->parameters(); }
  Vec<Tensor> gradients() { return layer_->gradients(); }

  std::string type() const override { return "op_node"; }
  void save_state(std::ofstream &file) override {}
  NodeConfig get_config() const override;

  static OpNode create_from_config(const NodeConfig &config);

  const Device &device() const { return layer_->device(); }

  Layer *layer() { return layer_.get(); }
  const Layer *layer() const { return layer_.get(); }

  std::unique_ptr<Layer> release_layer() { return std::move(layer_); }

private:
  std::unique_ptr<Layer> layer_;
};

}  // namespace tnn