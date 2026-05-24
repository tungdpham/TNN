#include <cstddef>

#include "nn/layer.hpp"
#include "type/type.hpp"

namespace tnn {
class IdentityLayer : public Layer {
private:
  void init_impl() override {
    // no-op
  }

  Vec<Tensor> forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id = 0) override {
    Vec<Tensor> outputs;
    outputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      Tensor output = get_output_tensor(inputs[i]->shape());
      output->share_from(inputs[i]);
      outputs.push_back(output);
    }
    return outputs;
  }

  Vec<Tensor> backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id = 0) override {
    Vec<Tensor> grad_inputs;
    grad_inputs.reserve(grad_outputs.size());
    for (size_t i = 0; i < grad_outputs.size(); ++i) {
      Tensor grad_input = get_output_tensor(grad_outputs[i]->shape());
      grad_input->share_from(grad_outputs[i]);
      grad_inputs.push_back(grad_input);
    }
    return grad_inputs;
  }

public:
  IdentityLayer(const std::string &name = "identity");

  static constexpr const char *TYPE_NAME = "identity";

  std::string type() const override { return TYPE_NAME; }
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override {
    return input_shapes;
  }
  Vec<ParamDescriptor> param_descriptors() override { return {}; }
  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = name();
    config.type = TYPE_NAME;
    return config;
  }
  static std::unique_ptr<IdentityLayer> create_from_config(const LayerConfig &config) {
    return std::make_unique<IdentityLayer>(config.name);
  }
};
}  // namespace tnn