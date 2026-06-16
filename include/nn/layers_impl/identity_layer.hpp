#include <cstddef>

#include "nn/layer.hpp"
#include "type/type.hpp"

namespace synet {
class IdentityLayerImpl : public LayerImpl {
private:
  void init_impl() override {
    // no-op
  }

  Vec<Tensor> forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) override {
    Vec<Tensor> outputs;
    outputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      Tensor output = get_tensor(inputs[i].shape(), inputs[i].data_type());
      output = inputs[i];
      outputs.push_back(output);
    }
    return outputs;
  }

  Vec<Tensor> backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) override {
    Vec<Tensor> grad_inputs;
    grad_inputs.reserve(grad_outputs.size());
    for (size_t i = 0; i < grad_outputs.size(); ++i) {
      Tensor grad_input = get_tensor(grad_outputs[i].shape(), grad_outputs[i].data_type());
      grad_input = grad_outputs[i];
      grad_inputs.push_back(grad_input);
    }
    return grad_inputs;
  }

public:
  IdentityLayerImpl(const std::string &name = "identity");

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
  static std::shared_ptr<IdentityLayerImpl> create_from_config(const LayerConfig &config) {
    return std::make_shared<IdentityLayerImpl>(config.name);
  }
};

class IdentityLayer : public LayerRef<IdentityLayerImpl> {
public:
  explicit IdentityLayer(const std::string &name = "identity")
      : LayerRef(std::make_shared<IdentityLayerImpl>(name)) {}

  using LayerRef<IdentityLayerImpl>::LayerRef;
};
}  // namespace synet