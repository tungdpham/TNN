#include <cstddef>

#include "nn/layer.hpp"
#include "type/type.hpp"

namespace tnn {
class IdentityLayer : public Layer {
private:
  void init_impl() override {
    // no-op
  }

  void forward_impl(const Vec<ConstTensor> &inputs, const Vec<Tensor> &outputs,
                    size_t mb_id = 0) override {
    for (size_t i = 0; i < inputs.size(); ++i) {
      outputs[i]->share_from(inputs[i]);
    }
  }

  void backward_impl(const Vec<ConstTensor> &grad_outputs, const Vec<Tensor> &grad_inputs,
                     size_t mb_id = 0) override {
    for (size_t i = 0; i < grad_outputs.size(); ++i) {
      grad_inputs[i]->share_from(grad_outputs[i]);
    }
  }

public:
  IdentityLayer(const std::string &name = "identity");

  static constexpr const char *TYPE_NAME = "identity";

  std::string type() const override { return TYPE_NAME; }
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override {
    return input_shapes;
  }
  std::vector<ParamDescriptor> param_descriptors() override { return {}; }
  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = name();
    config.type = TYPE_NAME;
    return config;
  }
  size_t fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const override { return 0; }
  size_t fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override { return 0; }
  size_t inf_workspace(const Vec<Vec<size_t>> &input_shapes) const override { return 0; }
  size_t bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override { return 0; }
  static std::unique_ptr<IdentityLayer> create_from_config(const LayerConfig &config) {
    return std::make_unique<IdentityLayer>(config.name);
  }
};
}  // namespace tnn