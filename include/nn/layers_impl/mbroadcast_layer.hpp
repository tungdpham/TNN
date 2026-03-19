#include <cstddef>

#include "device/task.hpp"
#include "nn/layer.hpp"
#include "nn/layers_impl/cpu/n_ary_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/n_ary_ops.hpp"
#endif
#include "type/type.hpp"

namespace tnn {
class MBroadcastLayer : public Layer {
private:
  size_t m_ = 0;

  void init_impl() override {
    // no-op
  }

  void forward_impl(const Vec<ConstTensor> &inputs, const Vec<Tensor> &outputs,
                    size_t mb_id = 0) override {
    if (outputs.size() != m_) {
      throw std::runtime_error("MBroadcastLayer forward: number of outputs must match m");
    }
    // Broadcast single input to all outputs
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i]->share_from(inputs[0]);
    }
  }

  template <typename IO_T>
  void compute_backward(const Vec<ConstTensor> &grad_outputs, const Vec<Tensor> &grad_inputs) {
    const auto &output_shape = grad_outputs[0]->shape();

    std::vector<const IO_T *> grad_output_ptrs;
    for (const auto &grad_output : grad_outputs) {
      grad_output_ptrs.push_back(grad_output->data_as<IO_T>());
    }
    IO_T *grad_input_ptr = grad_inputs[0]->data_as<IO_T>();

    if (device().device_type() == DeviceType::CPU) {
      cpu::nary_forward<IO_T>(grad_output_ptrs, grad_input_ptr, output_shape, NAryOp::ADD);
    } else if (device().device_type() == DeviceType::GPU) {
#ifdef USE_CUDA
      Tensor ws = this->get_workspace({cuda::nary_forward_workspace_bytes(grad_output_ptrs.size())},
                                      DType_t::BYTE);
      create_cuda_task(flow_handle_, cuda::nary_forward<IO_T>, grad_output_ptrs, grad_input_ptr,
                       output_shape, NAryOp::ADD, ws->data());
#else
      throw std::runtime_error("MBroadcastLayer backward: CUDA support not compiled");
#endif
    } else {
      throw std::runtime_error("MBroadcastLayer backward: unsupported device type");
    }
  }

  void backward_impl(const Vec<ConstTensor> &grad_outputs, const Vec<Tensor> &grad_inputs,
                     size_t mb_id = 0) override {
    DISPATCH_IO_DTYPE(compute_backward, grad_outputs, grad_inputs);
  }

public:
  MBroadcastLayer(size_t m, const std::string &name = "mbroadcast")
      : Layer(name),
        m_(m) {}

  static constexpr const char *TYPE_NAME = "mbroadcast";

  std::string type() const override { return TYPE_NAME; }

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override {
    if (input_shapes.size() != 1) {
      throw std::runtime_error("MBroadcastLayer expects exactly one input shape");
    }
    Vec<Vec<size_t>> output_shapes(m_, input_shapes[0]);
    return output_shapes;
  }

  std::vector<ParamDescriptor> param_descriptors() override { return {}; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = name();
    config.type = TYPE_NAME;
    config.set("m", m_);
    return config;
  }

  static std::unique_ptr<MBroadcastLayer> create_from_config(const LayerConfig &config) {
    return std::make_unique<MBroadcastLayer>(config.get<size_t>("m"), config.name);
  }
};
}  // namespace tnn
