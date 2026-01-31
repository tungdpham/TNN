#include "device/device_manager.hpp"
#include "nn/example_models.hpp"
#include "nn/layers.hpp"
#include "nn/layers_impl/dense_layer.hpp"
#include "type/type.hpp"

using namespace std;
using namespace tnn;

signed main() {
  ExampleModels::register_defaults();

  DenseLayer fp32_dense(128, 64, false, "fp32_dense");
  fp32_dense.set_io_dtype(DType_t::FP32);
  fp32_dense.set_device(getGPU());
  fp32_dense.init();

  DenseLayer fp16_dense(128, 64, false, "fp16_dense");
  fp16_dense.set_io_dtype(DType_t::FP16);
  fp16_dense.set_param_dtype(DType_t::FP16);
  fp16_dense.set_device(getGPU());
  fp16_dense.init();

  auto fp16_params = fp16_dense.parameters();
  auto fp32_params = fp32_dense.parameters();
  for (size_t i = 0; i < fp16_params.size(); ++i) {
    Tensor cpu_fp16_param = fp16_params[i]->to_cpu();
    Tensor cpu_fp32_param = fp32_params[i]->to_cpu();
    fp16 *fp16_data = cpu_fp16_param->data_as<fp16>();
    float *fp32_data = cpu_fp32_param->data_as<float>();
    for (size_t j = 0; j < cpu_fp16_param->size(); ++j) {
      fp32_data[j] = static_cast<float>(fp16_data[j]);
    }
    cpu_fp32_param->copy_to(fp32_params[i]);
  }

  Tensor fp16_input = Tensor::create(DType_t::FP16, {32, 128}, getCPU());
  fp16_input->fill_random_uniform(0.0f, 1.0f);
  Tensor fp32_input = Tensor::create(DType_t::FP32, {32, 128}, getCPU());

  fp16 *input_data = fp16_input->data_as<fp16>();
  fp32 *input_data_fp32 = fp32_input->data_as<float>();
  for (size_t i = 0; i < fp16_input->size(); ++i) {
    input_data_fp32[i] = static_cast<float>(input_data[i]);
  }

  Tensor input_fp32 = fp32_input->to_device(getGPU());
  Tensor input_fp16 = fp16_input->to_device(getGPU());

  Tensor output_fp32, output_fp16;
  output_fp32 = Tensor::create(DType_t::FP32, {32, 64}, getGPU());
  output_fp16 = Tensor::create(DType_t::FP16, {32, 64}, getGPU());

  fp32_dense.forward(input_fp32, output_fp32, 0);
  fp16_dense.forward(input_fp16, output_fp16, 0);

  Tensor cpu_output_fp32 = output_fp32->to_cpu();
  Tensor cpu_output_fp16 = output_fp16->to_cpu();

  float *output_data_fp32 = cpu_output_fp32->data_as<float>();
  fp16 *output_data_fp16 = cpu_output_fp16->data_as<fp16>();
  double max_diff = 0.0;
  constexpr double tolerance = 1e-4;
  for (size_t i = 0; i < cpu_output_fp32->size(); ++i) {
    double val_fp32 = static_cast<double>(output_data_fp32[i]);
    double val_fp16 = static_cast<double>(output_data_fp16[i]);
    double diff = std::abs(val_fp32 - val_fp16);
    if (diff > tolerance) {
      if (diff > max_diff) {
        max_diff = diff;
      }
      std::cout << "At index " << i << ": FP32 value = " << val_fp32
                << ", FP16 value = " << val_fp16 << ", diff = " << diff << std::endl;
    }
  }
  cout << "Max diff: " << max_diff << endl;
  return 0;
}