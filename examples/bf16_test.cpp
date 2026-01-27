#include "device/device_manager.hpp"
#include "nn/blocks_impl/attention_block.hpp"
#include "nn/example_models.hpp"
#include "nn/layers.hpp"
#include "nn/layers_impl/dense_layer.hpp"
#include "nn/loss.hpp"
#include "type/type.hpp"
#include <cstddef>

using namespace std;
using namespace tnn;

void test_dense() {
  constexpr size_t batch_size = 8;
  constexpr size_t input_dim = 32;
  constexpr size_t output_dim = 16;
  DenseLayer fp32_dense(input_dim, output_dim, false, "fp32_dense");
  fp32_dense.set_io_dtype(DType_t::FP32);
  fp32_dense.set_device(getGPU());
  fp32_dense.init();

  DenseLayer bf16_dense(input_dim, output_dim, false, "bf16_dense");
  bf16_dense.set_io_dtype(DType_t::BF16);
  bf16_dense.set_device(getGPU());
  bf16_dense.init();

  auto bf16_params = bf16_dense.parameters();
  auto fp32_params = fp32_dense.parameters();
  for (size_t i = 0; i < bf16_params.size(); ++i) {
    bf16_params[i]->copy_to(fp32_params[i]);
  }

  Tensor bf16_input = Tensor::create(DType_t::BF16, {batch_size, input_dim}, &getCPU());
  bf16_input->fill_random_uniform(0.0f, 1.0f);
  Tensor fp32_input = Tensor::create(DType_t::FP32, {batch_size, input_dim}, &getCPU());

  bf16 *input_data = bf16_input->data_as<bf16>();
  fp32 *input_data_fp32 = fp32_input->data_as<float>();
  for (size_t i = 0; i < bf16_input->size(); ++i) {
    input_data_fp32[i] = static_cast<float>(input_data[i]);
  }

  Tensor input_fp32 = fp32_input->to_device(&getGPU());
  Tensor input_bf16 = bf16_input->to_device(&getGPU());

  Tensor output_fp32, output_bf16;
  output_fp32 = Tensor::create(DType_t::FP32, {batch_size, output_dim}, &getGPU());
  output_bf16 = Tensor::create(DType_t::BF16, {batch_size, output_dim}, &getGPU());

  fp32_dense.forward(input_fp32, output_fp32, 0);
  bf16_dense.forward(input_bf16, output_bf16, 0);

  Tensor cpu_output_fp32 = output_fp32->to_cpu();
  Tensor cpu_output_bf16 = output_bf16->to_cpu();

  float *output_data_fp32 = cpu_output_fp32->data_as<float>();
  bf16 *output_data_bf16 = cpu_output_bf16->data_as<bf16>();
  double max_diff = 0.0;
  constexpr double tolerance = 2e-3;
  for (size_t i = 0; i < cpu_output_fp32->size(); ++i) {
    double val_fp32 = static_cast<double>(output_data_fp32[i]);
    double val_bf16 = static_cast<double>(output_data_bf16[i]);
    double diff = std::abs(val_fp32 - val_bf16);
    if (diff > tolerance) {
      if (diff > max_diff) {
        max_diff = diff;
      }
      std::cout << "At index " << i << ": FP32 value = " << val_fp32
                << ", BF16 value = " << val_bf16 << ", diff = " << diff << std::endl;
    }
  }
  cout << "Max diff: " << max_diff << endl;

  Tensor target_fp32 = Tensor::create(DType_t::FP32, {batch_size, output_dim});
  Tensor target_bf16 = Tensor::create(DType_t::BF16, {batch_size, output_dim});
  target_fp32->fill(0.0f);
  target_bf16->fill(bf16(0.0f));

  for (size_t i = 0; i < batch_size; ++i) {
    target_fp32->at<float>({i, i % output_dim}) = 1.0f;
    target_bf16->at<bf16>({i, i % output_dim}) = bf16(1.0f);
  }

  auto criterion = LossFactory::create_logsoftmax_crossentropy();

  auto gradient_fp32 = Tensor::create(DType_t::FP32, {batch_size, output_dim});
  auto gradient_bf16 = Tensor::create(DType_t::BF16, {batch_size, output_dim});

  criterion->compute_gradient(cpu_output_fp32, target_fp32, gradient_fp32);
  criterion->compute_gradient(cpu_output_bf16, target_bf16, gradient_bf16);

  auto gpu_gradient_fp32 = gradient_fp32->to_device(&getGPU());
  auto gpu_gradient_bf16 = gradient_bf16->to_device(&getGPU());

  Tensor grad_input_bf16 = Tensor::create(DType_t::BF16, {batch_size, input_dim}, &getGPU());
  Tensor grad_input_fp32 = Tensor::create(DType_t::FP32, {batch_size, input_dim}, &getGPU());

  bf16_dense.backward(gpu_gradient_bf16, grad_input_bf16, 0);
  fp32_dense.backward(gpu_gradient_fp32, grad_input_fp32, 0);

  Tensor cpu_grad_input_fp32 = grad_input_fp32->to_cpu();
  Tensor cpu_grad_input_bf16 = grad_input_bf16->to_cpu();
  float *grad_input_data_fp32 = cpu_grad_input_fp32->data_as<float>();
  bf16 *grad_input_data_bf16 = cpu_grad_input_bf16->data_as<bf16>();
  max_diff = 0.0;
  for (size_t i = 0; i < cpu_grad_input_fp32->size(); ++i) {
    double val_fp32 = static_cast<double>(grad_input_data_fp32[i]);
    double val_bf16 = static_cast<double>(grad_input_data_bf16[i]);
    double diff = std::abs(val_fp32 - val_bf16);
    if (diff > tolerance) {
      if (diff > max_diff) {
        max_diff = diff;
      }
      std::cout << "At index " << i << ": FP32 grad value = " << val_fp32
                << ", BF16 grad value = " << val_bf16 << ", diff = " << diff << std::endl;
    }
  }
  cout << "Max grad diff: " << max_diff << endl;
}

void test_attention() {
  constexpr size_t batch_size = 8;
  constexpr size_t seq_len = 16;
  constexpr size_t embed_dim = 16;
  constexpr size_t num_heads = 4;
  AttentionBlock fp32_attention(embed_dim, num_heads, false, "fp32_attention");
  fp32_attention.set_io_dtype(DType_t::FP32);
  fp32_attention.set_device(getGPU());
  fp32_attention.init();

  AttentionBlock bf16_attention(embed_dim, num_heads, false, "bf16_attention");
  bf16_attention.set_io_dtype(DType_t::BF16);
  bf16_attention.set_param_dtype(DType_t::BF16);
  bf16_attention.set_device(getGPU());
  bf16_attention.init();

  auto bf16_params = bf16_attention.parameters();
  auto fp32_params = fp32_attention.parameters();
  for (size_t i = 0; i < bf16_params.size(); ++i) {
    Tensor cpu_bf16_param = bf16_params[i]->to_cpu();
    Tensor cpu_fp32_param = fp32_params[i]->to_cpu();
    bf16 *bf16_data = cpu_bf16_param->data_as<bf16>();
    float *fp32_data = cpu_fp32_param->data_as<float>();
    for (size_t j = 0; j < cpu_bf16_param->size(); ++j) {
      fp32_data[j] = static_cast<float>(bf16_data[j]);
    }
    cpu_fp32_param->copy_to(fp32_params[i]);
  }

  Tensor bf16_input = Tensor::create(DType_t::BF16, {batch_size, seq_len, embed_dim}, &getCPU());
  bf16_input->fill_random_uniform(0.0f, 1.0f);
  Tensor fp32_input = Tensor::create(DType_t::FP32, {batch_size, seq_len, embed_dim}, &getCPU());

  bf16 *input_data = bf16_input->data_as<bf16>();
  fp32 *input_data_fp32 = fp32_input->data_as<float>();
  for (size_t i = 0; i < bf16_input->size(); ++i) {
    input_data_fp32[i] = static_cast<float>(input_data[i]);
  }

  Tensor input_fp32 = fp32_input->to_device(&getGPU());
  Tensor input_bf16 = bf16_input->to_device(&getGPU());

  Tensor output_fp32, output_bf16;
  output_fp32 = Tensor::create(DType_t::FP32, {batch_size, seq_len, embed_dim}, &getGPU());
  output_bf16 = Tensor::create(DType_t::BF16, {batch_size, seq_len, embed_dim}, &getGPU());

  fp32_attention.forward(input_fp32, output_fp32, 0);
  bf16_attention.forward(input_bf16, output_bf16, 0);

  Tensor cpu_output_fp32 = output_fp32->to_cpu();
  Tensor cpu_output_bf16 = output_bf16->to_cpu();

  float *output_data_fp32 = cpu_output_fp32->data_as<float>();
  bf16 *output_data_bf16 = cpu_output_bf16->data_as<bf16>();
  double max_diff = 0.0;
  constexpr double tolerance = 2e-3;
  for (size_t i = 0; i < cpu_output_fp32->size(); ++i) {
    double val_fp32 = static_cast<double>(output_data_fp32[i]);
    double val_bf16 = static_cast<double>(output_data_bf16[i]);
    double diff = std::abs(val_fp32 - val_bf16);
    if (diff > tolerance) {
      if (diff > max_diff) {
        max_diff = diff;
      }
      std::cout << "At index " << i << ": FP32 value = " << val_fp32
                << ", BF16 value = " << val_bf16 << ", diff = " << diff << std::endl;
    }
  }
  cout << "Max diff: " << max_diff << endl;

  Tensor target_fp32 = Tensor::create(DType_t::FP32, {batch_size, seq_len, embed_dim});
  Tensor target_bf16 = Tensor::create(DType_t::BF16, {batch_size, seq_len, embed_dim});
  target_fp32->fill(0.0f);
  target_bf16->fill(bf16(0.0f));

  for (size_t i = 0; i < 32; ++i) {
    target_fp32->at<float>({i, i % 16, i / 16}) = 1.0f;
    target_bf16->at<bf16>({i, i % 16, i / 16}) = bf16(1.0f);
  }

  auto criterion = LossFactory::create_logsoftmax_crossentropy();

  auto gradient_fp32 = Tensor::create(DType_t::FP32, {batch_size, seq_len, embed_dim});
  auto gradient_bf16 = Tensor::create(DType_t::BF16, {batch_size, seq_len, embed_dim});

  criterion->compute_gradient(cpu_output_fp32, target_fp32, gradient_fp32);
  criterion->compute_gradient(cpu_output_bf16, target_bf16, gradient_bf16);

  auto gpu_gradient_fp32 = gradient_fp32->to_device(&getGPU());
  auto gpu_gradient_bf16 = gradient_bf16->to_device(&getGPU());

  Tensor grad_input_bf16 =
      Tensor::create(DType_t::BF16, {batch_size, seq_len, embed_dim}, &getGPU());
  Tensor grad_input_fp32 =
      Tensor::create(DType_t::FP32, {batch_size, seq_len, embed_dim}, &getGPU());

  bf16_attention.backward(gpu_gradient_bf16, grad_input_bf16, 0);
  fp32_attention.backward(gpu_gradient_fp32, grad_input_fp32, 0);

  Tensor cpu_grad_input_fp32 = grad_input_fp32->to_cpu();
  Tensor cpu_grad_input_bf16 = grad_input_bf16->to_cpu();
  float *grad_input_data_fp32 = cpu_grad_input_fp32->data_as<float>();
  bf16 *grad_input_data_bf16 = cpu_grad_input_bf16->data_as<bf16>();
  max_diff = 0.0;
  for (size_t i = 0; i < cpu_grad_input_fp32->size(); ++i) {
    double val_fp32 = static_cast<double>(grad_input_data_fp32[i]);
    double val_bf16 = static_cast<double>(grad_input_data_bf16[i]);
    double diff = std::abs(val_fp32 - val_bf16);
    if (diff > tolerance) {
      if (diff > max_diff) {
        max_diff = diff;
      }
      std::cout << "At index " << i << ": FP32 grad value = " << val_fp32
                << ", BF16 grad value = " << val_bf16 << ", diff = " << diff << std::endl;
    }
  }
  cout << "Max grad diff: " << max_diff << endl;
}

signed main() {
  ExampleModels::register_defaults();

  test_dense();

  std::cout << std::endl;

  test_attention();

  return 0;
}