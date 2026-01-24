#include "device/device_manager.hpp"
#include "nn/blocks_impl/attention_block.hpp"
#include "nn/blocks_impl/flash_attention_block.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

using namespace tnn;
using namespace std;

signed main() {
  AttentionBlock attention_block(512, 8, true, "attention_test");
  attention_block.set_io_dtype(DType_t::FP16);
  attention_block.set_device(getGPU());
  attention_block.init();

  FlashAttentionBlock flash_attention_block(512, 8, true, "flash_attention_test");
  flash_attention_block.set_io_dtype(DType_t::FP16);
  flash_attention_block.set_device(getGPU());
  flash_attention_block.init();

  auto attn_params = attention_block.parameters();
  auto flash_attn_params = flash_attention_block.parameters();
  for (size_t i = 0; i < attn_params.size(); ++i) {
    attn_params[i]->copy_to(flash_attn_params[i]);
  }
  Tensor input = make_tensor<fp16>({16, 128, 512}, &getGPU());
  input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor full_attn_output = make_tensor<fp16>({16, 128, 512}, &getGPU());

  Tensor flash_attn_output = make_tensor<fp16>({16, 128, 512}, &getGPU());
  attention_block.forward(input, full_attn_output);
  flash_attention_block.forward(input, flash_attn_output);

  auto cpu_full_attn_output = full_attn_output->to_cpu();
  auto cpu_flash_attn_output = flash_attn_output->to_cpu();
  fp16 *full_attn_data = static_cast<fp16 *>(cpu_full_attn_output->data());
  fp16 *flash_attn_data = static_cast<fp16 *>(cpu_flash_attn_output->data());

  int mismatch_count = 0;
  for (size_t i = 0; i < full_attn_output->size(); ++i) {
    if (abs((float)(full_attn_data[i] - flash_attn_data[i])) > 1e-3) {
      printf("Mismatch at index %zu: full %f vs flash %f\n", i, (float)full_attn_data[i],
             (float)flash_attn_data[i]);
      ++mismatch_count;
    }
  }
  printf("Mismatch count: %d\n", mismatch_count);
  return 0;
}
