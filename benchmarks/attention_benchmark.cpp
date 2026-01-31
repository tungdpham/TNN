#include "device/device_manager.hpp"
#include "nn/blocks_impl/attention_block.hpp"
#include "nn/blocks_impl/flash_attention_block.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;
using namespace std;

constexpr size_t BATCH_SIZE = 16;
constexpr size_t SEQ_LEN = 512;
constexpr size_t EMBED_DIM = 768;

signed main() {
  AttentionBlock attention_block(EMBED_DIM, 8, true, "attention_test");
  attention_block.set_device(getGPU());
  attention_block.init();

  FlashAttentionBlock flash_attention_block(EMBED_DIM, 8, true, "flash_attention_test");
  flash_attention_block.set_device(getGPU());
  flash_attention_block.init();

  auto attn_params = attention_block.parameters();
  auto flash_attn_params = flash_attention_block.parameters();
  for (size_t i = 0; i < attn_params.size(); ++i) {
    attn_params[i]->copy_to(flash_attn_params[i]);
  }
  Tensor input = Tensor::create<float>({BATCH_SIZE, SEQ_LEN, EMBED_DIM}, getGPU());
  input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor full_attn_output = Tensor::create<float>({BATCH_SIZE, SEQ_LEN, EMBED_DIM}, getGPU());

  Tensor flash_attn_output = Tensor::create<float>({BATCH_SIZE, SEQ_LEN, EMBED_DIM}, getGPU());
  // cold pass
  attention_block.forward(input, full_attn_output);
  flash_attention_block.forward(input, flash_attn_output);

  for (int i = 0; i < 10; ++i) {
    auto vanilla_start = std::chrono::high_resolution_clock::now();
    attention_block.forward(input, full_attn_output);
    attention_block.get_device().getFlow("default")->synchronize();
    auto vanilla_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> vanilla_duration = vanilla_end - vanilla_start;
    printf("Vanilla Attention Forward Pass Time: %.3f ms\n", vanilla_duration.count());
  }

  for (int i = 0; i < 10; ++i) {
    auto flash_start = std::chrono::high_resolution_clock::now();
    flash_attention_block.forward(input, flash_attn_output);
    flash_attention_block.get_device().getFlow("default")->synchronize();
    auto flash_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> flash_duration = flash_end - flash_start;
    printf("Flash Attention Forward Pass Time: %.3f ms\n", flash_duration.count());
  }

  auto cpu_full_attn_output = full_attn_output->to_cpu();
  auto cpu_flash_attn_output = flash_attn_output->to_cpu();
  float *full_attn_data = static_cast<float *>(cpu_full_attn_output->data());
  float *flash_attn_data = static_cast<float *>(cpu_flash_attn_output->data());

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
