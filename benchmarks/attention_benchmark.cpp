#include "device/device_manager.hpp"
#include "device/flow.hpp"
#include "device/pool_allocator.hpp"
#include "nn/blocks_impl/attention_block.hpp"
#include "nn/blocks_impl/flash_attention_block.hpp"
#include "tensor/tensor.hpp"

using namespace synet;
using namespace std;

constexpr size_t BATCH_SIZE = 16;
constexpr size_t SEQ_LEN = 512;
constexpr size_t EMBED_DIM = 768;

signed main() {
  auto &device = getGPU();
  Graph graph;

  auto input = graph.make_node("input");

  auto attention_block = AttentionBlock(EMBED_DIM, 8, true, "attention_test");
  auto output_vanilla = attention_block(input);

  auto flash_block = FlashAttentionBlock(EMBED_DIM, 8, true, "flash_attention_test");
  auto output_flash = flash_block(input);

  graph.compile(PoolAllocator::instance(device, defaultFlowHandle));

  Vec<ParamDescriptor> attn_params = attention_block.param_descriptors();
  Vec<ParamDescriptor> flash_attn_params = flash_block.param_descriptors();
  for (size_t i = 0; i < attn_params.size(); ++i) {
    attn_params[i].data_ptr->copy_to(*flash_attn_params[i].data_ptr);
  }
  Tensor input_data = Tensor({BATCH_SIZE, SEQ_LEN, EMBED_DIM}, DType_t::FP32, getGPU());
  input_data.fill_random_normal(0.5f, 0.2f, 676767);

  Residuals residuals;
  Residuals legacy_residuals;

  // cold pass
  Tensor full_attn_output = attention_block.forward({input_data}, residuals)[0];
  Tensor flash_attn_output = flash_block.forward({input_data}, legacy_residuals)[0];

  for (int i = 0; i < 10; ++i) {
    auto vanilla_start = std::chrono::high_resolution_clock::now();
    full_attn_output = attention_block.forward({input_data}, residuals)[0];
    attention_block.device().getFlow(defaultFlowHandle)->synchronize();
    auto vanilla_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> vanilla_duration = vanilla_end - vanilla_start;
    printf("Vanilla Attention Forward Pass Time: %.3f ms\n", vanilla_duration.count());
  }

  for (int i = 0; i < 10; ++i) {
    auto flash_start = std::chrono::high_resolution_clock::now();
    flash_attn_output = flash_block.forward({input_data}, legacy_residuals)[0];
    flash_block.device().getFlow(defaultFlowHandle)->synchronize();
    auto flash_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> flash_duration = flash_end - flash_start;
    printf("Flash Attention Forward Pass Time: %.3f ms\n", flash_duration.count());
  }

  auto cpu_full_attn_output = full_attn_output.to_host();
  auto cpu_flash_attn_output = flash_attn_output.to_host();
  float *full_attn_data = static_cast<float *>(cpu_full_attn_output.data_as<void>());
  float *flash_attn_data = static_cast<float *>(cpu_flash_attn_output.data_as<void>());

  int mismatch_count = 0;
  for (size_t i = 0; i < full_attn_output.size(); ++i) {
    if (abs((float)(full_attn_data[i] - flash_attn_data[i])) > 1e-3) {
      printf("Mismatch at index %zu: full %f vs flash %f\n", i, (float)full_attn_data[i],
             (float)flash_attn_data[i]);
      ++mismatch_count;
    }
  }
  printf("Mismatch count: %d\n", mismatch_count);
  return 0;
}
