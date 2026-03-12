#include "device/device_manager.hpp"
#include "device/flow.hpp"
#include "device/pool_allocator.hpp"
#include "nn/blocks_impl/attention_block.hpp"
#include "nn/blocks_impl/flash_attention_block.hpp"
#include "nn/graph.hpp"
#include "nn/graph_builder.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;
using namespace std;

constexpr size_t BATCH_SIZE = 16;
constexpr size_t SEQ_LEN = 512;
constexpr size_t EMBED_DIM = 768;

signed main() {
  auto &allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  GraphBuilder builder;
  auto attention_block = make_unique<AttentionBlock>(EMBED_DIM, 8, true, "attention_test");
  auto &attention_op = builder.add_layer(std::move(attention_block));

  auto flash_block = make_unique<FlashAttentionBlock>(EMBED_DIM, 8, true, "flash_attention_test");
  auto &flash_op = builder.add_layer(std::move(flash_block));

  Graph graph = builder.compile(allocator);

  auto attn_params = attention_op.parameters();
  auto flash_attn_params = flash_op.parameters();
  for (size_t i = 0; i < attn_params.size(); ++i) {
    attn_params[i]->copy_to(flash_attn_params[i]);
  }
  Tensor input = make_tensor<float>({BATCH_SIZE, SEQ_LEN, EMBED_DIM}, getGPU());
  input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor full_attn_output = make_tensor<float>({BATCH_SIZE, SEQ_LEN, EMBED_DIM}, getGPU());

  Tensor flash_attn_output = make_tensor<float>({BATCH_SIZE, SEQ_LEN, EMBED_DIM}, getGPU());
  // cold pass
  attention_op.forward({input}, {full_attn_output});
  flash_op.forward({input}, {flash_attn_output});

  for (int i = 0; i < 10; ++i) {
    auto vanilla_start = std::chrono::high_resolution_clock::now();
    attention_op.forward({input}, {full_attn_output});
    attention_op.device().getFlow(defaultFlowHandle)->synchronize();
    auto vanilla_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> vanilla_duration = vanilla_end - vanilla_start;
    printf("Vanilla Attention Forward Pass Time: %.3f ms\n", vanilla_duration.count());
  }

  for (int i = 0; i < 10; ++i) {
    auto flash_start = std::chrono::high_resolution_clock::now();
    flash_op.forward({input}, {flash_attn_output});
    flash_op.device().getFlow(defaultFlowHandle)->synchronize();
    auto flash_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> flash_duration = flash_end - flash_start;
    printf("Flash Attention Forward Pass Time: %.3f ms\n", flash_duration.count());
  }

  auto cpu_full_attn_output = full_attn_output->to_host();
  auto cpu_flash_attn_output = flash_attn_output->to_host();
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
