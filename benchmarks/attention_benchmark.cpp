#include "device/device_manager.hpp"
#include "device/flow.hpp"
#include "device/pool_allocator.hpp"
#include "nn/blocks_impl/flash_attention_block.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"

using namespace tunx;
using namespace std;

constexpr size_t BATCH_SIZE = 16;
constexpr size_t SEQ_LEN = 512;
constexpr size_t EMBED_DIM = 768;

signed main() {
  auto &device = getGPU();
  Graph graph;

  auto input = graph.make_node("input");

  auto flash_block = FlashAttentionBlock(EMBED_DIM, 8, true, "flash_attention_test");
  auto output_flash = flash_block(input);

  graph.compile(PoolAllocator::instance(device, defaultFlowHandle));

  Tensor input_data = Tensor({BATCH_SIZE, SEQ_LEN, EMBED_DIM}, DType_t::FP32, getGPU());
  fill_normal(input_data, 0.5f, 0.2f, 676767);

  Residuals legacy_residuals;

  Tensor flash_attn_output = flash_block.forward({input_data}, legacy_residuals)[0];

  for (int i = 0; i < 10; ++i) {
    auto flash_start = std::chrono::high_resolution_clock::now();
    flash_attn_output = flash_block.forward({input_data}, legacy_residuals)[0];
    flash_block.device().getFlow(defaultFlowHandle)->synchronize();
    auto flash_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> flash_duration = flash_end - flash_start;
    printf("Flash Attention Forward Pass Time: %.3f ms\n", flash_duration.count());
  }

  return 0;
}
