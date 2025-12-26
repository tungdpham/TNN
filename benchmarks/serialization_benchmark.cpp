#include "pipeline/message.hpp"

#include "pipeline/binary_serializer.hpp"
#include "pipeline/buffer_pool.hpp"
#include "pipeline/job.hpp"
#include "pipeline/tbuffer.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_wrapper.hpp"
#include <cassert>

using namespace tnn;

constexpr size_t microbatch_id = 2;

signed main() {
  Tensor<float> tensor({128, 512, 16, 16});
  Job<float> job(std::move(tensor), microbatch_id);
  Message message("coordinator", CommandType::FORWARD_JOB, std::move(job));

  ThreadWrapper thread_wrapper({16});

  thread_wrapper.execute([&]() -> void {
    // Serialize the message
    PooledBuffer pooled_buffer = BufferPool::instance().get_buffer(message.size());
    TBuffer &buffer = *pooled_buffer;
    buffer.fill(0);
    auto serialization_start = std::chrono::high_resolution_clock::now();
    BinarySerializer::serialize(message, buffer);
    auto serialization_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> serialization_duration =
        serialization_end - serialization_start;
    std::cout << "Serialization took " << serialization_duration.count() << " ms" << std::endl;

    // Deserialize the message
    Message deserialized_message;
    auto deserialization_start = std::chrono::high_resolution_clock::now();
    size_t offset = 0;
    BinarySerializer::deserialize(buffer, offset, deserialized_message);
    auto deserialization_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> deserialization_duration =
        deserialization_end - deserialization_start;
    std::cout << "Deserialization took " << deserialization_duration.count() << " ms" << std::endl;
  });
  return 0;
}