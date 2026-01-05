#include "distributed/message.hpp"

#include "distributed/binary_serializer.hpp"
#include "distributed/buffer_pool.hpp"
#include "distributed/job.hpp"
#include "distributed/tbuffer.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_wrapper.hpp"
#include <cassert>
#include <cstdint>
#include <cstdlib>

using namespace tnn;

constexpr size_t microbatch_id = 2;

signed main() {
  Tensor<float> tensor({256, 512, 16, 16});
  PooledJob<float> job = JobPool<float>::instance().get_job(tensor.size());
  job->micro_batch_id = microbatch_id;
  job->data = std::move(tensor);
  Message message("coordinator", CommandType::FORWARD_JOB, std::move(job));

  ThreadWrapper thread_wrapper({16});

  thread_wrapper.execute([&]() -> void {
    size_t data_size = 256 * 512 * 16 * 16 * sizeof(float);
    uint8_t *data_ptr = (uint8_t *)std::aligned_alloc(64, data_size);
    Tensor<float> temp({256, 512, 16, 16});
    for (int i = 0; i < 10; i++) {
      // Serialize the message
      PooledBuffer pooled_buffer = BufferPool::instance().get_buffer(message.size());
      TBuffer &buffer = *pooled_buffer;
      buffer.fill(0);
      auto serialization_start = std::chrono::high_resolution_clock::now();
      size_t serialize_offset = 0;
      BinarySerializer::serialize(buffer, serialize_offset, message);
      auto serialization_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> serialization_duration =
          serialization_end - serialization_start;
      std::cout << "Serialization took " << serialization_duration.count()
                << " ms for buffer size: " << buffer.size() << std::endl;

      // Deserialize the message
      Message deserialized_message;
      auto deserialization_start = std::chrono::high_resolution_clock::now();
      size_t deserialize_offset = 0;
      BinarySerializer::deserialize(buffer, deserialize_offset, deserialized_message);
      auto deserialization_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> deserialization_duration =
          deserialization_end - deserialization_start;
      std::cout << "Deserialization took " << deserialization_duration.count() << " ms"
                << std::endl;

      // Raw copy speed

      auto copy_start = std::chrono::high_resolution_clock::now();
      std::memcpy(data_ptr, reinterpret_cast<uint8_t *>(temp.data()), data_size);
      auto copy_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> copy_duration = copy_end - copy_start;
      std::cout << "Raw copy took " << copy_duration.count() << " ms for size: " << data_size
                << std::endl;
    }
    std::free(data_ptr);
  });
  return 0;
}