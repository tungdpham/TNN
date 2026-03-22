#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "device/device_manager.hpp"
#include "device/flow.hpp"
#include "device/pool_allocator.hpp"
#include "distributed/binary_serializer.hpp"
#include "distributed/job.hpp"
#include "distributed/message.hpp"
#include "tensor/tensor.hpp"
#include "utils/misc.hpp"

using namespace tnn;

constexpr size_t microbatch_id = 2;
constexpr size_t data_size = 256 * 512 * 16 * 16;

signed main() {
  Tensor tensor = make_tensor<float>({data_size});
  tensor->fill_random_normal(0.0, 0.5);
  Job job;
  job.mb_id = microbatch_id;
  job.data = tensor->clone();
  Message message(CommandType::FORWARD_JOB, std::move(job));

  auto &device_allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);

  BinarySerializer bserializer(device_allocator);
  dptr buffer = device_allocator.allocate(data_size * sizeof(float) + 1024);

  Vec<float> raw_data(data_size, 1.2345f);
  Tensor temp = make_tensor<float>({data_size});

  benchmark(
      "BinarySerializer - Serialization",
      [&]() {
        Writer writer(buffer);
        bserializer.serialize(writer, message);
      },
      10);

  benchmark(
      "BinarySerializer - Deserialization",
      [&]() {
        // size_t deserialize_offset = 0
        Reader reader(buffer);
        bserializer.deserialize(reader, message);
      },
      10);

  // verify integrity
  {
    Sizer sizer;
    sizer(message);
    size_t msg_size = sizer.size();
    std::cout << msg_size << " vs " << buffer.capacity() << std::endl;
    auto &deserialized_job = message.get<Job>();
    auto &deserialized_tensor = deserialized_job.data;
    std::cout << "Deserialized tensor size: " << deserialized_tensor->size() << std::endl;
    assert(deserialized_tensor->size() == data_size);
    std::cout << "Deserialized tensor dims: " << deserialized_tensor->dims() << std::endl;
    assert(deserialized_tensor->dims() == 1);
    std::cout << "Deserialized tensor dtype: "
              << static_cast<uint32_t>(deserialized_tensor->data_type()) << std::endl;
    assert(deserialized_tensor->data_type() == DType_t::FP32);
    for (size_t i = 0; i < data_size; i++) {
      assert(std::abs(tensor->at<float>({i}) - deserialized_tensor->at<float>({i})) < 1e-6);
    }
  }

  benchmark(
      "Raw Memory Copy",
      [&]() {
        std::memcpy(raw_data.data(), reinterpret_cast<uint8_t *>(temp->data()),
                    data_size * sizeof(float));
      },
      10);

  return 0;
}