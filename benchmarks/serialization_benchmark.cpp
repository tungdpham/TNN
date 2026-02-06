#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "distributed/binary_serializer.hpp"
#include "distributed/job.hpp"
#include "distributed/message.hpp"
#include "distributed/tbuffer.hpp"
#include "distributed/vbuffer.hpp"
#include "distributed/vserializer.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"
#include "utils/misc.hpp"

using namespace tnn;

constexpr size_t microbatch_id = 2;
constexpr size_t data_size = 256 * 512 * 16 * 16;

signed main() {
  Tensor tensor = make_tensor<float>({data_size});
  Job job;
  job.mb_id = microbatch_id;
  job.data = tensor->clone();
  Message message(CommandType::FORWARD_JOB, std::move(job));

  auto &device_allocator = DeviceAllocator::instance(getCPU());

  BinarySerializer bserializer(device_allocator);
  TBuffer tbuffer(device_allocator, data_size * sizeof(float) + 1024);  // extra space for metadata

  std::vector<float> raw_data(data_size, 1.2345f);
  Tensor temp = make_tensor<float>({data_size});

  VSerializer vserializer(device_allocator);
  VBuffer vbuffer(device_allocator);
  vbuffer.alloc(device_allocator.allocate(message.size()));
  vbuffer.resize(message.size());

  benchmark(
      "BinarySerializer - Serialization",
      [&]() {
        size_t serialize_offset = 0;
        bserializer.serialize(tbuffer, serialize_offset, message);
      },
      10);

  benchmark(
      "BinarySerializer - Deserialization",
      [&]() {
        size_t deserialize_offset = 0;
        bserializer.deserialize(tbuffer, deserialize_offset, message);
      },
      10);

  // verify integrity
  {
    std::cout << message.size() << " vs " << tbuffer.size() << std::endl;
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

  benchmark(
      "VSerializer - Serialization",
      [&]() {
        size_t serialize_offset = 0;
        std::cout << "Message size: " << message.size() << std::endl;
        vserializer.serialize(vbuffer, serialize_offset, std::move(message));
        std::cout << "vbuffer capacity: " << vbuffer.capacity() << ", size: " << vbuffer.size()
                  << std::endl;
        size_t deserialize_offset = 0;
        vserializer.deserialize(vbuffer, deserialize_offset, message);
        auto deserialized_tensor = message.get<Job>().data;
        assert(deserialized_tensor->size() == data_size);
        assert(deserialized_tensor->dims() == 1);
        assert(deserialized_tensor->data_type() == DType_t::FP32);
        for (size_t i = 0; i < data_size; i++) {
          assert(std::abs(raw_data[i] - deserialized_tensor->at<float>({i})) < 1e-6);
        }
      },
      1);

  return 0;
}