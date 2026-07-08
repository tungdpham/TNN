#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "common/archiver.hpp"
#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "device/iallocator.hpp"
#include "distributed/binary_serializer.hpp"
#include "distributed/chunking.hpp"
#include "distributed/io.hpp"
#include "distributed/message.hpp"
#include "distributed/packet.hpp"
#include "profiling/profiler.hpp"
#include "type/type.hpp"

using namespace tunx;

class ArchiverTest : public ::testing::Test {
public:
  static void SetUpTestSuite() { initializeDefaultDevices(); }

  size_t packet_header_size =
      sizeof(PacketHeader::PROTOCOL_VERSION) + sizeof(PacketHeader::type) +
      sizeof(PacketHeader::endianess) + sizeof(PacketHeader::packet_length) +
      sizeof(PacketHeader::msg_length) + sizeof(PacketHeader::msg_serial_id) +
      sizeof(PacketHeader::packet_offset) + sizeof(PacketHeader::total_packets) +
      sizeof(PacketHeader::compression_type);

  IAllocator& allocator_ = DeviceAllocator::instance(getHost());
};

namespace {

void expect_float_tensors_equal(const Tensor& actual, const Tensor& expected) {
  ASSERT_TRUE(actual);
  ASSERT_TRUE(expected);
  EXPECT_EQ(actual.shape(), expected.shape());
  EXPECT_EQ(actual.dtype(), expected.dtype());
  EXPECT_EQ(actual.size(), expected.size());

  const float* actual_data = actual.data_as<float>();
  const float* expected_data = expected.data_as<float>();
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(actual_data[i], expected_data[i]);
  }
}

}  // namespace

TEST_F(ArchiverTest, TestHeaderSizer) {
  PacketHeader header;
  header.PROTOCOL_VERSION = 1;
  header.type = PacketType::MSG_PREPARE;
  header.endianess = Endianness::LITTLE;
  header.packet_length = 1024;
  header.msg_length = 4096;
  header.msg_serial_id = 12345;
  header.packet_offset = 0;
  header.total_packets = 4;
  header.compression_type = CompressionType::ZSTD;
  Sizer sizer;
  sizer(header);
  EXPECT_EQ(sizer.size(), packet_header_size);
}

TEST_F(ArchiverTest, TestHeaderWriter) {
  dptr buffer = allocator_.allocate(packet_header_size);
  PacketHeader header;
  header.PROTOCOL_VERSION = 1;
  header.type = PacketType::MSG_PREPARE;
  header.endianess = Endianness::LITTLE;
  header.packet_length = 1024;
  header.msg_length = 4096;
  header.msg_serial_id = 12345;
  header.packet_offset = 0;
  header.total_packets = 4;
  header.compression_type = CompressionType::ZSTD;
  Writer writer(buffer);
  writer(header);
  EXPECT_EQ(writer.bytes_written(), packet_header_size);
}

TEST_F(ArchiverTest, TestBlobArchiver) {
  constexpr size_t blob_size = 4 * 1024 * 1024;          // 4 MB
  constexpr size_t byte_size = blob_size * sizeof(int);  // Total size of the buffer needed
  auto data = std::make_unique<int[]>(blob_size);
  dptr buffer = allocator_.allocate(byte_size);
  auto blob_data = make_blob(data.get(), blob_size);
  Sizer sizer;
  sizer(blob_data);
  EXPECT_EQ(sizer.size(), byte_size);
  Writer writer(buffer);
  writer(blob_data);
  EXPECT_EQ(writer.bytes_written(), byte_size);
}

TEST_F(ArchiverTest, TestHeaderArchiver) {
  dptr buffer = allocator_.allocate(packet_header_size);
  PacketHeader header;
  header.PROTOCOL_VERSION = 1;
  header.type = PacketType::MSG_PREPARE;
  header.endianess = Endianness::LITTLE;
  header.packet_length = 1024;
  header.msg_length = 4096;
  header.msg_serial_id = 12345;
  header.packet_offset = 0;
  header.total_packets = 4;
  header.compression_type = CompressionType::ZSTD;

  // First, write the header to the buffer using Writer
  Writer writer(buffer);
  writer(header);

  // Now read it back using Reader
  PacketHeader read_header;
  Reader in_archiver(buffer);
  in_archiver(read_header);

  EXPECT_EQ(in_archiver.bytes_read(), packet_header_size);
  EXPECT_EQ(read_header.PROTOCOL_VERSION, header.PROTOCOL_VERSION);
  EXPECT_EQ(read_header.type, header.type);
  EXPECT_EQ(read_header.endianess, header.endianess);
  EXPECT_EQ(read_header.packet_length, header.packet_length);
  EXPECT_EQ(read_header.msg_length, header.msg_length);
  EXPECT_EQ(read_header.msg_serial_id, header.msg_serial_id);
  EXPECT_EQ(read_header.packet_offset, header.packet_offset);
  EXPECT_EQ(read_header.total_packets, header.total_packets);
  EXPECT_EQ(read_header.compression_type, header.compression_type);
}

TEST_F(ArchiverTest, TestStringArchiver) {
  std::string original_str = "Hello World!";
  Sizer sizer;
  sizer(original_str);
  size_t expected_size = sizeof(uint64_t) + original_str.size();
  dptr buffer = allocator_.allocate(expected_size);
  Writer writer(buffer);
  writer(original_str);

  Reader reader(buffer);
  std::string deserialized_str;
  BinarySerializer bserializer(allocator_);
  bserializer.deserialize(reader, deserialized_str);

  EXPECT_EQ(deserialized_str, original_str);
}

TEST_F(ArchiverTest, TestBoolArchiver) {
  bool original_flag = true;
  Sizer sizer;
  sizer(original_flag);
  size_t expected_size = sizeof(bool);
  dptr buffer = allocator_.allocate(expected_size);
  Writer writer(buffer);
  writer(original_flag);

  Reader reader(buffer);
  bool deserialized_flag;
  BinarySerializer bserializer(allocator_);
  bserializer.deserialize(reader, deserialized_flag);

  EXPECT_EQ(deserialized_flag, original_flag);
}

TEST_F(ArchiverTest, TestTensorArchiver) {
  Tensor tensor = Tensor({32, 512, 768}, DType_t::FP32);

  fill_normal(tensor, 0.0f, 1.0f);

  Sizer sizer;
  sizer(tensor);
  size_t expected_size = sizer.size();
  dptr buffer = allocator_.allocate(expected_size);
  EXPECT_EQ(tensor.size(), tensor.capacity());
  EXPECT_EQ(sizer.size(), sizeof(DType_t) + sizeof(uint64_t) +
                              sizeof(uint64_t) * tensor.shape().size() +
                              tensor.size() * sizeof(float));
  Writer writer(buffer);
  writer(tensor);

  Reader reader(buffer);
  Tensor deserialized_tensor;
  BinarySerializer bserializer(allocator_);
  bserializer.deserialize(reader, deserialized_tensor);

  EXPECT_EQ(deserialized_tensor.shape(), tensor.shape());
  EXPECT_EQ(deserialized_tensor.dtype(), tensor.dtype());
  EXPECT_EQ(deserialized_tensor.device(), tensor.device());
  EXPECT_EQ(deserialized_tensor.size(), tensor.size());
  float* tensor_data = tensor.data_as<float>();
  float* deserialized_tensor_data = deserialized_tensor.data_as<float>();
  for (size_t i = 0; i < tensor.size(); ++i) {
    EXPECT_FLOAT_EQ(deserialized_tensor_data[i], tensor_data[i]);
  }
}

TEST_F(ArchiverTest, TestJobArchiver) {
  Job job;
  job.pid = 123;
  Tensor left = Tensor({64, 256}, DType_t::FP32);
  Tensor right = Tensor({64, 128}, DType_t::FP32);
  fill_normal(left, 0.0f, 1.0f);
  fill_normal(right, 0.0f, 1.0f);
  job.data.set("left", left);
  job.data.set("right", right);
  Sizer sizer;
  sizer(job);
  size_t expected_size = sizer.size();
  dptr buffer = allocator_.allocate(expected_size);
  Writer writer(buffer);
  writer(job);

  Reader reader(buffer);
  Job deserialized_job;
  BinarySerializer bserializer(allocator_);
  bserializer.deserialize(reader, deserialized_job);

  EXPECT_EQ(deserialized_job.pid, job.pid);
  ASSERT_EQ(deserialized_job.data.size(), 2u);
  ASSERT_TRUE(deserialized_job.data.contains("left"));
  ASSERT_TRUE(deserialized_job.data.contains("right"));

  for (const auto& uid : {std::string("left"), std::string("right")}) {
    const Tensor& expected = job.data.get(uid);
    const Tensor& actual = deserialized_job.data.get(uid);
    ASSERT_TRUE(expected);
    ASSERT_TRUE(actual);
    EXPECT_EQ(actual.shape(), expected.shape());
    EXPECT_EQ(actual.dtype(), expected.dtype());
    EXPECT_EQ(actual.device(), expected.device());
    EXPECT_EQ(actual.size(), expected.size());

    const float* expected_data = expected.data_as<float>();
    const float* actual_data = actual.data_as<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_FLOAT_EQ(actual_data[i], expected_data[i]);
    }
  }
}

TEST_F(ArchiverTest, TestMessageDataArchiver) {
  MessageData data;
  data.payload = std::string("Test Message");
  Sizer sizer;
  sizer(data);
  size_t expected_size = sizer.size();
  dptr buffer = allocator_.allocate(expected_size);
  Writer writer(buffer);
  writer(data);

  Reader reader(buffer);
  MessageData deserialized_data;
  BinarySerializer bserializer(allocator_);
  bserializer.deserialize(reader, deserialized_data);

  EXPECT_TRUE(deserialized_data.payload.holds<std::string>());
  EXPECT_EQ(deserialized_data.payload.get<std::string>(), "Test Message");
}

TEST_F(ArchiverTest, LargeJobMessageSurvivesPacketSlicingAndReassembly) {
  Tensor left_host = Tensor({256, 64, 8, 8}, DType_t::FP32);
  Tensor right_host = Tensor({256, 64, 8, 8}, DType_t::FP32);
  fill_normal(left_host, 0.0f, 1.0f);
  fill_normal(right_host, 0.0f, 1.0f);

  Job job;
  job.pid = 77;
  job.data.set("left", left_host.to_device(getGPU()));
  job.data.set("right", right_host.to_device(getGPU()));

  Message original(CommandType::FORWARD_JOB, std::move(job));

  Sizer sizer;
  BinarySerializer serializer(allocator_);
  sizer(original.header());
  sizer(original.data().payload.index());
  sizer(original.get<Job>());
  dptr serialized_buffer = allocator_.allocate(sizer.size());
  Writer writer(serialized_buffer);
  serializer.serialize(writer, original);

  BlockSlicer slicer(64 * 1024);
  Vec<Packet> packets = slicer.slice(std::move(serialized_buffer));
  ASSERT_GT(packets.size(), 1u);

  RawAggregator aggregator(allocator_);
  dptr reassembled_buffer;
  for (auto it = packets.rbegin(); it != packets.rend(); ++it) {
    dptr packet_target = aggregator.fetch_packet(it->header);
    std::memcpy(packet_target.get<unsigned char>(), it->data.get<unsigned char>(),
                it->header.packet_length);
    if (aggregator.commit_packet(it->header)) {
      reassembled_buffer = aggregator.finalize(it->header);
    }
  }

  ASSERT_TRUE(reassembled_buffer);

  Reader reader(reassembled_buffer);
  Message reconstructed;
  serializer.deserialize(reader, reconstructed);

  EXPECT_EQ(reconstructed.header().command_type, CommandType::FORWARD_JOB);
  ASSERT_TRUE(reconstructed.has_type<Job>());

  const Job& reconstructed_job = reconstructed.get<Job>();
  EXPECT_EQ(reconstructed_job.pid, 77u);
  ASSERT_EQ(reconstructed_job.data.size(), 2u);
  ASSERT_TRUE(reconstructed_job.data.contains("left"));
  ASSERT_TRUE(reconstructed_job.data.contains("right"));

  expect_float_tensors_equal(reconstructed_job.data.get("left"), left_host);
  expect_float_tensors_equal(reconstructed_job.data.get("right"), right_host);
}

TEST_F(ArchiverTest, ProfilerArchiverRoundTripPreservesEvents) {
  Profiler profiler;
  profiler.init_start_time(Clock::time_point(Clock::duration(123456)));
  profiler.add_event(Event{EventType::COMMUNICATION, Clock::time_point(Clock::duration(111)),
                           Clock::time_point(Clock::duration(222)), "Packet Write", "worker-0"});
  profiler.add_event(Event{EventType::COMPUTE, Clock::time_point(Clock::duration(333)),
                           Clock::time_point(Clock::duration(444)), "Forward", "worker-1"});

  Sizer sizer;
  sizer(profiler);
  dptr buffer = allocator_.allocate(sizer.size());
  Writer writer(buffer);
  writer(profiler);

  Reader header_reader(buffer);
  int64 serialized_start_time = 0;
  uint32_t serialized_event_count = 0;
  header_reader(serialized_start_time, serialized_event_count);
  EXPECT_EQ(serialized_start_time, profiler.start_time().time_since_epoch().count());
  EXPECT_EQ(serialized_event_count, 2u);

  Reader reader(buffer);
  Profiler deserialized_profiler;
  BinarySerializer serializer(allocator_);
  serializer.deserialize(reader, deserialized_profiler);

  EXPECT_EQ(deserialized_profiler.start_time().time_since_epoch().count(),
            profiler.start_time().time_since_epoch().count());

  const auto& events = deserialized_profiler.get_events();
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].type, EventType::COMMUNICATION);
  EXPECT_EQ(events[0].start_time.time_since_epoch().count(), 111);
  EXPECT_EQ(events[0].end_time.time_since_epoch().count(), 222);
  EXPECT_EQ(events[0].name, "Packet Write");
  EXPECT_EQ(events[0].source, "worker-0");

  EXPECT_EQ(events[1].type, EventType::COMPUTE);
  EXPECT_EQ(events[1].start_time.time_since_epoch().count(), 333);
  EXPECT_EQ(events[1].end_time.time_since_epoch().count(), 444);
  EXPECT_EQ(events[1].name, "Forward");
  EXPECT_EQ(events[1].source, "worker-1");
}