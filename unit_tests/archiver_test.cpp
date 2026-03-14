#include "common/archiver.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "device/iallocator.hpp"
#include "distributed/binary_serializer.hpp"
#include "distributed/io.hpp"
#include "distributed/packet.hpp"
#include "tensor/tensor_factory.hpp"
#include "type/type.hpp"

using namespace tnn;

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
  Tensor tensor = make_tensor(DType_t::FP32, {64, 512, 768});

  tensor->fill_random_normal(0.0f, 1.0f);

  Sizer sizer;
  sizer(tensor);
  size_t expected_size = sizer.size();
  dptr buffer = allocator_.allocate(expected_size);
  EXPECT_EQ(tensor->size(), tensor->capacity());
  EXPECT_EQ(sizer.size(), sizeof(DType_t) + sizeof(uint64_t) +
                              sizeof(uint64_t) * tensor->shape().size() +
                              tensor->size() * sizeof(float));
  Writer writer(buffer);
  writer(tensor);

  Reader reader(buffer);
  Tensor deserialized_tensor;
  BinarySerializer bserializer(allocator_);
  bserializer.deserialize(reader, deserialized_tensor);

  EXPECT_EQ(deserialized_tensor->shape(), tensor->shape());
  EXPECT_EQ(deserialized_tensor->data_type(), tensor->data_type());
  EXPECT_EQ(deserialized_tensor->device(), tensor->device());
  EXPECT_EQ(deserialized_tensor->size(), tensor->size());
  float* tensor_data = tensor->data_as<float>();
  float* deserialized_tensor_data = deserialized_tensor->data_as<float>();
  for (size_t i = 0; i < tensor->size(); ++i) {
    EXPECT_FLOAT_EQ(deserialized_tensor_data[i], tensor_data[i]);
  }
}

TEST_F(ArchiverTest, TestJobArchiver) {
  Job job;
  job.mb_id = 123;
  job.data = make_tensor(DType_t::FP32, {128, 256});
  job.data->fill_random_normal(0.0f, 1.0f);
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

  EXPECT_EQ(deserialized_job.mb_id, job.mb_id);
  EXPECT_EQ(deserialized_job.data->shape(), job.data->shape());
  EXPECT_EQ(deserialized_job.data->data_type(), job.data->data_type());
  EXPECT_EQ(deserialized_job.data->device(), job.data->device());
  EXPECT_EQ(deserialized_job.data->size(), job.data->size());
  float* job_data = job.data->data_as<float>();
  float* deserialized_job_data = deserialized_job.data->data_as<float>();
  for (size_t i = 0; i < job.data->size(); ++i) {
    EXPECT_FLOAT_EQ(deserialized_job_data[i], job_data[i]);
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