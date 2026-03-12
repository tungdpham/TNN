#include "common/archiver.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "distributed/packet.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor_factory.hpp"
#include "type/type.hpp"

using namespace tnn;

class ArchiverTest : public ::testing::Test {
public:
  static void SetUpTestSuite() { initializeDefaultDevices(); }
};

TEST_F(ArchiverTest, TestSizeArchiver) {
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
  SizeArchiver size_archiver;
  size_archiver & header;
  EXPECT_EQ(size_archiver.size(), PacketHeader::size());
}

TEST_F(ArchiverTest, TestOutArchiver) {
  char buffer[1024];
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
  OutArchiver out_archiver(buffer, sizeof(buffer));
  out_archiver & header;
  EXPECT_EQ(out_archiver.bytes_written(), PacketHeader::size());
}

TEST_F(ArchiverTest, TestBlobArchiver) {
  constexpr size_t header_size = sizeof(uint64_t);  // 8 bytes for data size
  constexpr size_t blob_size = 4 * 1024 * 1024;     // 4 MB
  constexpr size_t byte_size =
      blob_size * sizeof(int) + header_size;  // Total size of the buffer needed
  auto data = std::make_unique<int[]>(blob_size);
  auto buffer = std::make_unique<char[]>(byte_size);
  Blob<int> blob_data(data.get(), blob_size);
  SizeArchiver size_archiver;
  size_archiver & blob_data;
  EXPECT_EQ(size_archiver.size(), byte_size);
  OutArchiver out_archiver(buffer.get(), byte_size);
  out_archiver & blob_data;
  EXPECT_EQ(out_archiver.bytes_written(), byte_size);
}

TEST_F(ArchiverTest, TestInArchiver) {
  char buffer[1024];
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

  // First, write the header to the buffer using OutArchiver
  OutArchiver out_archiver(buffer, sizeof(buffer));
  out_archiver & header;

  // Now read it back using InArchiver
  PacketHeader read_header;
  InArchiver in_archiver(buffer, sizeof(buffer));
  in_archiver & read_header;

  EXPECT_EQ(in_archiver.bytes_read(), PacketHeader::size());
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

TEST_F(ArchiverTest, TestDptrArchiver) {
  constexpr size_t byte_size = 4 * 1024 * 1024;     // 4 MB
  constexpr size_t header_size = sizeof(uint64_t);  // 8 bytes for the header
  constexpr size_t total_size = byte_size + header_size;
  auto &allocator = DeviceAllocator::instance(getHost());
  dptr data = allocator.allocate(byte_size);
  ops::set_scalar<float>(data, 0.5, byte_size / 4);
  SizeArchiver size_archiver;
  size_archiver & data;
  EXPECT_EQ(size_archiver.size(), total_size);  // 8 bytes for the size header
  auto buffer = std::make_unique<char[]>(total_size);
  OutArchiver out_archiver(buffer.get(), total_size);
  out_archiver & data;
  EXPECT_EQ(out_archiver.bytes_written(), total_size);  // 8 bytes for the size header
  InArchiver in_archiver(buffer.get(), total_size);
  dptr read_data = allocator.allocate(total_size);
  in_archiver & read_data;
  EXPECT_EQ(in_archiver.bytes_read(), total_size);  // 8 bytes for the size header
  EXPECT_EQ(read_data.capacity(), data.capacity());
  // Verify data correctness
  float *ptr = read_data.get<float>();
  for (size_t i = 0; i < byte_size / sizeof(float); i++) {
    EXPECT_EQ(ptr[i], 0.5f);
  }
}

TEST_F(ArchiverTest, TestTensorArchiver) {
  constexpr size_t N = 4;
  constexpr size_t S = 1024;
  constexpr size_t E = 768;
  constexpr size_t total_elements = N * S * E;

  Tensor tensor = make_tensor<float>({N, S, E}, getHost());

  SizeArchiver size_archiver;
  size_archiver &*tensor;
  EXPECT_EQ(size_archiver.size(),
            sizeof(DType_t) + sizeof(uint64_t) + sizeof(uint64_t) * 3 + sizeof(uint64_t) +
                total_elements *
                    sizeof(float));  // dtype + shape size + shape data + data size + tensor data

  auto buffer = std::make_unique<char[]>(size_archiver.size());
  OutArchiver out_archiver(buffer.get(), size_archiver.size());
  out_archiver &*tensor;
  EXPECT_EQ(out_archiver.bytes_written(), size_archiver.size());

  InArchiver in_archiver(buffer.get(), size_archiver.size());
  Tensor read_tensor;
  in_archiver &*read_tensor;
  EXPECT_EQ(in_archiver.bytes_read(), size_archiver.size());

  EXPECT_EQ(read_tensor->data_type(), dtype_of<float>());
  EXPECT_EQ(read_tensor->shape(), std::vector<size_t>({N, S, E}));
}