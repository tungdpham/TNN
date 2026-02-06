/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "device/device_allocator.hpp"
#include "distributed/vbuffer.hpp"
#include "distributed/vserializer.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

class VBufferSerializerTest : public ::testing::Test {
protected:
  IAllocator& allocator = HostAllocator();
  VSerializer serializer{allocator};

  void SetUp() override {}
};

TEST_F(VBufferSerializerTest, VBufferBasicOperations) {
  VBuffer buffer(allocator);

  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.size(), 0);
  EXPECT_EQ(buffer.capacity(), 0);

  buffer.alloc(100);
  EXPECT_FALSE(buffer.empty());
  EXPECT_EQ(buffer.size(), 0);
  EXPECT_EQ(buffer.capacity(), 100);

  buffer.resize(50);
  EXPECT_EQ(buffer.size(), 50);

  buffer.reset();
  EXPECT_EQ(buffer.size(), 0);
  EXPECT_GE(buffer.capacity(), 100);

  buffer.clear();
  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.size(), 0);
  EXPECT_EQ(buffer.capacity(), 0);
}

TEST_F(VBufferSerializerTest, VBufferReadWrite) {
  VBuffer buffer(allocator);
  buffer.alloc(1024);
  size_t offset = 0;

  int val_i = 42;
  float val_f = 3.14f;
  std::string val_s = "Hello VBuffer";
  int arr[3] = {1, 2, 3};

  buffer.write(offset, val_i);
  buffer.write(offset, val_f);
  buffer.write(offset, val_s);
  buffer.write(offset, arr, 3);

  size_t read_offset = 0;
  int res_i;
  float res_f;
  std::string res_s;
  int res_arr[3];

  buffer.read(read_offset, res_i);
  buffer.read(read_offset, res_f);
  buffer.read(read_offset, res_s);
  buffer.read(read_offset, res_arr, 3);

  EXPECT_EQ(res_i, val_i);
  EXPECT_FLOAT_EQ(res_f, val_f);
  EXPECT_EQ(res_s, val_s);
  EXPECT_EQ(res_arr[0], arr[0]);
  EXPECT_EQ(res_arr[1], arr[1]);
  EXPECT_EQ(res_arr[2], arr[2]);
}

TEST_F(VBufferSerializerTest, VBufferPollAndGet) {
  VBuffer buffer(allocator);
  buffer.alloc(100);
  buffer.alloc(200);

  EXPECT_EQ(buffer.capacity(), 300);
  EXPECT_EQ(buffer.size(), 0);

  buffer.resize(300);

  dptr p1 = buffer.get(50);
  EXPECT_EQ(p1.capacity(), 50);  // 100 - 50

  dptr p2 = buffer.get(150);
  EXPECT_EQ(p2.capacity(), 150);  // (100+200) - 150

  dptr polled = buffer.poll();
  EXPECT_EQ(polled.capacity(), 100);
  EXPECT_EQ(buffer.size(), 200);
  EXPECT_EQ(buffer.capacity(), 200);
}

TEST_F(VBufferSerializerTest, VBufferAppend) {
  VBuffer buffer(allocator);
  size_t size = 50;
  dptr d = allocator.allocate(size);

  buffer.append(std::move(d));
  EXPECT_EQ(buffer.size(), size);
  EXPECT_EQ(buffer.capacity(), size);
}

TEST_F(VBufferSerializerTest, SerializerPacketHeader) {
  VBuffer buffer(allocator);
  buffer.alloc(1024);
  size_t offset = 0;

  PacketHeader header;
  header.type = PacketType::DATA_FRAGMENT;
  header.packet_length = 1234;
  header.msg_length = 5678;
  header.msg_serial_id = 999;
  header.packet_offset = 10;
  header.total_packets = 20;
  header.compression_type = CompressionType::ZSTD;

  serializer.serialize(buffer, offset, header);

  size_t read_offset = 0;
  PacketHeader res_header;
  serializer.deserialize(buffer, read_offset, res_header);

  EXPECT_EQ(res_header.type, header.type);
  EXPECT_EQ(res_header.packet_length, header.packet_length);
  EXPECT_EQ(res_header.msg_length, header.msg_length);
  EXPECT_EQ(res_header.msg_serial_id, header.msg_serial_id);
  EXPECT_EQ(res_header.packet_offset, header.packet_offset);
  EXPECT_EQ(res_header.total_packets, header.total_packets);
  EXPECT_EQ(res_header.compression_type, header.compression_type);
}

TEST_F(VBufferSerializerTest, SerializerMessageHeader) {
  VBuffer buffer(allocator);
  buffer.alloc(1024);
  size_t offset = 0;

  MessageHeader header;
  header.command_type = CommandType::FORWARD_JOB;

  serializer.serialize(buffer, offset, header);

  size_t read_offset = 0;
  MessageHeader res_header;
  serializer.deserialize(buffer, read_offset, res_header);

  EXPECT_EQ(res_header.command_type, header.command_type);
}

TEST_F(VBufferSerializerTest, SerializerTensor) {
  VBuffer buffer(allocator);
  buffer.alloc(1024);
  size_t offset = 0;

  std::vector<uint64_t> shape = {2, 3};
  Tensor tensor = make_tensor(allocator, DType_t::FP32, shape);
  float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  tensor->data_ptr().copy_from_host(data, 6 * sizeof(float));

  serializer.serialize(buffer, offset, std::move(tensor));

  size_t read_offset = 0;
  Tensor res_tensor;
  serializer.deserialize(buffer, read_offset, res_tensor);

  EXPECT_EQ(res_tensor->data_type(), DType_t::FP32);
  EXPECT_EQ(res_tensor->shape(), (std::vector<size_t>{2, 3}));

  float res_data[6];
  res_tensor->data_ptr().copy_to_host(res_data, 6 * sizeof(float));
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(res_data[i], data[i]);
  }
}

TEST_F(VBufferSerializerTest, SerializerMessageData) {
  VBuffer buffer(allocator);
  buffer.alloc(4096);

  // Test String Payload
  {
    size_t offset = 0;
    MessageData data;
    data.payload = std::string("Test Payload");
    serializer.serialize(buffer, offset, std::move(data));

    size_t read_offset = 0;
    MessageData res_data;
    serializer.deserialize(buffer, read_offset, res_data);
    EXPECT_TRUE(std::holds_alternative<std::string>(res_data.payload));
    EXPECT_EQ(std::get<std::string>(res_data.payload), "Test Payload");
  }

  // Test Bool Payload
  {
    buffer.reset();
    size_t offset = 0;
    MessageData data;
    data.payload = true;
    serializer.serialize(buffer, offset, std::move(data));

    size_t read_offset = 0;
    MessageData res_data;
    serializer.deserialize(buffer, read_offset, res_data);
    EXPECT_TRUE(std::holds_alternative<bool>(res_data.payload));
    EXPECT_TRUE(std::get<bool>(res_data.payload));
  }
}

TEST_F(VBufferSerializerTest, SerializerMessage) {
  VBuffer buffer(allocator);
  buffer.alloc(4096);
  size_t offset = 0;

  Message msg;
  msg.header().command_type = CommandType::BACKWARD_JOB;
  msg.data().payload = std::string("Message Test");

  serializer.serialize(buffer, offset, std::move(msg));

  size_t read_offset = 0;
  Message res_msg;
  serializer.deserialize(buffer, read_offset, res_msg);

  EXPECT_EQ(res_msg.header().command_type, CommandType::BACKWARD_JOB);
  EXPECT_TRUE(res_msg.has_type<std::string>());
  EXPECT_EQ(res_msg.get<std::string>(), "Message Test");
}
