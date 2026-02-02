/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "distributed/vserializer.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "device/sref.hpp"
#include "distributed/command_type.hpp"
#include "distributed/job.hpp"
#include "distributed/message.hpp"
#include "distributed/packet.hpp"
#include "distributed/vbuffer.hpp"
#include "profiling/event.hpp"
#include "profiling/profiler.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

class VSerializerTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { initializeDefaultDevices(); }

  void SetUp() override { serializer_ = std::make_unique<VSerializer>(*allocator_); }

  void TearDown() override { serializer_.reset(); }

  static void TearDownTestSuite() {}

  sref<IAllocator> allocator_ = DeviceAllocator::instance(getCPU());
  std::unique_ptr<VSerializer> serializer_;
};

// Tensor Serialization Tests

TEST_F(VSerializerTest, SerializeDeserializeTensor1D) {
  // Create a 1D tensor
  Tensor tensor = make_tensor<float>({10}, getCPU());
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  std::memcpy(tensor->data_as<float>(), data.data(), data.size() * sizeof(float));

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(4096));
  buffer.resize(4096);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(tensor));

  // Deserialize
  offset = 0;
  Tensor deserialized_tensor;
  serializer_->deserialize(buffer, offset, deserialized_tensor);

  // Verify
  ASSERT_EQ(deserialized_tensor->shape().size(), 1);
  EXPECT_EQ(deserialized_tensor->shape()[0], 10);
  EXPECT_EQ(deserialized_tensor->data_type(), DType_t::FP32);

  std::vector<float> result_data(
      deserialized_tensor->data_as<float>(),
      deserialized_tensor->data_as<float>() + deserialized_tensor->size());
  ASSERT_EQ(result_data.size(), 10);
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(result_data[i], data[i]);
  }
}

TEST_F(VSerializerTest, SerializeDeserializeTensor2D) {
  // Create a 2D tensor
  Tensor tensor = make_tensor<float>({3, 4}, getCPU());
  std::vector<float> data(12);
  for (int i = 0; i < 12; ++i) {
    data[i] = static_cast<float>(i);
  }
  std::memcpy(tensor->data_as<float>(), data.data(), data.size() * sizeof(float));

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(4096));
  buffer.resize(4096);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(tensor));

  // Deserialize
  offset = 0;
  Tensor deserialized_tensor;
  serializer_->deserialize(buffer, offset, deserialized_tensor);

  // Verify
  ASSERT_EQ(deserialized_tensor->shape().size(), 2);
  EXPECT_EQ(deserialized_tensor->shape()[0], 3);
  EXPECT_EQ(deserialized_tensor->shape()[1], 4);

  std::vector<float> result_data(
      deserialized_tensor->data_as<float>(),
      deserialized_tensor->data_as<float>() + deserialized_tensor->size());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(result_data[i], data[i]);
  }
}

TEST_F(VSerializerTest, SerializeDeserializeTensor4D) {
  // Create a 4D tensor (common in CNN: batch, channels, height, width)
  Tensor tensor = make_tensor<float>({2, 3, 4, 5}, getCPU());
  size_t total_size = 2 * 3 * 4 * 5;
  std::vector<float> data(total_size);
  for (size_t i = 0; i < total_size; ++i) {
    data[i] = static_cast<float>(i) * 0.1f;
  }
  std::memcpy(tensor->data_as<float>(), data.data(), data.size() * sizeof(float));

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(8192));
  buffer.resize(8192);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(tensor));

  // Deserialize
  offset = 0;
  Tensor deserialized_tensor;
  serializer_->deserialize(buffer, offset, deserialized_tensor);

  // Verify shape
  ASSERT_EQ(deserialized_tensor->shape().size(), 4);
  EXPECT_EQ(deserialized_tensor->shape()[0], 2);
  EXPECT_EQ(deserialized_tensor->shape()[1], 3);
  EXPECT_EQ(deserialized_tensor->shape()[2], 4);
  EXPECT_EQ(deserialized_tensor->shape()[3], 5);

  // Verify data
  std::vector<float> result_data(
      deserialized_tensor->data_as<float>(),
      deserialized_tensor->data_as<float>() + deserialized_tensor->size());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(result_data[i], data[i]);
  }
}

TEST_F(VSerializerTest, SerializeDeserializeInt32Tensor) {
  // Create an int32 tensor
  Tensor tensor = make_tensor<int32_t>({5, 6}, getCPU());
  std::vector<int32_t> data(30);
  for (int i = 0; i < 30; ++i) {
    data[i] = i * 10;
  }
  std::memcpy(tensor->data_as<int32_t>(), data.data(), data.size() * sizeof(int32_t));

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(4096));
  buffer.resize(4096);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(tensor));

  // Deserialize
  offset = 0;
  Tensor deserialized_tensor;
  serializer_->deserialize(buffer, offset, deserialized_tensor);

  // Verify
  EXPECT_EQ(deserialized_tensor->data_type(), DType_t::INT32_T);
  std::vector<int32_t> result_data(
      deserialized_tensor->data_as<int32_t>(),
      deserialized_tensor->data_as<int32_t>() + deserialized_tensor->size());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(result_data[i], data[i]);
  }
}

// Event Serialization Tests

TEST_F(VSerializerTest, SerializeDeserializeEvent) {
  Event event;
  event.type = EventType::COMPUTE;
  event.start_time = Clock::now();
  event.end_time = event.start_time + std::chrono::milliseconds(100);
  event.name = "TestEvent";
  event.source = "TestSource";

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(4096));
  buffer.resize(4096);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(event));

  // Deserialize
  offset = 0;
  Event deserialized_event;
  serializer_->deserialize(buffer, offset, deserialized_event);

  // Verify
  EXPECT_EQ(deserialized_event.type, EventType::COMPUTE);
  EXPECT_EQ(deserialized_event.start_time, event.start_time);
  EXPECT_EQ(deserialized_event.end_time, event.end_time);
  EXPECT_EQ(deserialized_event.name, "TestEvent");
  EXPECT_EQ(deserialized_event.source, "TestSource");
}

TEST_F(VSerializerTest, SerializeDeserializeMultipleEventTypes) {
  std::vector<Event> events(3);

  events[0].type = EventType::COMPUTE;
  events[0].start_time = Clock::now();
  events[0].end_time = events[0].start_time + std::chrono::milliseconds(50);
  events[0].name = "Compute1";
  events[0].source = "GPU";

  events[1].type = EventType::COMMUNICATION;
  events[1].start_time = events[0].end_time;
  events[1].end_time = events[1].start_time + std::chrono::milliseconds(30);
  events[1].name = "Send";
  events[1].source = "Network";

  events[2].type = EventType::OTHER;
  events[2].start_time = events[1].end_time;
  events[2].end_time = events[2].start_time + std::chrono::milliseconds(20);
  events[2].name = "Other";
  events[2].source = "System";

  // Serialize all events
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(8192));
  buffer.resize(8192);
  size_t offset = 0;
  for (size_t i = 0; i < events.size(); ++i) {
    serializer_->serialize(buffer, offset, std::move(events[i]));
  }

  // Deserialize and verify
  offset = 0;
  for (size_t i = 0; i < events.size(); ++i) {
    Event deserialized_event;
    serializer_->deserialize(buffer, offset, deserialized_event);
    EXPECT_EQ(deserialized_event.type, events[i].type);
    EXPECT_EQ(deserialized_event.name, events[i].name);
    EXPECT_EQ(deserialized_event.source, events[i].source);
  }
}

// Profiler Serialization Tests

TEST_F(VSerializerTest, SerializeDeserializeEmptyProfiler) {
  Profiler profiler;

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(4096));
  buffer.resize(4096);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(profiler));

  // Deserialize
  offset = 0;
  Profiler deserialized_profiler;
  serializer_->deserialize(buffer, offset, deserialized_profiler);

  // Verify
  EXPECT_EQ(deserialized_profiler.get_events().size(), 0);
}

TEST_F(VSerializerTest, SerializeDeserializeProfilerWithEvents) {
  Profiler profiler;

  Event event1;
  event1.type = EventType::COMPUTE;
  event1.start_time = Clock::now();
  event1.end_time = event1.start_time + std::chrono::milliseconds(100);
  event1.name = "Forward";
  event1.source = "Layer1";

  Event event2;
  event2.type = EventType::COMPUTE;
  event2.start_time = event1.end_time;
  event2.end_time = event2.start_time + std::chrono::milliseconds(150);
  event2.name = "Backward";
  event2.source = "Layer1";

  profiler.add_event(event1);
  profiler.add_event(event2);

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(8192));
  buffer.resize(8192);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(profiler));

  // Deserialize
  offset = 0;
  Profiler deserialized_profiler;
  serializer_->deserialize(buffer, offset, deserialized_profiler);

  // Verify
  auto events = deserialized_profiler.get_events();
  ASSERT_EQ(events.size(), 2);

  EXPECT_EQ(events[0].type, EventType::COMPUTE);
  EXPECT_EQ(events[0].name, "Forward");
  EXPECT_EQ(events[0].source, "Layer1");

  EXPECT_EQ(events[1].type, EventType::COMPUTE);
  EXPECT_EQ(events[1].name, "Backward");
  EXPECT_EQ(events[1].source, "Layer1");
}

// PacketHeader Serialization Tests

TEST_F(VSerializerTest, SerializeDeserializePacketHeader) {
  PacketHeader header;
  header.type = PacketType::DATA_FRAGMENT;
  header.length = 1024;
  header.msg_length = 4096;
  header.msg_serial_id = 12345;
  header.packet_offset = 0;
  header.total_packets = 4;
  header.compression_type = CompressionType::NONE;

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(1024));
  buffer.resize(1024);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, header);

  // Deserialize
  offset = 0;
  PacketHeader deserialized_header;
  serializer_->deserialize(buffer, offset, deserialized_header);

  // Verify
  EXPECT_EQ(deserialized_header.PROTOCOL_VERSION, header.PROTOCOL_VERSION);
  EXPECT_EQ(deserialized_header.type, PacketType::DATA_FRAGMENT);
  EXPECT_EQ(deserialized_header.length, 1024);
  EXPECT_EQ(deserialized_header.msg_length, 4096);
  EXPECT_EQ(deserialized_header.msg_serial_id, 12345);
  EXPECT_EQ(deserialized_header.packet_offset, 0);
  EXPECT_EQ(deserialized_header.total_packets, 4);
  EXPECT_EQ(deserialized_header.compression_type, CompressionType::NONE);
}

TEST_F(VSerializerTest, SerializeDeserializePacketHeaderWithCompression) {
  PacketHeader header;
  header.type = PacketType::DATA_FRAGMENT;
  header.length = 512;
  header.msg_length = 2048;
  header.msg_serial_id = 67890;
  header.packet_offset = 1;
  header.total_packets = 2;
  header.compression_type = CompressionType::ZSTD;

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(1024));
  buffer.resize(1024);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, header);

  // Deserialize
  offset = 0;
  PacketHeader deserialized_header;
  serializer_->deserialize(buffer, offset, deserialized_header);

  // Verify
  EXPECT_EQ(deserialized_header.compression_type, CompressionType::ZSTD);
  EXPECT_EQ(deserialized_header.packet_offset, 1);
  EXPECT_EQ(deserialized_header.total_packets, 2);
}

// MessageHeader Serialization Tests

TEST_F(VSerializerTest, SerializeDeserializeMessageHeader) {
  MessageHeader header(CommandType::FORWARD_JOB);

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(1024));
  buffer.resize(1024);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, header);

  // Deserialize
  offset = 0;
  MessageHeader deserialized_header;
  serializer_->deserialize(buffer, offset, deserialized_header);

  // Verify
  EXPECT_EQ(deserialized_header.command_type, CommandType::FORWARD_JOB);
}

TEST_F(VSerializerTest, SerializeDeserializeMultipleMessageHeaders) {
  std::vector<CommandType> commands = {CommandType::FORWARD_JOB, CommandType::BACKWARD_JOB,
                                       CommandType::UPDATE_PARAMETERS, CommandType::SHUTDOWN};

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(1024));
  buffer.resize(1024);
  size_t offset = 0;
  for (auto cmd : commands) {
    MessageHeader header(cmd);
    serializer_->serialize(buffer, offset, header);
  }

  // Deserialize and verify
  offset = 0;
  for (auto cmd : commands) {
    MessageHeader deserialized_header;
    serializer_->deserialize(buffer, offset, deserialized_header);
    EXPECT_EQ(deserialized_header.command_type, cmd);
  }
}

// MessageData Serialization Tests

TEST_F(VSerializerTest, SerializeDeserializeMessageDataMonostate) {
  MessageData data(std::monostate{});

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(1024));
  buffer.resize(1024);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(data));

  // Deserialize
  offset = 0;
  MessageData deserialized_data(std::monostate{});
  serializer_->deserialize(buffer, offset, deserialized_data);

  // Verify
  EXPECT_TRUE(std::holds_alternative<std::monostate>(deserialized_data.payload));
}

TEST_F(VSerializerTest, SerializeDeserializeMessageDataString) {
  std::string test_string = "Hello, distributed TNN!";
  MessageData data(std::move(test_string));

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(2048));
  buffer.resize(2048);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(data));

  // Deserialize
  offset = 0;
  MessageData deserialized_data(std::monostate{});
  serializer_->deserialize(buffer, offset, deserialized_data);

  // Verify
  ASSERT_TRUE(std::holds_alternative<std::string>(deserialized_data.payload));
  EXPECT_EQ(std::get<std::string>(deserialized_data.payload), "Hello, distributed TNN!");
}

TEST_F(VSerializerTest, SerializeDeserializeMessageDataBool) {
  MessageData data_true(true);
  MessageData data_false(false);

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(1024));
  buffer.resize(1024);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(data_true));
  serializer_->serialize(buffer, offset, std::move(data_false));

  // Deserialize
  offset = 0;
  MessageData deserialized_true(std::monostate{});
  MessageData deserialized_false(std::monostate{});
  serializer_->deserialize(buffer, offset, deserialized_true);
  serializer_->deserialize(buffer, offset, deserialized_false);

  // Verify
  ASSERT_TRUE(std::holds_alternative<bool>(deserialized_true.payload));
  EXPECT_TRUE(std::get<bool>(deserialized_true.payload));

  ASSERT_TRUE(std::holds_alternative<bool>(deserialized_false.payload));
  EXPECT_FALSE(std::get<bool>(deserialized_false.payload));
}

TEST_F(VSerializerTest, SerializeDeserializeMessageDataJob) {
  Tensor tensor = make_tensor<float>({2, 3}, getCPU());
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::memcpy(tensor->data_as<float>(), data.data(), data.size() * sizeof(float));

  Job job(tensor, 42);
  MessageData msg_data(std::move(job));

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(4096));
  buffer.resize(4096);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(msg_data));

  // Deserialize
  offset = 0;
  MessageData deserialized_data(std::monostate{});
  serializer_->deserialize(buffer, offset, deserialized_data);

  // Verify
  ASSERT_TRUE(std::holds_alternative<Job>(deserialized_data.payload));
  const Job &deserialized_job = std::get<Job>(deserialized_data.payload);
  EXPECT_EQ(deserialized_job.mb_id, 42);

  std::vector<float> result_data(
      deserialized_job.data->data_as<float>(),
      deserialized_job.data->data_as<float>() + deserialized_job.data->size());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(result_data[i], data[i]);
  }
}

TEST_F(VSerializerTest, SerializeDeserializeMessageDataProfiler) {
  Profiler profiler;

  Event event;
  event.type = EventType::COMPUTE;
  event.start_time = Clock::now();
  event.end_time = event.start_time + std::chrono::milliseconds(100);
  event.name = "TestComputation";
  event.source = "Worker1";

  profiler.add_event(event);
  MessageData data(std::move(profiler));

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(4096));
  buffer.resize(4096);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(data));

  // Deserialize
  offset = 0;
  MessageData deserialized_data(std::monostate{});
  serializer_->deserialize(buffer, offset, deserialized_data);

  // Verify
  ASSERT_TRUE(std::holds_alternative<Profiler>(deserialized_data.payload));
  const Profiler &deserialized_profiler = std::get<Profiler>(deserialized_data.payload);
  auto events = deserialized_profiler.get_events();
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0].name, "TestComputation");
  EXPECT_EQ(events[0].source, "Worker1");
}

// Message Serialization Tests

TEST_F(VSerializerTest, SerializeDeserializeMessageWithString) {
  MessageHeader header(CommandType::FORWARD_JOB);
  MessageData data(std::string("Test message payload"));
  Message message(std::move(header), std::move(data));

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(2048));
  buffer.resize(2048);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(message));

  // Deserialize
  offset = 0;
  Message deserialized_message;
  serializer_->deserialize(buffer, offset, deserialized_message);

  // Verify
  EXPECT_EQ(deserialized_message.header().command_type, CommandType::FORWARD_JOB);
  ASSERT_TRUE(std::holds_alternative<std::string>(deserialized_message.data().payload));
  EXPECT_EQ(std::get<std::string>(deserialized_message.data().payload), "Test message payload");
}

TEST_F(VSerializerTest, SerializeDeserializeMessageWithJob) {
  Tensor tensor = make_tensor<float>({3, 3}, getCPU());
  std::vector<float> data(9);
  for (int i = 0; i < 9; ++i) {
    data[i] = static_cast<float>(i);
  }
  std::memcpy(tensor->data_as<float>(), data.data(), data.size() * sizeof(float));

  Job job(tensor, 100);
  MessageHeader header(CommandType::BACKWARD_JOB);
  MessageData msg_data(std::move(job));
  Message message(std::move(header), std::move(msg_data));

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(4096));
  buffer.resize(4096);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(message));

  // Deserialize
  offset = 0;
  Message deserialized_message;
  serializer_->deserialize(buffer, offset, deserialized_message);

  // Verify
  EXPECT_EQ(deserialized_message.header().command_type, CommandType::BACKWARD_JOB);
  ASSERT_TRUE(std::holds_alternative<Job>(deserialized_message.data().payload));

  const Job &deserialized_job = std::get<Job>(deserialized_message.data().payload);
  EXPECT_EQ(deserialized_job.mb_id, 100);

  std::vector<float> result_data(
      deserialized_job.data->data_as<float>(),
      deserialized_job.data->data_as<float>() + deserialized_job.data->size());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(result_data[i], data[i]);
  }
}

// Complex Scenarios

TEST_F(VSerializerTest, SerializeDeserializeMultipleTensors) {
  std::vector<Tensor> tensors;
  tensors.push_back(make_tensor<float>({5}, getCPU()));
  tensors.push_back(make_tensor<float>({3, 4}, getCPU()));
  tensors.push_back(make_tensor<float>({2, 2, 2}, getCPU()));

  for (auto &tensor : tensors) {
    size_t size = tensor->size();
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<float>(i);
    }
    std::memcpy(tensor->data_as<float>(), data.data(), data.size() * sizeof(float));
  }

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(8192));
  buffer.resize(8192);
  size_t offset = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    serializer_->serialize(buffer, offset, std::move(tensors[i]));
  }

  // Deserialize
  offset = 0;
  std::vector<Tensor> deserialized_tensors;
  for (size_t i = 0; i < 3; ++i) {
    Tensor tensor;
    serializer_->deserialize(buffer, offset, tensor);
    deserialized_tensors.push_back(tensor);
  }

  // Verify
  ASSERT_EQ(deserialized_tensors.size(), 3);

  EXPECT_EQ(deserialized_tensors[0]->shape().size(), 1);
  EXPECT_EQ(deserialized_tensors[0]->shape()[0], 5);

  EXPECT_EQ(deserialized_tensors[1]->shape().size(), 2);
  EXPECT_EQ(deserialized_tensors[1]->shape()[0], 3);
  EXPECT_EQ(deserialized_tensors[1]->shape()[1], 4);

  EXPECT_EQ(deserialized_tensors[2]->shape().size(), 3);
  EXPECT_EQ(deserialized_tensors[2]->shape()[0], 2);
  EXPECT_EQ(deserialized_tensors[2]->shape()[1], 2);
  EXPECT_EQ(deserialized_tensors[2]->shape()[2], 2);
}

TEST_F(VSerializerTest, SerializeDeserializeFullWorkflow) {
  // Simulate a complete message workflow
  Tensor tensor = make_tensor<float>({4, 4}, getCPU());
  std::vector<float> data(16);
  for (int i = 0; i < 16; ++i) {
    data[i] = static_cast<float>(i) * 0.5f;
  }
  std::memcpy(tensor->data_as<float>(), data.data(), data.size() * sizeof(float));

  Job job(tensor, 123);
  MessageHeader msg_header(CommandType::FORWARD_JOB);
  MessageData msg_data(std::move(job));
  Message message(std::move(msg_header), std::move(msg_data));

  PacketHeader pkt_header;
  pkt_header.type = PacketType::DATA_FRAGMENT;
  pkt_header.length = 2048;
  pkt_header.msg_length = 2048;
  pkt_header.msg_serial_id = 999;
  pkt_header.packet_offset = 0;
  pkt_header.total_packets = 1;
  pkt_header.compression_type = CompressionType::NONE;

  // Serialize packet header and message
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(8192));
  buffer.resize(8192);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, pkt_header);
  serializer_->serialize(buffer, offset, std::move(message));

  // Deserialize
  offset = 0;
  PacketHeader deserialized_pkt_header;
  Message deserialized_message;
  serializer_->deserialize(buffer, offset, deserialized_pkt_header);
  serializer_->deserialize(buffer, offset, deserialized_message);

  // Verify packet header
  EXPECT_EQ(deserialized_pkt_header.msg_serial_id, 999);
  EXPECT_EQ(deserialized_pkt_header.total_packets, 1);

  // Verify message
  EXPECT_EQ(deserialized_message.header().command_type, CommandType::FORWARD_JOB);
  ASSERT_TRUE(std::holds_alternative<Job>(deserialized_message.data().payload));

  const Job &deserialized_job = std::get<Job>(deserialized_message.data().payload);
  EXPECT_EQ(deserialized_job.mb_id, 123);

  std::vector<float> result_data(
      deserialized_job.data->data_as<float>(),
      deserialized_job.data->data_as<float>() + deserialized_job.data->size());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(result_data[i], data[i]);
  }
}

// Edge Cases and Error Handling

TEST_F(VSerializerTest, SerializeDeserializeEmptyTensor) {
  Tensor tensor = make_tensor<float>({0}, getCPU());

  // Serialize
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(1024));
  buffer.resize(1024);
  size_t offset = 0;
  serializer_->serialize(buffer, offset, std::move(tensor));

  // Deserialize
  offset = 0;
  Tensor deserialized_tensor;
  serializer_->deserialize(buffer, offset, deserialized_tensor);

  // Verify
  EXPECT_EQ(deserialized_tensor->size(), 0);
}

TEST_F(VSerializerTest, VariantIndexCompileTime) {
  // Test the variant_index compile-time function
  constexpr uint64_t monostate_idx = variant_index<PayloadType, std::monostate>();
  constexpr uint64_t job_idx = variant_index<PayloadType, Job>();
  constexpr uint64_t string_idx = variant_index<PayloadType, std::string>();
  constexpr uint64_t bool_idx = variant_index<PayloadType, bool>();

  EXPECT_EQ(monostate_idx, 0);
  EXPECT_EQ(job_idx, 1);
  EXPECT_EQ(string_idx, 2);
  EXPECT_EQ(bool_idx, 3);
}
