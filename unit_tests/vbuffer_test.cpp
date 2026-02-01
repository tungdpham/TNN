/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "distributed/vbuffer.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "device/dptr.hpp"
#include "device/sref.hpp"

using namespace tnn;

class VBufferTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { initializeDefaultDevices(); }

  void SetUp() override {}

  void TearDown() override {}

  static void TearDownTestSuite() {}

  sref<IAllocator> allocator_ = DeviceAllocator::instance(getCPU());
};

// Basic Construction and Properties Tests

TEST_F(VBufferTest, DefaultConstructor) {
  VBuffer buffer;
  EXPECT_EQ(buffer.size(), 0);
  EXPECT_EQ(buffer.capacity(), 0);
  EXPECT_TRUE(buffer.empty());
}

TEST_F(VBufferTest, AllocSingleBuffer) {
  VBuffer buffer;
  dptr ptr = allocator_->allocate(1024);
  buffer.alloc(std::move(ptr));

  EXPECT_EQ(buffer.capacity(), 1024);
  EXPECT_EQ(buffer.size(), 0);  // Size is separate from capacity
  EXPECT_FALSE(buffer.empty());
}

TEST_F(VBufferTest, AllocMultipleBuffers) {
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(512));
  buffer.alloc(allocator_->allocate(256));
  buffer.alloc(allocator_->allocate(1024));

  EXPECT_EQ(buffer.capacity(), 512 + 256 + 1024);
  EXPECT_FALSE(buffer.empty());
}

TEST_F(VBufferTest, ClearBuffer) {
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(1024));

  EXPECT_FALSE(buffer.empty());
  buffer.clear();

  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.size(), 0);
  EXPECT_EQ(buffer.capacity(), 0);
}

TEST_F(VBufferTest, ResetBuffer) {
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(1024));
  buffer.resize(512);

  EXPECT_EQ(buffer.size(), 512);
  buffer.reset();

  EXPECT_EQ(buffer.size(), 0);
  EXPECT_EQ(buffer.capacity(), 1024);  // Capacity unchanged
}

TEST_F(VBufferTest, ResizeBuffer) {
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(1024));

  buffer.resize(512);
  EXPECT_EQ(buffer.size(), 512);

  buffer.resize(256);
  EXPECT_EQ(buffer.size(), 256);

  buffer.resize(1024);
  EXPECT_EQ(buffer.size(), 1024);
}

TEST_F(VBufferTest, ResizeExceedsCapacity) {
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(512));

  EXPECT_THROW(buffer.resize(1024), std::runtime_error);
}

// Write/Read Primitive Types Tests

TEST_F(VBufferTest, WriteReadInt32) {
  VBuffer buffer;
  dptr ptr = allocator_->allocate(1024);
  buffer.alloc(std::move(ptr));
  buffer.resize(1024);

  size_t offset = 0;
  int32_t write_value = 12345;
  buffer.write(offset, write_value);

  offset = 0;
  int32_t read_value = 0;
  buffer.read(offset, read_value);

  EXPECT_EQ(write_value, read_value);
  EXPECT_EQ(offset, sizeof(int32_t));
}

TEST_F(VBufferTest, WriteReadUInt64) {
  VBuffer buffer;
  dptr ptr = allocator_->allocate(1024);
  buffer.alloc(std::move(ptr));
  buffer.resize(1024);

  size_t offset = 0;
  uint64_t write_value = 0xDEADBEEFCAFEBABE;
  buffer.write(offset, write_value);

  offset = 0;
  uint64_t read_value = 0;
  buffer.read(offset, read_value);

  EXPECT_EQ(write_value, read_value);
}

TEST_F(VBufferTest, WriteReadFloat) {
  VBuffer buffer;
  dptr ptr = allocator_->allocate(1024);
  buffer.alloc(std::move(ptr));
  buffer.resize(1024);

  size_t offset = 0;
  float write_value = 3.14159f;
  buffer.write(offset, write_value);

  offset = 0;
  float read_value = 0.0f;
  buffer.read(offset, read_value);

  EXPECT_FLOAT_EQ(write_value, read_value);
}

TEST_F(VBufferTest, WriteReadDouble) {
  VBuffer buffer;
  dptr ptr = allocator_->allocate(1024);
  buffer.alloc(std::move(ptr));
  buffer.resize(1024);

  size_t offset = 0;
  double write_value = 2.718281828459045;
  buffer.write(offset, write_value);

  offset = 0;
  double read_value = 0.0;
  buffer.read(offset, read_value);

  EXPECT_DOUBLE_EQ(write_value, read_value);
}

TEST_F(VBufferTest, WriteReadMultipleValues) {
  VBuffer buffer;
  dptr ptr = allocator_->allocate(1024);
  buffer.alloc(std::move(ptr));
  buffer.resize(1024);

  size_t offset = 0;
  int32_t val1 = 42;
  uint64_t val2 = 123456789ULL;
  float val3 = 1.23f;
  double val4 = 4.56;

  buffer.write(offset, val1);
  buffer.write(offset, val2);
  buffer.write(offset, val3);
  buffer.write(offset, val4);

  offset = 0;
  int32_t read1;
  uint64_t read2;
  float read3;
  double read4;

  buffer.read(offset, read1);
  buffer.read(offset, read2);
  buffer.read(offset, read3);
  buffer.read(offset, read4);

  EXPECT_EQ(val1, read1);
  EXPECT_EQ(val2, read2);
  EXPECT_FLOAT_EQ(val3, read3);
  EXPECT_DOUBLE_EQ(val4, read4);
}

// String Tests

TEST_F(VBufferTest, WriteReadEmptyString) {
  VBuffer buffer;
  dptr ptr = allocator_->allocate(1024);
  buffer.alloc(std::move(ptr));
  buffer.resize(1024);

  size_t offset = 0;
  std::string write_str = "";
  buffer.write(offset, write_str);

  offset = 0;
  std::string read_str;
  buffer.read(offset, read_str);

  EXPECT_EQ(write_str, read_str);
}

TEST_F(VBufferTest, WriteReadShortString) {
  VBuffer buffer;
  dptr ptr = allocator_->allocate(1024);
  buffer.alloc(std::move(ptr));
  buffer.resize(1024);

  size_t offset = 0;
  std::string write_str = "Hello, World!";
  buffer.write(offset, write_str);

  offset = 0;
  std::string read_str;
  buffer.read(offset, read_str);

  EXPECT_EQ(write_str, read_str);
}

TEST_F(VBufferTest, WriteReadLongString) {
  VBuffer buffer;
  dptr ptr = allocator_->allocate(2048);
  buffer.alloc(std::move(ptr));
  buffer.resize(2048);

  size_t offset = 0;
  std::string write_str(512, 'A');
  write_str += "Some special characters: !@#$%^&*()";
  buffer.write(offset, write_str);

  offset = 0;
  std::string read_str;
  buffer.read(offset, read_str);

  EXPECT_EQ(write_str, read_str);
}

TEST_F(VBufferTest, WriteReadMultipleStrings) {
  VBuffer buffer;
  dptr ptr = allocator_->allocate(2048);
  buffer.alloc(std::move(ptr));
  buffer.resize(2048);

  size_t offset = 0;
  std::string str1 = "First";
  std::string str2 = "Second String";
  std::string str3 = "Third String with More Characters";

  buffer.write(offset, str1);
  buffer.write(offset, str2);
  buffer.write(offset, str3);

  offset = 0;
  std::string read1, read2, read3;
  buffer.read(offset, read1);
  buffer.read(offset, read2);
  buffer.read(offset, read3);

  EXPECT_EQ(str1, read1);
  EXPECT_EQ(str2, read2);
  EXPECT_EQ(str3, read3);
}

// Array Tests

TEST_F(VBufferTest, WriteReadIntArray) {
  VBuffer buffer;
  dptr ptr = allocator_->allocate(1024);
  buffer.alloc(std::move(ptr));
  buffer.resize(1024);

  size_t offset = 0;
  int32_t write_arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  size_t arr_len = sizeof(write_arr) / sizeof(write_arr[0]);

  buffer.write(offset, write_arr, arr_len);

  offset = 0;
  int32_t read_arr[10];
  buffer.read(offset, read_arr, arr_len);

  for (size_t i = 0; i < arr_len; ++i) {
    EXPECT_EQ(write_arr[i], read_arr[i]);
  }
}

TEST_F(VBufferTest, WriteReadFloatArray) {
  VBuffer buffer;
  dptr ptr = allocator_->allocate(1024);
  buffer.alloc(std::move(ptr));
  buffer.resize(1024);

  size_t offset = 0;
  float write_arr[] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
  size_t arr_len = sizeof(write_arr) / sizeof(write_arr[0]);

  buffer.write(offset, write_arr, arr_len);

  offset = 0;
  float read_arr[5];
  buffer.read(offset, read_arr, arr_len);

  for (size_t i = 0; i < arr_len; ++i) {
    EXPECT_FLOAT_EQ(write_arr[i], read_arr[i]);
  }
}

// Append Tests

TEST_F(VBufferTest, AppendSingleBuffer) {
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(512));

  dptr append_ptr = allocator_->allocate(256);
  size_t prev_capacity = buffer.capacity();

  buffer.append(std::move(append_ptr));

  EXPECT_EQ(buffer.capacity(), prev_capacity + 256);
  EXPECT_EQ(buffer.size(), 256);
}

TEST_F(VBufferTest, AppendMultipleBuffers) {
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(512));

  buffer.append(allocator_->allocate(128));
  buffer.append(allocator_->allocate(256));
  buffer.append(allocator_->allocate(64));

  EXPECT_EQ(buffer.size(), 128 + 256 + 64);
}

TEST_F(VBufferTest, AppendWithData) {
  VBuffer buffer;
  dptr ptr = allocator_->allocate(1024);
  buffer.alloc(std::move(ptr));
  buffer.resize(1024);

  // Write some data
  size_t offset = 0;
  int32_t value = 42;
  buffer.write(offset, value);

  // Create another buffer with data
  dptr append_ptr = allocator_->allocate(sizeof(int32_t));
  int32_t append_value = 99;
  append_ptr.copy_from_host(&append_value, sizeof(int32_t));

  buffer.append(std::move(append_ptr));

  // Read back both values
  offset = 0;
  int32_t read1, read2;
  buffer.read(offset, read1);
  buffer.read(offset, read2);

  EXPECT_EQ(read1, value);
  EXPECT_EQ(read2, append_value);
}

// Get Tests

TEST_F(VBufferTest, GetFromSingleBuffer) {
  VBuffer buffer;
  dptr ptr = allocator_->allocate(1024);
  buffer.alloc(std::move(ptr));
  buffer.resize(1024);

  // Write some data
  size_t offset = 0;
  int32_t value = 12345;
  buffer.write(offset, value);

  // Get dptr at offset 0
  dptr data_ptr = buffer.get(0);
  EXPECT_GE(data_ptr.capacity(), sizeof(int32_t));

  // Read from the dptr
  int32_t read_value;
  data_ptr.copy_to_host(&read_value, sizeof(int32_t));
  EXPECT_EQ(read_value, value);
}

TEST_F(VBufferTest, GetFromMultipleBuffers) {
  VBuffer buffer;
  dptr ptr1 = allocator_->allocate(512);
  int32_t val1 = 111;
  ptr1.copy_from_host(&val1, sizeof(int32_t));
  buffer.alloc(std::move(ptr1));
  buffer.resize(512);

  dptr ptr2 = allocator_->allocate(512);
  int32_t val2 = 222;
  ptr2.copy_from_host(&val2, sizeof(int32_t));
  buffer.append(std::move(ptr2));

  // Get from first buffer
  dptr data_ptr1 = buffer.get(0);
  int32_t read1;
  data_ptr1.copy_to_host(&read1, sizeof(int32_t));
  EXPECT_EQ(read1, val1);

  // Get from second buffer
  dptr data_ptr2 = buffer.get(512);
  int32_t read2;
  data_ptr2.copy_to_host(&read2, sizeof(int32_t));
  EXPECT_EQ(read2, val2);
}

TEST_F(VBufferTest, GetOutOfRange) {
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(512));
  buffer.resize(512);

  EXPECT_THROW(buffer.get(1024), std::out_of_range);
}

TEST_F(VBufferTest, GetFromEmptyBuffer) {
  VBuffer buffer;
  EXPECT_THROW(buffer.get(0), std::runtime_error);
}

// Poll Tests

TEST_F(VBufferTest, PollSingleBuffer) {
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(512));
  buffer.resize(512);

  EXPECT_FALSE(buffer.empty());
  dptr polled = buffer.poll();

  EXPECT_EQ(polled.capacity(), 512);
  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.size(), 0);
}

TEST_F(VBufferTest, PollMultipleBuffers) {
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(256));
  buffer.alloc(allocator_->allocate(512));
  buffer.alloc(allocator_->allocate(128));
  buffer.resize(256 + 512 + 128);

  dptr first = buffer.poll();
  EXPECT_EQ(first.capacity(), 256);
  EXPECT_EQ(buffer.size(), 512 + 128);

  dptr second = buffer.poll();
  EXPECT_EQ(second.capacity(), 512);
  EXPECT_EQ(buffer.size(), 128);

  dptr third = buffer.poll();
  EXPECT_EQ(third.capacity(), 128);
  EXPECT_TRUE(buffer.empty());
}

TEST_F(VBufferTest, PollEmptyBuffer) {
  VBuffer buffer;
  EXPECT_THROW(buffer.poll(), std::runtime_error);
}

// Edge Cases and Error Handling

TEST_F(VBufferTest, WriteAtOffsetBeyondSize) {
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(512));
  buffer.resize(512);

  size_t offset = 1024;  // Beyond size
  int32_t value = 42;

  EXPECT_THROW(buffer.write(offset, value), std::out_of_range);
}

TEST_F(VBufferTest, ReadAtOffsetBeyondSize) {
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(512));
  buffer.resize(512);

  size_t offset = 1024;  // Beyond size
  int32_t value;

  EXPECT_THROW(buffer.read(offset, value), std::out_of_range);
}

TEST_F(VBufferTest, WriteReadAtBoundary) {
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(sizeof(int32_t)));
  buffer.resize(sizeof(int32_t));

  size_t offset = 0;
  int32_t value = 42;
  buffer.write(offset, value);

  offset = 0;
  int32_t read_value;
  buffer.read(offset, read_value);

  EXPECT_EQ(value, read_value);
}

TEST_F(VBufferTest, MoveConstructor) {
  VBuffer buffer1;
  buffer1.alloc(allocator_->allocate(512));
  buffer1.resize(512);

  size_t offset = 0;
  int32_t value = 42;
  buffer1.write(offset, value);

  VBuffer buffer2(std::move(buffer1));

  offset = 0;
  int32_t read_value;
  buffer2.read(offset, read_value);

  EXPECT_EQ(value, read_value);
  EXPECT_EQ(buffer2.size(), 512);
}

TEST_F(VBufferTest, MoveAssignment) {
  VBuffer buffer1;
  buffer1.alloc(allocator_->allocate(512));
  buffer1.resize(512);

  size_t offset = 0;
  int32_t value = 42;
  buffer1.write(offset, value);

  VBuffer buffer2;
  buffer2 = std::move(buffer1);

  offset = 0;
  int32_t read_value;
  buffer2.read(offset, read_value);

  EXPECT_EQ(value, read_value);
  EXPECT_EQ(buffer2.size(), 512);
}

TEST_F(VBufferTest, ComplexDataFlow) {
  VBuffer buffer;
  buffer.alloc(allocator_->allocate(2048));
  buffer.resize(2048);

  // Write a complex sequence of data
  size_t offset = 0;

  // Header
  uint32_t magic = 0xDEADBEEF;
  buffer.write(offset, magic);

  // Version
  uint16_t version = 1;
  buffer.write(offset, version);

  // String
  std::string message = "Complex data structure";
  buffer.write(offset, message);

  // Array
  float values[] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
  buffer.write(offset, values, 5);

  // Footer
  uint64_t checksum = 0xCAFEBABE;
  buffer.write(offset, checksum);

  // Read back and verify
  offset = 0;

  uint32_t read_magic;
  buffer.read(offset, read_magic);
  EXPECT_EQ(magic, read_magic);

  uint16_t read_version;
  buffer.read(offset, read_version);
  EXPECT_EQ(version, read_version);

  std::string read_message;
  buffer.read(offset, read_message);
  EXPECT_EQ(message, read_message);

  float read_values[5];
  buffer.read(offset, read_values, 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(values[i], read_values[i]);
  }

  uint64_t read_checksum;
  buffer.read(offset, read_checksum);
  EXPECT_EQ(checksum, read_checksum);
}
