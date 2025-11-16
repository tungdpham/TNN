#include "device/device.hpp"
#include "device/device_manager.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace tnn;

class DeviceManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize default devices before each test
    initializeDefaultDevices();
  }

  void TearDown() override {
    // Clean up after each test
    DeviceManager::getInstance().clearDevices();
  }
};

// Test device discovery and initialization
TEST_F(DeviceManagerTest, InitializeDefaultDevices) {
  DeviceManager &manager = DeviceManager::getInstance();

  // Should have at least CPU device (ID 0)
  EXPECT_TRUE(manager.hasDevice(0));

  std::vector<int> device_ids = manager.getAvailableDeviceIDs();
  EXPECT_GE(device_ids.size(), 1); // At least CPU device

  // Verify CPU device exists and is accessible
  const Device &cpu_device = manager.getDevice(0);
  EXPECT_EQ(cpu_device.getDeviceType(), DeviceType::CPU);
  EXPECT_EQ(cpu_device.getID(), 0);
}

// Test device discovery
TEST_F(DeviceManagerTest, DiscoverDevices) {
  initializeDefaultDevices();

  DeviceManager &manager = DeviceManager::getInstance();

  // Should have at least CPU device
  EXPECT_TRUE(manager.hasDevice(0));

  std::vector<int> device_ids = manager.getAvailableDeviceIDs();
  EXPECT_GE(device_ids.size(), 1);

  // Check that device ID 0 is CPU
  const Device &cpu_device = manager.getDevice(0);
  EXPECT_EQ(cpu_device.getDeviceType(), DeviceType::CPU);
}

// Test device allocation and deallocation
TEST_F(DeviceManagerTest, CPUDeviceAllocation) {
  DeviceManager &manager = DeviceManager::getInstance();

  ASSERT_TRUE(manager.hasDevice(0));
  const Device &cpu_device = manager.getDevice(0);

  // Test basic allocation
  size_t size = 1024;
  void *ptr = cpu_device.allocateMemory(size);
  ASSERT_NE(ptr, nullptr);

  // Test that allocated memory is accessible
  memset(ptr, 0x42, size);
  char *char_ptr = static_cast<char *>(ptr);
  EXPECT_EQ(char_ptr[0], 0x42);
  EXPECT_EQ(char_ptr[size - 1], 0x42);

  // Test deallocation
  EXPECT_NO_THROW(cpu_device.deallocateMemory(ptr));
}

// Test multiple device allocation
TEST_F(DeviceManagerTest, MultipleDeviceAllocations) {
  DeviceManager &manager = DeviceManager::getInstance();

  std::vector<int> device_ids = manager.getAvailableDeviceIDs();

  for (int device_id : device_ids) {
    const Device &device = manager.getDevice(device_id);

    // Test allocation on each device
    size_t size = 512;
    void *ptr = device.allocateMemory(size);
    ASSERT_NE(ptr, nullptr) << "Failed to allocate on device " << device_id;

    // For CPU devices, we can test memory access
    if (device.getDeviceType() == DeviceType::CPU) {
      memset(ptr, static_cast<int>(device_id & 0xFF), size);
      char *char_ptr = static_cast<char *>(ptr);
      EXPECT_EQ(char_ptr[0], static_cast<char>(device_id & 0xFF));
    }

    EXPECT_NO_THROW(device.deallocateMemory(ptr));
  }
}

// Test device manager utility functions
TEST_F(DeviceManagerTest, DeviceManagerUtilities) {
  DeviceManager &manager = DeviceManager::getInstance();

  // Test getAvailableDeviceIDs
  std::vector<int> device_ids = manager.getAvailableDeviceIDs();
  EXPECT_GE(device_ids.size(), 1);

  // Test hasDevice
  for (int device_id : device_ids) {
    EXPECT_TRUE(manager.hasDevice(device_id));
  }

  // Test with non-existent device
  EXPECT_FALSE(manager.hasDevice(9999));
}

// Test device removal
TEST_F(DeviceManagerTest, DeviceRemoval) {
  DeviceManager &manager = DeviceManager::getInstance();

  // Get initial device count
  std::vector<int> initial_ids = manager.getAvailableDeviceIDs();
  size_t initial_count = initial_ids.size();

  if (initial_count > 1) {
    // Remove a device (not CPU)
    int device_to_remove = initial_ids[1];
    manager.removeDevice(device_to_remove);

    // Verify it's removed
    EXPECT_FALSE(manager.hasDevice(device_to_remove));

    std::vector<int> remaining_ids = manager.getAvailableDeviceIDs();
    EXPECT_EQ(remaining_ids.size(), initial_count - 1);
  }
}

// Test error conditions
TEST_F(DeviceManagerTest, ErrorConditions) {
  DeviceManager &manager = DeviceManager::getInstance();

  // Test getting non-existent device
  EXPECT_THROW(manager.getDevice(9999), std::runtime_error);

  // Test removing non-existent device (should not throw)
  EXPECT_NO_THROW(manager.removeDevice(9999));
}

// Test clear devices
TEST_F(DeviceManagerTest, ClearDevices) {
  DeviceManager &manager = DeviceManager::getInstance();

  // Ensure we have devices
  std::vector<int> device_ids = manager.getAvailableDeviceIDs();
  EXPECT_GT(device_ids.size(), 0);

  // Clear all devices
  manager.clearDevices();

  // Verify no devices remain
  std::vector<int> remaining_ids = manager.getAvailableDeviceIDs();
  EXPECT_EQ(remaining_ids.size(), 0);

  // Verify hasDevice returns false for previously existing devices
  for (int device_id : device_ids) {
    EXPECT_FALSE(manager.hasDevice(device_id));
  }
}

#ifdef USE_CUDA
// Test CUDA device discovery (if available)
TEST_F(DeviceManagerTest, CUDADeviceDiscovery) {
  DeviceManager &manager = DeviceManager::getInstance();

  std::vector<int> device_ids = manager.getAvailableDeviceIDs();

  // Look for GPU devices (ID > 0)
  bool found_gpu = false;
  for (int device_id : device_ids) {
    if (device_id > 0) {
      const Device &device = manager.getDevice(device_id);
      if (device.getDeviceType() == DeviceType::GPU) {
        found_gpu = true;

        // Test basic GPU allocation
        void *ptr = device.allocateMemory(1024);
        if (ptr != nullptr) {
          EXPECT_NO_THROW(device.deallocateMemory(ptr));
        }
        break;
      }
    }
  }

  // Note: This test doesn't fail if no GPU is found, as it's environment-dependent
  if (found_gpu) {
    std::cout << "CUDA GPU device found and tested successfully" << std::endl;
  } else {
    std::cout << "No CUDA GPU devices found (this is not an error)" << std::endl;
  }
}
#endif

// Test device memory info
TEST_F(DeviceManagerTest, DeviceMemoryInfo) {
  DeviceManager &manager = DeviceManager::getInstance();

  std::vector<int> device_ids = manager.getAvailableDeviceIDs();

  for (int device_id : device_ids) {
    const Device &device = manager.getDevice(device_id);

    // Test memory info methods
    size_t total_memory = device.getTotalMemory();
    size_t available_memory = device.getAvailableMemory();

    EXPECT_GT(total_memory, 0) << "Device " << device_id << " reports zero total memory";
    EXPECT_LE(available_memory, total_memory)
        << "Available memory exceeds total memory for device " << device_id;

    std::cout << "Device " << device_id << " (" << device.getName()
              << "): " << "Total: " << total_memory / (1024 * 1024) << " MB, "
              << "Available: " << available_memory / (1024 * 1024) << " MB" << std::endl;
  }
}