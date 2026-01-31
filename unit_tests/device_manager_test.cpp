#include "device/device_manager.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "device/device.hpp"

using namespace tnn;

class DeviceManagerTest : public ::testing::Test {
protected:
  void SetUp() override { initializeDefaultDevices(); }

  void TearDown() override { DeviceManager::getInstance().clearDevices(); }
};

TEST_F(DeviceManagerTest, InitializeDefaultDevices) {
  DeviceManager &manager = DeviceManager::getInstance();

  EXPECT_TRUE(manager.hasDevice("CPU:0"));

  std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();
  EXPECT_GE(device_ids.size(), 1);

  const Device &cpu_device = manager.getDevice("CPU:0");
  EXPECT_EQ(cpu_device.device_type(), DeviceType::CPU);
  EXPECT_EQ(cpu_device.getID(), 0);
}

TEST_F(DeviceManagerTest, DiscoverDevices) {
  initializeDefaultDevices();

  DeviceManager &manager = DeviceManager::getInstance();

  EXPECT_TRUE(manager.hasDevice("CPU:0"));

  std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();
  EXPECT_GE(device_ids.size(), 1);

  const Device &cpu_device = manager.getDevice("CPU:0");
  EXPECT_EQ(cpu_device.device_type(), DeviceType::CPU);
}

TEST_F(DeviceManagerTest, CPUDeviceAllocation) {
  DeviceManager &manager = DeviceManager::getInstance();

  ASSERT_TRUE(manager.hasDevice("CPU:0"));
  const Device &cpu_device = manager.getDevice("CPU:0");

  size_t size = 1024;
  void *ptr = cpu_device.allocateMemory(size);
  ASSERT_NE(ptr, nullptr);

  memset(ptr, 0x42, size);
  char *char_ptr = static_cast<char *>(ptr);
  EXPECT_EQ(char_ptr[0], 0x42);
  EXPECT_EQ(char_ptr[size - 1], 0x42);

  EXPECT_NO_THROW(cpu_device.deallocateMemory(ptr));
}

TEST_F(DeviceManagerTest, MultipleDeviceAllocations) {
  DeviceManager &manager = DeviceManager::getInstance();

  std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();

  for (const std::string &device_id : device_ids) {
    const Device &device = manager.getDevice(device_id);

    size_t size = 512;
    void *ptr = device.allocateMemory(size);
    ASSERT_NE(ptr, nullptr) << "Failed to allocate on device " << device_id;

    if (device.device_type() == DeviceType::CPU) {
      memset(ptr, static_cast<int>(device.getID() & 0xFF), size);
      char *char_ptr = static_cast<char *>(ptr);
      EXPECT_EQ(char_ptr[0], static_cast<char>(device.getID() & 0xFF));
    }

    EXPECT_NO_THROW(device.deallocateMemory(ptr));
  }
}

TEST_F(DeviceManagerTest, DeviceManagerUtilities) {
  DeviceManager &manager = DeviceManager::getInstance();

  std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();
  EXPECT_GE(device_ids.size(), 1);

  for (const std::string &device_id : device_ids) {
    EXPECT_TRUE(manager.hasDevice(device_id));
  }

  EXPECT_FALSE(manager.hasDevice("GPU:9999"));
}

TEST_F(DeviceManagerTest, DeviceRemoval) {
  DeviceManager &manager = DeviceManager::getInstance();

  std::vector<std::string> initial_ids = manager.getAvailableDeviceIDs();
  size_t initial_count = initial_ids.size();

  if (initial_count > 1) {
    std::string device_to_remove = initial_ids[1];
    manager.removeDevice(device_to_remove);

    EXPECT_FALSE(manager.hasDevice(device_to_remove));

    std::vector<std::string> remaining_ids = manager.getAvailableDeviceIDs();
    EXPECT_EQ(remaining_ids.size(), initial_count - 1);
  }
}

TEST_F(DeviceManagerTest, ErrorConditions) {
  DeviceManager &manager = DeviceManager::getInstance();

  EXPECT_THROW(manager.getDevice("GPU:9999"), std::runtime_error);

  EXPECT_NO_THROW(manager.removeDevice("GPU:9999"));
}

TEST_F(DeviceManagerTest, ClearDevices) {
  DeviceManager &manager = DeviceManager::getInstance();

  std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();
  EXPECT_GT(device_ids.size(), 0);

  manager.clearDevices();

  std::vector<std::string> remaining_ids = manager.getAvailableDeviceIDs();
  EXPECT_EQ(remaining_ids.size(), 0);

  for (const std::string &device_id : device_ids) {
    EXPECT_FALSE(manager.hasDevice(device_id));
  }
}

#ifdef USE_CUDA

TEST_F(DeviceManagerTest, CUDADeviceDiscovery) {
  DeviceManager &manager = DeviceManager::getInstance();

  std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();

  bool found_gpu = false;
  for (const std::string &device_id : device_ids) {
    if (device_id.find("GPU") == 0) {
      const Device &device = manager.getDevice(device_id);
      if (device.device_type() == DeviceType::GPU) {
        found_gpu = true;

        void *ptr = device.allocateMemory(1024);
        if (ptr != nullptr) {
          EXPECT_NO_THROW(device.deallocateMemory(ptr));
        }
        break;
      }
    }
  }

  if (found_gpu) {
    std::cout << "CUDA GPU device found and tested successfully" << std::endl;
  } else {
    std::cout << "No CUDA GPU devices found (this is not an error)" << std::endl;
  }
}
#endif

TEST_F(DeviceManagerTest, DeviceMemoryInfo) {
  DeviceManager &manager = DeviceManager::getInstance();

  std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();

  for (const std::string &device_id : device_ids) {
    const Device &device = manager.getDevice(device_id);

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