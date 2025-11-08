#include "device/device_manager.hpp"
#include <iostream>
#include <vector>

using namespace tdevice;

int main() {
  try {
    std::cout << "=== Device Manager Example ===" << std::endl;

    // Initialize default devices
    std::cout << "\nInitializing devices..." << std::endl;
    initializeDefaultDevices();

    // Get device manager instance
    DeviceManager &manager = DeviceManager::getInstance();

    // List all available devices
    std::vector<int> device_ids = manager.getAvailableDeviceIDs();
    std::cout << "\nFound " << device_ids.size() << " device(s):" << std::endl;

    for (int device_id : device_ids) {
      const Device &device = manager.getDevice(device_id);
      std::cout << "  Device " << device_id << ": " << device.getName()
                << " (Type: " << (device.getDeviceType() == DeviceType::CPU ? "CPU" : "GPU") << ")"
                << std::endl;

      // Show memory information
      size_t total_mem = device.getTotalMemory();
      size_t avail_mem = device.getAvailableMemory();
      std::cout << "    Memory - Total: " << total_mem / (1024 * 1024) << " MB, "
                << "Available: " << avail_mem / (1024 * 1024) << " MB" << std::endl;
    }

    // Test allocation on CPU device (ID 0)
    if (manager.hasDevice(0)) {
      std::cout << "\nTesting allocation on CPU device..." << std::endl;
      const Device &cpu_device = manager.getDevice(0);

      size_t test_size = 1024 * 1024; // 1 MB
      void *ptr = cpu_device.allocateMemory(test_size);

      if (ptr != nullptr) {
        std::cout << "  Successfully allocated " << test_size << " bytes" << std::endl;

        // Test memory access (CPU only)
        memset(ptr, 0xAA, test_size);
        char *char_ptr = static_cast<char *>(ptr);
        if (char_ptr[0] == (char)0xAA && char_ptr[test_size - 1] == (char)0xAA) {
          std::cout << "  Memory access test passed" << std::endl;
        } else {
          std::cout << "  Memory access test failed" << std::endl;
        }

        cpu_device.deallocateMemory(ptr);
        std::cout << "  Successfully deallocated memory" << std::endl;
      } else {
        std::cout << "  Failed to allocate memory" << std::endl;
      }
    }

    // Test allocation on GPU device (if available)
    bool found_gpu = false;
    for (int device_id : device_ids) {
      if (device_id > 0) {
        const Device &device = manager.getDevice(device_id);
        if (device.getDeviceType() == DeviceType::GPU) {
          std::cout << "\nTesting allocation on GPU device " << device_id << "..." << std::endl;

          size_t test_size = 1024 * 1024 * 1024; // 1 GB
          void *ptr = device.allocateMemory(test_size);

          if (ptr != nullptr) {
            std::cout << "  Successfully allocated " << test_size << " bytes on GPU" << std::endl;
            device.deallocateMemory(ptr);
            std::cout << "  Successfully deallocated GPU memory" << std::endl;
          } else {
            std::cout << "  Failed to allocate GPU memory" << std::endl;
          }
          found_gpu = true;
          break;
        }
      }
    }

    if (!found_gpu) {
      std::cout << "\nNo GPU devices available for testing" << std::endl;
    }

    std::cout << "\n=== Example completed successfully ===" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}