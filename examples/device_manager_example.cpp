#include "device/device_manager.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace tnn;
using namespace std;

int main() {
  try {
    cout << "=== Device Manager Example ===" << endl;

    // Initialize default devices
    cout << "\nInitializing devices..." << endl;
    initializeDefaultDevices();

    // Get device manager instance
    DeviceManager &manager = DeviceManager::getInstance();

    // List all available devices
    vector<string> device_ids = manager.getAvailableDeviceIDs();
    cout << "\nFound " << device_ids.size() << " device(s):" << endl;

    for (const string &device_id : device_ids) {
      const Device &device = manager.getDevice(device_id);
      cout << "  Device " << device_id << ": " << device.getName()
           << " (Type: " << (device.device_type() == DeviceType::CPU ? "CPU" : "GPU") << ")"
           << endl;

      // Show memory information
      size_t total_mem = device.getTotalMemory();
      size_t avail_mem = device.getAvailableMemory();
      cout << "    Memory - Total: " << total_mem / (1024 * 1024) << " MB, "
           << "Available: " << avail_mem / (1024 * 1024) << " MB" << endl;
    }

    // Test allocation on CPU device (ID 0)
    if (manager.hasDevice(0)) {
      cout << "\nTesting allocation on CPU device..." << endl;
      const Device &cpu_device = manager.getDevice(0);

      size_t test_size = 1024 * 1024; // 1 MB
      void *ptr = cpu_device.allocateMemory(test_size);

      if (ptr != nullptr) {
        cout << "  Successfully allocated " << test_size << " bytes" << endl;

        // Test memory access (CPU only)
        memset(ptr, 0xAA, test_size);
        char *char_ptr = static_cast<char *>(ptr);
        if (char_ptr[0] == (char)0xAA && char_ptr[test_size - 1] == (char)0xAA) {
          cout << "  Memory access test passed" << endl;
        } else {
          cout << "  Memory access test failed" << endl;
        }

        cpu_device.deallocateMemory(ptr);
        cout << "  Successfully deallocated memory" << endl;
      } else {
        cout << "  Failed to allocate memory" << endl;
      }
    }

    // Test allocation on GPU device (if available)
    bool found_gpu = false;
    for (const string &device_id : device_ids) {
      if (device_id > "0") { // Assuming GPU devices have IDs greater than 0
        const Device &device = manager.getDevice(device_id);
        if (device.device_type() == DeviceType::GPU) {
          cout << "\nTesting allocation on GPU device " << device_id << "..." << endl;

          size_t test_size = 1024 * 1024 * 1024; // 1 GB
          void *ptr = device.allocateMemory(test_size);

          if (ptr != nullptr) {
            cout << "  Successfully allocated " << test_size << " bytes on GPU" << endl;
            device.deallocateMemory(ptr);
            cout << "  Successfully deallocated GPU memory" << endl;
          } else {
            cout << "  Failed to allocate GPU memory" << endl;
          }
          found_gpu = true;
          break;
        }
      }
    }

    if (!found_gpu) {
      cout << "\nNo GPU devices available for testing" << endl;
    }

    cout << "\n=== Example completed successfully ===" << endl;

  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  return 0;
}