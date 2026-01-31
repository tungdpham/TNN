#include <iostream>
#include <string>
#include <vector>

#include "device/device_manager.hpp"

using namespace tnn;
using namespace std;

int main() {
  try {
    cout << "=== Device Manager Example ===" << endl;

    cout << "Initializing devices..." << endl;
    initializeDefaultDevices();

    DeviceManager &manager = DeviceManager::getInstance();

    vector<string> device_ids = manager.getAvailableDeviceIDs();
    cout << "Found " << device_ids.size() << " device(s):" << endl;

    for (const string &device_id : device_ids) {
      const Device &device = manager.getDevice(device_id);
      cout << "  Device " << device_id << ": " << device.getName()
           << " (Type: " << (device.device_type() == DeviceType::CPU ? "CPU" : "GPU") << ")"
           << endl;

      size_t total_mem = device.getTotalMemory();
      size_t avail_mem = device.getAvailableMemory();
      cout << "    Memory - Total: " << total_mem / (1024 * 1024) << " MB, "
           << "Available: " << avail_mem / (1024 * 1024) << " MB" << endl;
    }

    if (manager.hasDevice(0)) {
      cout << "Testing allocation on CPU device..." << endl;
      const Device &cpu_device = manager.getDevice(0);

      size_t test_size = 1024 * 1024;
      void *ptr = cpu_device.allocateMemory(test_size);

      if (ptr != nullptr) {
        cout << "  Successfully allocated " << test_size << " bytes" << endl;

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

    bool found_gpu = false;
    for (const string &device_id : device_ids) {
      if (device_id > "0") {
        const Device &device = manager.getDevice(device_id);
        if (device.device_type() == DeviceType::GPU) {
          cout << "Testing allocation on GPU device " << device_id << "..." << endl;

          size_t test_size = 1024 * 1024 * 1024;
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
      cout << "No GPU devices available for testing" << endl;
    }

  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  return 0;
}