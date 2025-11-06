#include "device/device_manager.hpp"

#include <iostream>
#include <stdexcept>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "device/cpu/cpu_context.hpp"
#ifdef USE_CUDA
#include "device/cuda/cuda_context.hpp"
#endif

namespace tdevice {
DeviceManager DeviceManager::instance_;

DeviceManager &DeviceManager::getInstance() { return instance_; }

DeviceManager::DeviceManager() = default;

DeviceManager::~DeviceManager() = default;

void DeviceManager::addDevice(Device &&device) {
  int id = device.getID();
  devices_.emplace(id, std::move(device));
}

void DeviceManager::removeDevice(int id) { devices_.erase(id); }

void DeviceManager::clearDevices() { devices_.clear(); }

const Device &DeviceManager::getDevice(int id) const {
  auto it = devices_.find(id);
  if (it != devices_.end()) {
    return it->second;
  }
  throw std::runtime_error("Device with the given ID not found");
}

std::vector<int> DeviceManager::getAvailableDeviceIDs() const {
  std::vector<int> ids;
  ids.reserve(devices_.size());
  for (const auto &pair : devices_) {
    ids.push_back(pair.first);
  }
  return ids;
}

bool DeviceManager::hasDevice(int id) const { return devices_.find(id) != devices_.end(); }

void initializeDefaultDevices() {
  DeviceManager &manager = DeviceManager::getInstance();

  // Clear any existing devices
  manager.clearDevices();

  size_t device_index = 0;
  // Always add CPU device with ID 0
  try {
    std::cout << "Discovered CPU device with ID: " << device_index << std::endl;
    Device cpu_device(DeviceType::CPU, device_index++, std::make_unique<CPUContext>());
    manager.addDevice(std::move(cpu_device));
  } catch (const std::exception &e) {
    std::cerr << "Failed to create CPU device: " << e.what() << std::endl;
  }

#ifdef USE_CUDA
  // Discover CUDA devices
  int cuda_device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&cuda_device_count);

  if (err == cudaSuccess && cuda_device_count > 0) {
    for (int i = 0; i < cuda_device_count; ++i) {
      try {
        // Get device properties for logging
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Discovered CUDA device with ID: " << device_index << " (CUDA Device " << i
                  << ": " << prop.name << ")" << std::endl;
        Device gpu_device(DeviceType::GPU, device_index++, std::make_unique<CUDAContext>(i));
        manager.addDevice(std::move(gpu_device));
      } catch (const std::exception &e) {
        std::cerr << "Failed to create CUDA device " << i << ": " << e.what() << std::endl;
      }
    }
  } else {
    std::cout << "No CUDA devices found or CUDA not available" << std::endl;
  }
#else
  std::cout << "CUDA support not compiled in" << std::endl;
#endif

  std::cout << "Default devices initialized" << std::endl;
}

} // namespace tdevice