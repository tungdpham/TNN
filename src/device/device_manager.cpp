#include "device/device_manager.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "device/cpu/cpu_context.hpp"
#ifdef USE_CUDA
#include "device/cuda/cuda_context.hpp"
#endif

namespace tnn {
DeviceManager DeviceManager::instance_;

DeviceManager &DeviceManager::getInstance() { return instance_; }

DeviceManager::DeviceManager() { discoverDevices(); }

void DeviceManager::discoverDevices() {
  clearDevices();

  size_t device_index = 0;
  // Always add CPU device with ID 0
  try {
    std::cout << "Discovered CPU device with ID: " << device_index << std::endl;
    Device cpu_device(DeviceType::CPU, device_index++, std::make_unique<CPUContext>());
    addDevice(std::move(cpu_device));
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
        std::cout << "Discovered CUDA device with ID: " << i << " (CUDA Device " << i << ": "
                  << prop.name << ")" << std::endl;
        Device gpu_device(DeviceType::GPU, i, std::make_unique<CUDAContext>(i));
        addDevice(std::move(gpu_device));
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

DeviceManager::~DeviceManager() = default;

void DeviceManager::addDevice(Device &&device) {
  std::string device_type = (device.device_type() == DeviceType::CPU) ? "CPU" : "GPU";
  int id = device.getID();
  devices_.emplace(device_type + ":" + std::to_string(id), std::move(device));
}

void DeviceManager::removeDevice(std::string id) { devices_.erase(id); }

void DeviceManager::clearDevices() { devices_.clear(); }

const Device &DeviceManager::getDevice(std::string id) const {
  auto it = devices_.find(id);
  if (it != devices_.end()) {
    return it->second;
  }
  throw std::runtime_error("Device with the given ID not found");
}

const Device &DeviceManager::getDevice(DeviceType type) const {
  for (const auto &pair : devices_) {
    if (pair.second.device_type() == type) {
      return pair.second;
    }
  }
  throw std::runtime_error("No device of the specified type found");
}

std::vector<std::string> DeviceManager::getAvailableDeviceIDs() const {
  std::vector<std::string> ids;
  ids.reserve(devices_.size());
  for (const auto &pair : devices_) {
    ids.push_back(pair.first);
  }
  return ids;
}

bool DeviceManager::hasDevice(std::string id) const { return devices_.find(id) != devices_.end(); }

void DeviceManager::setDefaultDevice(std::string id) {
  if (hasDevice(id)) {
    default_device_id_ = id;
  } else {
    throw std::runtime_error("Device with the given ID not found");
  }
}

void DeviceManager::setDefaultDevice(const DeviceType &type) {
  for (const auto &pair : devices_) {
    if (pair.second.device_type() == type) {
      default_device_id_ = pair.first;
      return;
    }
  }
  throw std::runtime_error("No device of the specified type found");
}

void initializeDefaultDevices() {
  DeviceManager &manager = DeviceManager::getInstance();

  manager.discoverDevices();
}

const Device &getGPU(size_t gpu_index) {
  DeviceManager &manager = DeviceManager::getInstance();
  size_t current_gpu = 0;
  for (std::string id : manager.getAvailableDeviceIDs()) {
    const Device &device = manager.getDevice(id);
    if (device.device_type() == DeviceType::GPU) {
      if (current_gpu == gpu_index) {
        return device;
      }
      current_gpu++;
    }
  }
  throw std::runtime_error("Requested GPU index not found");
}

const Device &getCPU() {
  DeviceManager &manager = DeviceManager::getInstance();
  for (std::string id : manager.getAvailableDeviceIDs()) {
    const Device &device = manager.getDevice(id);
    if (device.device_type() == DeviceType::CPU) {
      return device;
    }
  }
  throw std::runtime_error("CPU device not found");
}

} // namespace tnn