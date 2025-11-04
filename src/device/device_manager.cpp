#include "device/device_manager.hpp"

#include <stdexcept>

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

const Device &DeviceManager::getDevice(int id) const {
  auto it = devices_.find(id);
  if (it != devices_.end()) {
    return it->second;
  }
  throw std::runtime_error("Device with the given ID not found");
}

} // namespace tdevice