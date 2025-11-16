#pragma once

#include "device.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {
class DeviceManager {
public:
  static DeviceManager &getInstance();

private:
  static DeviceManager instance_;

public:
  DeviceManager();
  ~DeviceManager();

  void discoverDevices();
  void addDevice(Device &&device);
  void removeDevice(std::string id);
  void clearDevices();
  const Device &getDevice(std::string id) const;
  std::vector<std::string> getAvailableDeviceIDs() const;
  bool hasDevice(std::string id) const;
  void setDefaultDevice(std::string id);
  void setDefaultDevice(const DeviceType &type);

private:
  std::unordered_map<std::string, Device> devices_;
  std::string default_device_id_;
};

void initializeDefaultDevices();
const Device &getGPU(size_t gpu_index = 0);
const Device &getCPU();

} // namespace tnn