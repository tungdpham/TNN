#pragma once

#include "device.hpp"
#include <unordered_map>
#include <vector>

namespace tdevice {
class DeviceManager {
public:
  static DeviceManager &getInstance();

private:
  static DeviceManager instance_;

public:
  DeviceManager();
  ~DeviceManager();

  void addDevice(Device &&device);
  void removeDevice(int id);
  void clearDevices();
  const Device &getDevice(int id) const;
  std::vector<int> getAvailableDeviceIDs() const;
  bool hasDevice(int id) const;

private:
  std::unordered_map<int, Device> devices_;
};

void initializeDefaultDevices();
const Device &getGPU(size_t gpu_index = 0);
const Device &getCPU();

} // namespace tdevice