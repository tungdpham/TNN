#pragma once

#include "device/device.hpp"
#include <unordered_map>

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
  const Device &getDevice(int id) const;

private:
  std::unordered_map<int, Device> devices_;
};

void discoverDevices();
void initializeDefaultDevices();
} // namespace tdevice