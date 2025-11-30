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

  /**
   * Discover all available devices on the system.
   */
  void discoverDevices();

  /**
   * Add a device to the manager.
   * @param device The device to add.
   */
  void addDevice(Device &&device);

  /**
   * Remove a device from the manager.
   * @param id The ID of the device to remove.
   */
  void removeDevice(std::string id);

  /**
   * Clear all devices from the manager.
   */
  void clearDevices();

  /**
   * Get a device by its ID.
   * @param id The ID of the device to retrieve.
   * @return The device with the specified ID.
   */
  const Device &getDevice(std::string id) const;
  const Device &getDevice(DeviceType type) const;

  /**
   * Get all available device ids.
   */
  std::vector<std::string> getAvailableDeviceIDs() const;

  /**
   * Check if device manager has a device with an id.
   */
  bool hasDevice(std::string id) const;

  /**
   * Set the default device using an id
   */
  void setDefaultDevice(std::string id);

  /**
   */
  void setDefaultDevice(const DeviceType &type);

private:
  std::unordered_map<std::string, Device> devices_;
  std::string default_device_id_;
};

void initializeDefaultDevices();
const Device &getGPU(size_t gpu_index = 0);
const Device &getCPU();

} // namespace tnn