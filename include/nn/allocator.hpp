#pragma once

#include "device/device_ptr.hpp"
#include <unordered_map>
namespace tnn {

template <typename T> class Allocator {
public:
  Allocator() = default;
  ~Allocator() = default;

  Allocator(const Allocator &) = delete;
  Allocator &operator=(const Allocator &) = delete;

  device_ptr<T[]> get_buffer(size_t size, const Device *device) {
    auto it_device_pool = buffer_pool_.find(const_cast<Device *>(device));
    if (it_device_pool != buffer_pool_.end()) {
      auto &buffers = it_device_pool->second;
      for (auto &buffer : buffers) {
        if (buffer.size() >= size) {
          device_ptr<T[]> ret_buffer = buffer;
          buffer.reset();
          return ret_buffer;
        }
      }
    }
    return make_array_ptr<T[]>(device, size);
  }

  void return_buffer(device_ptr<T[]> buffer) {
    Device *device = const_cast<Device *>(buffer.getDevice());
    if (!device) {
      throw std::runtime_error("Buffer has no associated device.");
    }
    auto it_device_pool = buffer_pool_.find(device);
    if (it_device_pool == buffer_pool_.end()) {
      buffer_pool_[device] = std::vector<device_ptr<T[]>>();
    }
    buffer_pool_[device].push_back(buffer);
  }

private:
  void allocate_buffer(size_t size, const Device *device) {
    auto it_device_pool = buffer_pool_.find(device);
    if (it_device_pool == buffer_pool_.end()) {
      buffer_pool_[device] = std::vector<device_ptr<T[]>>();
    }
    device_ptr<T[]> buffer = make_array_ptr<T[]>(device, size);
    buffer_pool_[device].push_back(buffer);
  }

private:
  std::unordered_map<Device *, std::vector<device_ptr<T[]>>> buffer_pool_;
};
} // namespace tnn