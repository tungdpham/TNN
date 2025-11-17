#pragma once

#include "cpu/kernels.hpp"
#include "device/device_ptr.hpp"
#ifdef USE_CUDA
#include "cuda/kernels.hpp"
#endif
#include "device/task.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tnn {
namespace ops {

template <typename Func, typename... Args>
std::unique_ptr<Task> create_cpu_task(const Device *device, Func &&func, Args &&...args) {
  return std::make_unique<CPUTask>(device, std::forward<Func>(func), std::forward<Args>(args)...);
}

#ifdef USE_CUDA
template <typename Func, typename... Args>
std::unique_ptr<Task> create_gpu_task(const Device *device, Func &&func, Args &&...args) {
  return std::make_unique<CUDATask>(device, std::forward<Func>(func), std::forward<Args>(args)...);
}
#endif

template <typename T>
std::unique_ptr<Task> add(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                          size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("add: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("add: All device pointers must be on the same device");
  }
  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::add<T>, a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_add<T>, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> sub(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                          size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("sub: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("sub: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::sub<T>, a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_sub<T>, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> mul(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                          size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("mul: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("mul: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::mul<T>, a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_mul<T>, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> div(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                          size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("div: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("div: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::div<T>, a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_div<T>, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fmadd(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                            size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("fmadd: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("fmadd: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::fmadd<T>, a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_fmadd<T>, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fmsub(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                            size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("fmsub: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("fmsub: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::fmsub<T>, a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_fmsub<T>, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fnmadd(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                             size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("fnmadd: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("fnmadd: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::fnmadd<T>, a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_fnmadd<T>, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> add_scalar(const device_ptr<T[]> &a, T scalar, device_ptr<T[]> &c,
                                 size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("add_scalar: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("add_scalar: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::add_scalar<T>, a.get(), scalar, c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_add_scalar<T>, a.get(), scalar, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> mul_scalar(const device_ptr<T[]> &a, T scalar, device_ptr<T[]> &c,
                                 size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("mul_scalar: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    std::cout << "Device pointers: " << a.getDevice() << " " << c.getDevice() << std::endl;
    throw std::runtime_error("mul_scalar: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::mul_scalar<T>, a.get(), scalar, c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_mul_scalar<T>, a.get(), scalar, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> div_scalar(const device_ptr<T[]> &a, T scalar, device_ptr<T[]> &c,
                                 size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("div_scalar: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("div_scalar: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::div_scalar<T>, a.get(), scalar, c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_div_scalar<T>, a.get(), scalar, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> std::unique_ptr<Task> set_scalar(device_ptr<T[]> &c, T scalar, size_t size) {
  if (!c.getDevice()) {
    throw std::runtime_error("set_scalar: Device pointer has no associated device");
  }

  auto device = c.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::set_scalar<T>, c.get(), scalar, size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_set_scalar<T>, c.get(), scalar, size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> sqrt(const device_ptr<T[]> &a, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("sqrt: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("sqrt: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::sqrt<T>, a.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_sqrt<T>, a.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline std::unique_ptr<Task> rsqrt(const device_ptr<float[]> &a, device_ptr<float[]> &c,
                                   size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("rsqrt: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("rsqrt: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::rsqrt<float>, a.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_rsqrt, a.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline std::unique_ptr<Task> rcp(const device_ptr<float[]> &a, device_ptr<float[]> &c,
                                 size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("rcp: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("rcp: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::rcp<float>, a.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_rcp, a.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> abs(const device_ptr<T[]> &a, device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("abs: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("abs: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::abs<T>, a.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_abs<T>, a.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> min(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                          size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("min: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("min: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::min<T>, a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_min<T>, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> max(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                          size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("max: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("max: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::max<T>, a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_max<T>, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> scalar_max(const device_ptr<T[]> &a, T scalar, device_ptr<T[]> &c,
                                 size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("scalar_max: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("scalar_max: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::scalar_max<T>, a.get(), scalar, c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_scalar_max<T>, a.get(), scalar, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> clamp(const device_ptr<T[]> &a, T min_val, T max_val, device_ptr<T[]> &c,
                            size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("clamp: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("clamp: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::clamp<T>, a.get(), min_val, max_val, c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_clamp<T>, a.get(), min_val, max_val, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> equal(const device_ptr<T[]> &a, const device_ptr<T[]> &b, device_ptr<T[]> &c,
                            size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("equal: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("equal: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::equal<T>, a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_equal<T>, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> greater(const device_ptr<T[]> &a, const device_ptr<T[]> &b,
                              device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("greater: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("greater: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::greater<T>, a.get(), b.get(), c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_greater<T>, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> copy(const device_ptr<T[]> &a, device_ptr<T[]> &c, size_t size,
                           size_t a_offset = 0, size_t c_offset = 0) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("copy: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("copy: All device pointers must be on the same device");
  }

  if (a.get() == nullptr || c.get() == nullptr) {
    throw std::runtime_error("copy: Null pointer exception in copy operation");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::copy<T>, a.get() + a_offset, c.get() + c_offset, size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_copy<T>, a.get() + a_offset, c.get() + c_offset,
                           size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> std::unique_ptr<Task> zero(device_ptr<T[]> &c, size_t size) {
  if (!c.getDevice()) {
    throw std::runtime_error("zero: Device pointer has no associated device");
  }

  auto device = c.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::zero<T>, c.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_zero<T>, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> T sum(const device_ptr<T[]> &a, size_t size) {
  if (!a.getDevice()) {
    throw std::runtime_error("sum: Device pointer has no associated device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return cpu::sum(a.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return cuda::cuda_sum(a.get(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
T dot_product(const device_ptr<T[]> &a, const device_ptr<T[]> &b, size_t size) {
  if (!a.getDevice() || !b.getDevice()) {
    throw std::runtime_error("dot_product: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice()) {
    throw std::runtime_error("dot_product: All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return cpu::dot_product(a.get(), b.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return cuda::cuda_dot_product(a.get(), b.get(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> T norm_squared(const device_ptr<T[]> &a, size_t size) {
  if (!a.getDevice()) {
    throw std::runtime_error("norm_squared: Device pointer has no associated device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return cpu::norm_squared(a.get(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return cuda::cuda_norm_squared(a.get(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> T sum_squared_diff(const device_ptr<T[]> &a, T mean, size_t size) {
  if (!a.getDevice()) {
    throw std::runtime_error("sum_squared_diff: Device pointer has no associated device");
  }

  auto device_type = a.getDevice()->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return cpu::sum_squared_diff(a.get(), mean, size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return cuda::cuda_sum_squared_diff(a.get(), mean, size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> sub_mul_scalar(const device_ptr<T[]> &a, T sub_scalar, T mul_scalar,
                                     device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("sub_mul_scalar: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("sub_mul_scalar: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::sub_mul_scalar<T>, a.get(), sub_scalar, mul_scalar, c.get(),
                           size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_sub_mul_scalar<T>, a.get(), sub_scalar, mul_scalar,
                           c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> mul_add_scalar(const device_ptr<T[]> &a, T mul_scalar, T add_scalar,
                                     device_ptr<T[]> &c, size_t size) {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("mul_add_scalar: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("mul_add_scalar: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::mul_add_scalar<T>, a.get(), mul_scalar, add_scalar, c.get(),
                           size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_mul_add_scalar<T>, a.get(), mul_scalar, add_scalar,
                           c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fill_random_uniform(device_ptr<T[]> &data, size_t size, T min_val, T max_val,
                                          unsigned long long seed) {
  if (!data.getDevice()) {
    throw std::runtime_error("fill_random_uniform: Device pointer has no associated device");
  }

  auto device = data.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::fill_random_uniform<T>, data.get(), size, min_val, max_val,
                           seed);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_fill_random_uniform<T>, data.get(), size, min_val,
                           max_val, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fill_random_normal(device_ptr<T[]> &data, size_t size, T mean, T stddev,
                                         unsigned long long seed) {
  if (!data.getDevice()) {
    throw std::runtime_error("fill_random_normal: Device pointer has no associated device");
  }

  auto device = data.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::fill_random_normal<T>, data.get(), size, mean, stddev,
                           seed);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_fill_random_normal<T>, data.get(), size, mean, stddev,
                           seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> transpose_2d(const device_ptr<T[]> &input, device_ptr<T[]> &output,
                                   size_t rows, size_t cols) {
  if (!input.getDevice() || !output.getDevice()) {
    throw std::runtime_error("transpose_2d: Device pointer has no associated device");
  }

  if (output.getDevice() != input.getDevice()) {
    throw std::runtime_error("transpose_2d: Input and output must be on the same device");
  }

  auto device = input.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::transpose_2d<T>, input.get(), output.get(), rows, cols);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_transpose_2d<T>, input.get(), output.get(), rows,
                           cols);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> nchw_to_cnhw(const device_ptr<T[]> &input, device_ptr<T[]> &output, size_t n,
                                   size_t c, size_t h, size_t w) {
  if (!input.getDevice() || !output.getDevice()) {
    throw std::runtime_error("nchw_to_cnhw: Device pointer has no associated device");
  }

  if (output.getDevice() != input.getDevice()) {
    throw std::runtime_error("nchw_to_cnhw: Input and output must be on the same device");
  }

  auto device = input.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::nchw_to_cnhw<T>, input.get(), output.get(), n, c, h, w);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_nchw_to_cnhw<T>, input.get(), output.get(), n, c, h,
                           w);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> cnhw_to_nchw(const device_ptr<T[]> &input, device_ptr<T[]> &output, size_t n,
                                   size_t c, size_t h, size_t w) {
  if (!input.getDevice() || !output.getDevice()) {
    throw std::runtime_error("cnhw_to_nchw: Device pointer has no associated device");
  }
  if (output.getDevice() != input.getDevice()) {
    throw std::runtime_error("cnhw_to_nchw: Input and output must be on the same device");
  }
  auto device = input.getDevice();
  auto device_type = device->getDeviceType();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, cpu::cnhw_to_nchw<T>, input.get(), output.get(), n, c, h, w);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_gpu_task(device, cuda::cuda_cnhw_to_nchw<T>, input.get(), output.get(), n, c, h,
                           w);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

} // namespace ops
} // namespace tnn