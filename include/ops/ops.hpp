#pragma once

#include "device/dptr.hpp"
#include "ops/cpu/kernels.hpp"
#ifdef USE_CUDA
#include "ops/cuda/kernels.hpp"
#endif
#include "device/task.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace tnn {
namespace ops {

template <typename T>
std::unique_ptr<Task> add(dptr a, dptr b, dptr c, size_t size,
                          const std::string &flow_id = "default") {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("add: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("add: All device pointers must be on the same device");
  }
  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::add<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_add<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> sub(dptr a, dptr b, dptr c, size_t size,
                          const std::string &flow_id = "default") {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("sub: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("sub: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::sub<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_sub<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> mul(dptr a, dptr b, dptr c, size_t size,
                          const std::string &flow_id = "default") {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("mul: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("mul: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::mul<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_mul<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> div(dptr a, dptr b, dptr c, size_t size,
                          const std::string &flow_id = "default") {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("div: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("div: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::div<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_div<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fmadd(dptr a, dptr b, dptr c, size_t size,
                            const std::string &flow_id = "default") {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("fmadd: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("fmadd: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::fmadd<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_fmadd<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fmsub(dptr a, dptr b, dptr c, size_t size,
                            const std::string &flow_id = "default") {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("fmsub: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("fmsub: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::fmsub<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_fmsub<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fnmadd(dptr a, dptr b, dptr c, size_t size,
                             const std::string &flow_id = "default") {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("fnmadd: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("fnmadd: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::fnmadd<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_fnmadd<T>, a.get<T>(), b.get<T>(), c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> add_scalar(dptr a, T scalar, dptr c, size_t size,
                                 const std::string &flow_id = "default") {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("add_scalar: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("add_scalar: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::add_scalar<T>, a.get<T>(), scalar, c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_add_scalar<T>, a.get<T>(), scalar, c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> sub_scalar(dptr a, T scalar, dptr c, size_t size,
                                 const std::string &flow_id = "default") {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("sub_scalar: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("sub_scalar: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::sub_scalar<T>, a.get<T>(), scalar, c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_sub_scalar<T>, a.get<T>(), scalar, c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> mul_scalar(dptr a, T scalar, dptr c, size_t size,
                                 const std::string &flow_id = "default") {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("mul_scalar: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    std::cout << "Device pointers: " << a.getDevice() << " " << c.getDevice() << std::endl;
    throw std::runtime_error("mul_scalar: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::mul_scalar<T>, a.get<T>(), scalar, c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_mul_scalar<T>, a.get<T>(), scalar, c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> div_scalar(dptr a, T scalar, dptr c, size_t size,
                                 const std::string &flow_id = "default") {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("div_scalar: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("div_scalar: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::div_scalar<T>, a.get<T>(), scalar, c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_div_scalar<T>, a.get<T>(), scalar, c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> set_scalar(dptr c, T scalar, size_t size,
                                 const std::string &flow_id = "default") {
  if (!c.getDevice()) {
    throw std::runtime_error("set_scalar: Device pointer has no associated device");
  }

  auto device = c.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::set_scalar<T>, c.get<T>(), scalar, size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_set_scalar<T>, c.get<T>(), scalar, size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> axpy(T alpha, dptr x, dptr y, size_t size,
                           const std::string &flow_id = "default") {
  if (!x.getDevice() || !y.getDevice()) {
    throw std::runtime_error("axpy: Device pointer has no associated device");
  }
  if (x.getDevice() != y.getDevice()) {
    throw std::runtime_error("axpy: All device pointers must be on the same device");
  }

  auto device = x.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::axpy<T>, alpha, x.get<T>(), y.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_axpy<T>, alpha, x.get<T>(), y.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> sqrt(dptr a, dptr c, size_t size, const std::string &flow_id = "default") {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("sqrt: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("sqrt: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::sqrt<T>, a.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_sqrt<T>, a.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
inline std::unique_ptr<Task> rsqrt(dptr a, dptr c, size_t size,
                                   const std::string &flow_id = "default") {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("rsqrt: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("rsqrt: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::rsqrt<T>, a.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_rsqrt<T>, a.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
inline std::unique_ptr<Task> rcp(dptr a, dptr c, size_t size,
                                 const std::string &flow_id = "default") {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("rcp: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("rcp: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::rcp<T>, a.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_rcp<T>, a.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> abs(dptr a, dptr c, size_t size, const std::string &flow_id = "default") {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("abs: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("abs: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::abs<T>, a.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_abs<T>, a.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> min(dptr a, dptr b, dptr c, size_t size,
                          const std::string &flow_id = "default") {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("min: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("min: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::min<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_min<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> max(dptr a, dptr b, dptr c, size_t size,
                          const std::string &flow_id = "default") {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("max: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("max: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::max<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_max<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> scalar_max(dptr a, T scalar, dptr c, size_t size,
                                 const std::string &flow_id = "default") {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("scalar_max: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("scalar_max: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::scalar_max<T>, a.get<T>(), scalar, c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_scalar_max<T>, a.get<T>(), scalar, c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> clamp(dptr a, T min_val, T max_val, dptr c, size_t size,
                            const std::string &flow_id = "default") {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("clamp: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("clamp: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::clamp<T>, a.get<T>(), min_val, max_val, c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_clamp<T>, a.get<T>(), min_val, max_val, c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> equal(dptr a, dptr b, dptr c, size_t size,
                            const std::string &flow_id = "default") {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("equal: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("equal: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::equal<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_equal<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> greater(dptr a, dptr b, dptr c, size_t size,
                              const std::string &flow_id = "default") {
  if (!a.getDevice() || !b.getDevice() || !c.getDevice()) {
    throw std::runtime_error("greater: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice() || a.getDevice() != c.getDevice()) {
    throw std::runtime_error("greater: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::greater<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_greater<T>, a.get<T>(), b.get<T>(), c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> copy(dptr a, dptr c, size_t size, const std::string &flow_id = "default") {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("copy: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("copy: All device pointers must be on the same device");
  }

  if (a.get<T>() == nullptr || c.get<T>() == nullptr) {
    throw std::runtime_error("copy: Null pointer exception in copy operation");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::copy<T>, a.get<T>(), c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_copy<T>, a.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Special copy for copying cross devices (resort to same device/host copy if applicable)
template <typename T>
std::unique_ptr<Task> cd_copy(dptr a, dptr c, size_t size, const std::string &flow_id = "default") {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("cd_copy: Device pointer has no associated device");
  }
  auto a_device = a.getDevice();
  auto c_device = c.getDevice();
  if (a_device == c_device) {
    // same device copy
    return copy<T>(a, c, size, flow_id);
  }
  auto a_device_type = a_device->device_type();
  auto c_device_type = c_device->device_type();

  if (a_device_type == DeviceType::CPU && c_device_type == DeviceType::GPU) {
    // host to device copy
#ifdef USE_CUDA
    return create_cuda_task(flow_id, cuda::cuda_h2d_copy<T>, a.get<T>(), c.get<T>(), size);
#else
    throw std::runtime_error("cd_copy: CUDA not enabled for CPU to GPU copy");
#endif
  } else if (a_device_type == DeviceType::GPU && c_device_type == DeviceType::CPU) {
    // device to host copy
#ifdef USE_CUDA
    return create_cuda_task(flow_id, cuda::cuda_d2h_copy<T>, a.get<T>(), c.get<T>(), size);
#else
    throw std::runtime_error("cd_copy: CUDA not enabled for GPU to CPU copy");
#endif
  } else {
    throw std::runtime_error("cd_copy: Unsupported device type combination");
  }
}

template <typename T>
std::unique_ptr<Task> zero(dptr c, size_t size, const std::string &flow_id = "default") {
  if (!c.getDevice()) {
    throw std::runtime_error("zero: Device pointer has no associated device");
  }

  auto device = c.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::zero<T>, c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_zero<T>, c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> T sum(dptr a, size_t size) {
  if (!a.getDevice()) {
    throw std::runtime_error("sum: Device pointer has no associated device");
  }

  auto device_type = a.getDevice()->device_type();

  if (device_type == DeviceType::CPU) {
    return cpu::sum(a.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return cuda::cuda_sum(a.get<T>(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> T dot_product(dptr a, dptr b, size_t size) {
  if (!a.getDevice() || !b.getDevice()) {
    throw std::runtime_error("dot_product: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice()) {
    throw std::runtime_error("dot_product: All device pointers must be on the same device");
  }

  auto device_type = a.getDevice()->device_type();

  if (device_type == DeviceType::CPU) {
    return cpu::dot_product(a.get<T>(), b.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return cuda::cuda_dot_product(a.get<T>(), b.get<T>(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> T norm_squared(dptr a, size_t size) {
  if (!a.getDevice()) {
    throw std::runtime_error("norm_squared: Device pointer has no associated device");
  }

  auto device_type = a.getDevice()->device_type();

  if (device_type == DeviceType::CPU) {
    return cpu::norm_squared(a.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return cuda::cuda_norm_squared(a.get<T>(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> T sum_squared_diff(dptr a, T mean, size_t size) {
  if (!a.getDevice()) {
    throw std::runtime_error("sum_squared_diff: Device pointer has no associated device");
  }

  auto device_type = a.getDevice()->device_type();

  if (device_type == DeviceType::CPU) {
    return cpu::sum_squared_diff(a.get<T>(), mean, size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return cuda::cuda_sum_squared_diff(a.get<T>(), mean, size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> sub_mul_scalar(dptr a, T sub_scalar, T mul_scalar, dptr c, size_t size,
                                     const std::string &flow_id = "default") {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("sub_mul_scalar: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("sub_mul_scalar: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::sub_mul_scalar<T>, a.get<T>(), sub_scalar, mul_scalar,
                           c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_sub_mul_scalar<T>, a.get<T>(), sub_scalar,
                            mul_scalar, c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> mul_add_scalar(dptr a, T mul_scalar, T add_scalar, dptr c, size_t size,
                                     const std::string &flow_id = "default") {
  if (!a.getDevice() || !c.getDevice()) {
    throw std::runtime_error("mul_add_scalar: Device pointer has no associated device");
  }
  if (a.getDevice() != c.getDevice()) {
    throw std::runtime_error("mul_add_scalar: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::mul_add_scalar<T>, a.get<T>(), mul_scalar, add_scalar,
                           c.get<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_mul_add_scalar<T>, a.get<T>(), mul_scalar,
                            add_scalar, c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fill_random_uniform(dptr data, size_t size, T min_val, T max_val,
                                          unsigned long long seed,
                                          const std::string &flow_id = "default") {
  if (!data.getDevice()) {
    throw std::runtime_error("fill_random_uniform: Device pointer has no associated device");
  }

  auto device = data.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::fill_random_uniform<T>, data.get<T>(), size, min_val,
                           max_val, seed);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_fill_random_uniform<T>, data.get<T>(), size,
                            min_val, max_val, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fill_random_normal(dptr data, size_t size, T mean, T stddev,
                                         unsigned long long seed,
                                         const std::string &flow_id = "default") {
  if (!data.getDevice()) {
    throw std::runtime_error("fill_random_normal: Device pointer has no associated device");
  }

  auto device = data.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::fill_random_normal<T>, data.get<T>(), size, mean, stddev,
                           seed);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_fill_random_normal<T>, data.get<T>(), size, mean,
                            stddev, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename A_T, typename B_T>
std::unique_ptr<Task> cast(dptr a, dptr b, size_t size, const std::string &flow_id = "default") {
  if (!a.getDevice() || !b.getDevice()) {
    throw std::runtime_error("cast: Device pointer has no associated device");
  }
  if (a.getDevice() != b.getDevice()) {
    throw std::runtime_error("cast: All device pointers must be on the same device");
  }

  auto device = a.getDevice();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::cast<A_T, B_T>, a.get<A_T>(), b.get<B_T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, cuda::cuda_cast<A_T, B_T>, a.get<A_T>(), b.get<B_T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}
} // namespace ops
} // namespace tnn