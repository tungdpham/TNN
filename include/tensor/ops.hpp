#pragma once

#include "ops/cpu/kernels.hpp"
#ifdef USE_CUDA
#include "ops/cuda/kernels.hpp"
#endif
#include "device/task.hpp"
#include "tensor/tensor.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace tnn {
namespace TensorOps {
template <typename T>
std::unique_ptr<Task> add(const Tensor &a, const Tensor &b, Tensor &c, size_t size,
                          const std::string &flow_id = "default") {
  if (!a->device() || !b->device() || !c->device()) {
    throw std::runtime_error("add: Device pointer has no associated device");
  }
  if (a->device() != b->device() || a->device() != c->device()) {
    throw std::runtime_error("add: All device pointers must be on the same device");
  }
  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::add<T>, a->data_as<T>(), b->data_as<T>(),
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_add<T>, a->data_as<T>(), b->data_as<T>(),
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> sub(const Tensor &a, const Tensor &b, Tensor &c, size_t size,
                          const std::string &flow_id = "default") {
  if (!a->device() || !b->device() || !c->device()) {
    throw std::runtime_error("sub: Device pointer has no associated device");
  }
  if (a->device() != b->device() || a->device() != c->device()) {
    throw std::runtime_error("sub: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::sub<T>, a->data_as<T>(), b->data_as<T>(),
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_sub<T>, a->data_as<T>(), b->data_as<T>(),
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> mul(const Tensor &a, const Tensor &b, Tensor &c, size_t size,
                          const std::string &flow_id = "default") {
  if (!a->device() || !b->device() || !c->device()) {
    throw std::runtime_error("mul: Device pointer has no associated device");
  }
  if (a->device() != b->device() || a->device() != c->device()) {
    throw std::runtime_error("mul: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::mul<T>, a->data_as<T>(), b->data_as<T>(),
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_mul<T>, a->data_as<T>(), b->data_as<T>(),
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> div(const Tensor &a, const Tensor &b, Tensor &c, size_t size,
                          const std::string &flow_id = "default") {
  if (!a->device() || !b->device() || !c->device()) {
    throw std::runtime_error("div: Device pointer has no associated device");
  }
  if (a->device() != b->device() || a->device() != c->device()) {
    throw std::runtime_error("div: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::div<T>, a->data_as<T>(), b->data_as<T>(),
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_div<T>, a->data_as<T>(), b->data_as<T>(),
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fmadd(const Tensor &a, const Tensor &b, Tensor &c, size_t size,
                            const std::string &flow_id = "default") {
  if (!a->device() || !b->device() || !c->device()) {
    throw std::runtime_error("fmadd: Device pointer has no associated device");
  }
  if (a->device() != b->device() || a->device() != c->device()) {
    throw std::runtime_error("fmadd: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::fmadd<T>, a->data_as<T>(), b->data_as<T>(),
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_fmadd<T>, a->data_as<T>(), b->data_as<T>(),
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fmsub(const Tensor &a, const Tensor &b, Tensor &c, size_t size,
                            const std::string &flow_id = "default") {
  if (!a->device() || !b->device() || !c->device()) {
    throw std::runtime_error("fmsub: Device pointer has no associated device");
  }
  if (a->device() != b->device() || a->device() != c->device()) {
    throw std::runtime_error("fmsub: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::fmsub<T>, a->data_as<T>(), b->data_as<T>(),
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_fmsub<T>, a->data_as<T>(), b->data_as<T>(),
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fnmadd(const Tensor &a, const Tensor &b, Tensor &c, size_t size,
                             const std::string &flow_id = "default") {
  if (!a->device() || !b->device() || !c->device()) {
    throw std::runtime_error("fnmadd: Device pointer has no associated device");
  }
  if (a->device() != b->device() || a->device() != c->device()) {
    throw std::runtime_error("fnmadd: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::fnmadd<T>, a->data_as<T>(), b->data_as<T>(),
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_fnmadd<T>, a->data_as<T>(), b->data_as<T>(),
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> add_scalar(const Tensor &a, T scalar, Tensor &c, size_t size,
                                 const std::string &flow_id = "default") {
  if (!a->device() || !c->device()) {
    throw std::runtime_error("add_scalar: Device pointer has no associated device");
  }
  if (a->device() != c->device()) {
    throw std::runtime_error("add_scalar: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::add_scalar<T>, a->data_as<T>(), scalar,
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_add_scalar<T>, a->data_as<T>(), scalar,
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> sub_scalar(const Tensor &a, T scalar, Tensor &c, size_t size,
                                 const std::string &flow_id = "default") {
  if (!a->device() || !c->device()) {
    throw std::runtime_error("sub_scalar: Device pointer has no associated device");
  }
  if (a->device() != c->device()) {
    throw std::runtime_error("sub_scalar: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::sub_scalar<T>, a->data_as<T>(), scalar,
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_sub_scalar<T>, a->data_as<T>(), scalar,
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> mul_scalar(const Tensor &a, T scalar, Tensor &c, size_t size,
                                 const std::string &flow_id = "default") {
  if (!a->device() || !c->device()) {
    throw std::runtime_error("mul_scalar: Device pointer has no associated device");
  }
  if (a->device() != c->device()) {
    std::cout << "Device pointers: " << a->device() << " " << c->device() << std::endl;
    throw std::runtime_error("mul_scalar: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::mul_scalar<T>, a->data_as<T>(), scalar,
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_mul_scalar<T>, a->data_as<T>(), scalar,
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> div_scalar(const Tensor &a, T scalar, Tensor &c, size_t size,
                                 const std::string &flow_id = "default") {
  if (!a->device() || !c->device()) {
    throw std::runtime_error("div_scalar: Device pointer has no associated device");
  }
  if (a->device() != c->device()) {
    throw std::runtime_error("div_scalar: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::div_scalar<T>, a->data_as<T>(), scalar,
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_div_scalar<T>, a->data_as<T>(), scalar,
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> set_scalar(Tensor &c, T scalar, size_t size,
                                 const std::string &flow_id = "default") {
  if (!c->device()) {
    throw std::runtime_error("set_scalar: Device pointer has no associated device");
  }

  auto device = c->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::set_scalar<T>, c->data_as<T>(), scalar, size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_set_scalar<T>, c->data_as<T>(), scalar, size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> axpy(T alpha, const Tensor &x, Tensor &y, size_t size,
                           const std::string &flow_id = "default") {
  if (!x->device() || !y->device()) {
    throw std::runtime_error("axpy: Device pointer has no associated device");
  }
  if (x->device() != y->device()) {
    throw std::runtime_error("axpy: All device pointers must be on the same device");
  }

  auto device = x->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::axpy<T>, alpha, x->data_as<T>(), y->data_as<T>(),
                           size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_axpy<T>, alpha, x->data_as<T>(),
                            y->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> sqrt(const Tensor &a, Tensor &c, size_t size,
                           const std::string &flow_id = "default") {
  if (!a->device() || !c->device()) {
    throw std::runtime_error("sqrt: Device pointer has no associated device");
  }
  if (a->device() != c->device()) {
    throw std::runtime_error("sqrt: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::sqrt<T>, a->data_as<T>(), c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_sqrt<T>, a->data_as<T>(), c->data_as<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
inline std::unique_ptr<Task> rsqrt(const Tensor &a, Tensor &c, size_t size,
                                   const std::string &flow_id = "default") {
  if (!a->device() || !c->device()) {
    throw std::runtime_error("rsqrt: Device pointer has no associated device");
  }
  if (a->device() != c->device()) {
    throw std::runtime_error("rsqrt: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::rsqrt<T>, a->data_as<T>(), c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_rsqrt<T>, a->data_as<T>(), c->data_as<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
inline std::unique_ptr<Task> rcp(const Tensor &a, Tensor &c, size_t size,
                                 const std::string &flow_id = "default") {
  if (!a->device() || !c->device()) {
    throw std::runtime_error("rcp: Device pointer has no associated device");
  }
  if (a->device() != c->device()) {
    throw std::runtime_error("rcp: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::rcp<T>, a->data_as<T>(), c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_rcp<T>, a->data_as<T>(), c->data_as<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> abs(const Tensor &a, Tensor &c, size_t size,
                          const std::string &flow_id = "default") {
  if (!a->device() || !c->device()) {
    throw std::runtime_error("abs: Device pointer has no associated device");
  }
  if (a->device() != c->device()) {
    throw std::runtime_error("abs: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::abs<T>, a->data_as<T>(), c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_abs<T>, a->data_as<T>(), c->data_as<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> min(const Tensor &a, const Tensor &b, Tensor &c, size_t size,
                          const std::string &flow_id = "default") {
  if (!a->device() || !b->device() || !c->device()) {
    throw std::runtime_error("min: Device pointer has no associated device");
  }
  if (a->device() != b->device() || a->device() != c->device()) {
    throw std::runtime_error("min: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::min<T>, a->data_as<T>(), b->data_as<T>(),
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_min<T>, a->data_as<T>(), b->data_as<T>(),
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> max(const Tensor &a, const Tensor &b, Tensor &c, size_t size,
                          const std::string &flow_id = "default") {
  if (!a->device() || !b->device() || !c->device()) {
    throw std::runtime_error("max: Device pointer has no associated device");
  }
  if (a->device() != b->device() || a->device() != c->device()) {
    throw std::runtime_error("max: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::max<T>, a->data_as<T>(), b->data_as<T>(),
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_max<T>, a->data_as<T>(), b->data_as<T>(),
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> scalar_max(const Tensor &a, T scalar, Tensor &c, size_t size,
                                 const std::string &flow_id = "default") {
  if (!a->device() || !c->device()) {
    throw std::runtime_error("scalar_max: Device pointer has no associated device");
  }
  if (a->device() != c->device()) {
    throw std::runtime_error("scalar_max: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::scalar_max<T>, a->data_as<T>(), scalar,
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_scalar_max<T>, a->data_as<T>(), scalar,
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> clamp(const Tensor &a, T min_val, T max_val, Tensor &c, size_t size,
                            const std::string &flow_id = "default") {
  if (!a->device() || !c->device()) {
    throw std::runtime_error("clamp: Device pointer has no associated device");
  }
  if (a->device() != c->device()) {
    throw std::runtime_error("clamp: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::clamp<T>, a->data_as<T>(), min_val, max_val,
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_clamp<T>, a->data_as<T>(), min_val, max_val,
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> equal(const Tensor &a, const Tensor &b, Tensor &c, size_t size,
                            const std::string &flow_id = "default") {
  if (!a->device() || !b->device() || !c->device()) {
    throw std::runtime_error("equal: Device pointer has no associated device");
  }
  if (a->device() != b->device() || a->device() != c->device()) {
    throw std::runtime_error("equal: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::equal<T>, a->data_as<T>(), b->data_as<T>(),
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_equal<T>, a->data_as<T>(), b->data_as<T>(),
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> greater(const Tensor &a, const Tensor &b, Tensor &c, size_t size,
                              const std::string &flow_id = "default") {
  if (!a->device() || !b->device() || !c->device()) {
    throw std::runtime_error("greater: Device pointer has no associated device");
  }
  if (a->device() != b->device() || a->device() != c->device()) {
    throw std::runtime_error("greater: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::greater<T>, a->data_as<T>(), b->data_as<T>(),
                           c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_greater<T>, a->data_as<T>(), b->data_as<T>(),
                            c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> copy(const Tensor &a, Tensor &c, size_t size, size_t a_offset = 0,
                           size_t c_offset = 0, const std::string &flow_id = "default") {
  if (!a->device() || !c->device()) {
    throw std::runtime_error("copy: Device pointer has no associated device");
  }
  if (a->device() != c->device()) {
    throw std::runtime_error("copy: All device pointers must be on the same device");
  }

  if (a->data_as<T>() == nullptr || c->data_as<T>() == nullptr) {
    throw std::runtime_error("copy: Null pointer exception in copy operation");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::copy<T>, a->data_as<T>() + a_offset,
                           c->data_as<T>() + c_offset, size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_copy<T>, a->data_as<T>() + a_offset,
                            c->data_as<T>() + c_offset, size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Special copy for copying cross devices (resort to same device/host copy if applicable)
template <typename T>
std::unique_ptr<Task> cd_copy(const Tensor &a, Tensor &c, size_t size, size_t a_offset = 0,
                              size_t c_offset = 0, const std::string &flow_id = "default") {
  if (!a->device() || !c->device()) {
    throw std::runtime_error("cd_copy: Device pointer has no associated device");
  }
  auto a_device = a->device();
  auto c_device = c->device();
  if (a_device == c_device) {
    // same device copy
    return copy<T>(a, c, size, a_offset, c_offset, flow_id);
  }
  auto a_device_type = a_device->device_type();
  auto c_device_type = c_device->device_type();

  if (a_device_type == DeviceType::CPU && c_device_type == DeviceType::GPU) {
    // host to device copy
#ifdef USE_CUDA
    return create_cuda_task(flow_id, ops::cuda::cuda_h2d_copy<T>, a->data_as<T>() + a_offset,
                            c->data_as<T>() + c_offset, size);
#else
    throw std::runtime_error("cd_copy: CUDA not enabled for CPU to GPU copy");
#endif
  } else if (a_device_type == DeviceType::GPU && c_device_type == DeviceType::CPU) {
    // device to host copy
#ifdef USE_CUDA
    return create_cuda_task(flow_id, ops::cuda::cuda_d2h_copy<T>, a->data_as<T>() + a_offset,
                            c->data_as<T>() + c_offset, size);
#else
    throw std::runtime_error("cd_copy: CUDA not enabled for GPU to CPU copy");
#endif
  } else {
    throw std::runtime_error("cd_copy: Unsupported device type combination");
  }
}

template <typename T>
std::unique_ptr<Task> zero(Tensor &c, size_t size, const std::string &flow_id = "default") {
  if (!c->device()) {
    throw std::runtime_error("zero: Device pointer has no associated device");
  }

  auto device = c->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::zero<T>, c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_zero<T>, c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> T sum(const Tensor &a, size_t size) {
  if (!a->device()) {
    throw std::runtime_error("sum: Device pointer has no associated device");
  }

  auto device_type = a->device()->device_type();

  if (device_type == DeviceType::CPU) {
    return ops::cpu::sum(a->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return ops::cuda::cuda_sum(a->data_as<T>(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> T dot_product(const Tensor &a, const Tensor &b, size_t size) {
  if (!a->device() || !b->device()) {
    throw std::runtime_error("dot_product: Device pointer has no associated device");
  }
  if (a->device() != b->device()) {
    throw std::runtime_error("dot_product: All device pointers must be on the same device");
  }

  auto device_type = a->device()->device_type();

  if (device_type == DeviceType::CPU) {
    return ops::cpu::dot_product(a->data_as<T>(), b->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return ops::cuda::cuda_dot_product(a->data_as<T>(), b->data_as<T>(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> T norm_squared(const Tensor &a, size_t size) {
  if (!a->device()) {
    throw std::runtime_error("norm_squared: Device pointer has no associated device");
  }

  auto device_type = a->device()->device_type();

  if (device_type == DeviceType::CPU) {
    return ops::cpu::norm_squared(a->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return ops::cuda::cuda_norm_squared(a->data_as<T>(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T> T sum_squared_diff(const Tensor &a, T mean, size_t size) {
  if (!a->device()) {
    throw std::runtime_error("sum_squared_diff: Device pointer has no associated device");
  }

  auto device_type = a->device()->device_type();

  if (device_type == DeviceType::CPU) {
    return ops::cpu::sum_squared_diff(a->data_as<T>(), mean, size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return ops::cuda::cuda_sum_squared_diff(a->data_as<T>(), mean, size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> sub_mul_scalar(const Tensor &a, T sub_scalar, T mul_scalar, Tensor &c,
                                     size_t size, const std::string &flow_id = "default") {
  if (!a->device() || !c->device()) {
    throw std::runtime_error("sub_mul_scalar: Device pointer has no associated device");
  }
  if (a->device() != c->device()) {
    throw std::runtime_error("sub_mul_scalar: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::sub_mul_scalar<T>, a->data_as<T>(), sub_scalar,
                           mul_scalar, c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_sub_mul_scalar<T>, a->data_as<T>(), sub_scalar,
                            mul_scalar, c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> mul_add_scalar(const Tensor &a, T mul_scalar, T add_scalar, Tensor &c,
                                     size_t size, const std::string &flow_id = "default") {
  if (!a->device() || !c->device()) {
    throw std::runtime_error("mul_add_scalar: Device pointer has no associated device");
  }
  if (a->device() != c->device()) {
    throw std::runtime_error("mul_add_scalar: All device pointers must be on the same device");
  }

  auto device = a->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::mul_add_scalar<T>, a->data_as<T>(), mul_scalar,
                           add_scalar, c->data_as<T>(), size);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_mul_add_scalar<T>, a->data_as<T>(), mul_scalar,
                            add_scalar, c->data_as<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fill_random_uniform(Tensor &data, size_t size, T min_val, T max_val,
                                          unsigned long long seed,
                                          const std::string &flow_id = "default") {
  if (!data->device()) {
    throw std::runtime_error("fill_random_uniform: Device pointer has no associated device");
  }

  auto device = data->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::fill_random_uniform<T>, data->data_as<T>(), size,
                           min_val, max_val, seed);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_fill_random_uniform<T>, data->data_as<T>(),
                            size, min_val, max_val, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
std::unique_ptr<Task> fill_random_normal(Tensor &data, size_t size, T mean, T stddev,
                                         unsigned long long seed,
                                         const std::string &flow_id = "default") {
  if (!data->device()) {
    throw std::runtime_error("fill_random_normal: Device pointer has no associated device");
  }

  auto device = data->device();
  auto device_type = device->device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(flow_id, ops::cpu::fill_random_normal<T>, data->data_as<T>(), size, mean,
                           stddev, seed);
  }
#ifdef USE_CUDA
  else if (device_type == DeviceType::GPU) {
    return create_cuda_task(flow_id, ops::cuda::cuda_fill_random_normal<T>, data->data_as<T>(),
                            size, mean, stddev, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}
}; // namespace TensorOps
} // namespace tnn