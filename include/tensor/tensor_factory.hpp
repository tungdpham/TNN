#include "device/iallocator.hpp"
#include "tensor/tensor.hpp"
#include "tensor/typed_tensor.hpp"

namespace tnn {

template <typename T>
DType_t get_dtype_if_tensor(const T &val) {
  if constexpr (std::is_convertible_v<T, tnn::Tensor>) {
    return val ? val->data_type() : DType_t::UNKNOWN;
  }
  return DType_t::UNKNOWN;
}

template <typename... Args>
void check_all_match(DType_t expected, const Args &...args) {
  auto validator = [&](DType_t current) {
    if (current != DType_t::UNKNOWN && current != expected) {
      throw std::runtime_error("Tensor DType mismatch in operation!");
    }
  };
  (validator(get_dtype_if_tensor(args)), ...);
}

template <typename... Args>
DType_t find_and_verify_dtype(const Args &...args) {
  DType_t found = DType_t::UNKNOWN;

  auto find_first = [&](DType_t current) {
    if (found == DType_t::UNKNOWN && current != DType_t::UNKNOWN) {
      found = current;
    }
  };
  (find_first(get_dtype_if_tensor(args)), ...);

  if (found == DType_t::UNKNOWN) {
    throw std::runtime_error("No Tensor found in arguments.");
  }

  check_all_match(found, args...);
  return found;
}

#define DISPATCH_ON_DTYPE(dtype_value, type_alias, ...)        \
  do {                                                         \
    switch (dtype_value) {                                     \
      case DType_t::FP16: {                                    \
        using type_alias = fp16;                               \
        __VA_ARGS__;                                           \
        break;                                                 \
      }                                                        \
      case DType_t::BF16: {                                    \
        using type_alias = bf16;                               \
        __VA_ARGS__;                                           \
        break;                                                 \
      }                                                        \
      case DType_t::FP32: {                                    \
        using type_alias = float;                              \
        __VA_ARGS__;                                           \
        break;                                                 \
      }                                                        \
      case DType_t::FP64: {                                    \
        using type_alias = double;                             \
        __VA_ARGS__;                                           \
        break;                                                 \
      }                                                        \
      default:                                                 \
        throw std::runtime_error("Unknown dtype in dispatch"); \
    }                                                          \
  } while (0)

#define DISPATCH_AUTO(type_alias, func_body, ...) \
  DISPATCH_ON_DTYPE(tnn::find_and_verify_dtype(__VA_ARGS__), type_alias, func_body)

#define DISPATCH_AUTO_T(func, ...) DISPATCH_AUTO(T, func<T>(__VA_ARGS__), __VA_ARGS__)

template <typename T>
inline Tensor make_tensor(std::vector<size_t> shape, const Device &device = getCPU()) {
  auto &allocator = DeviceAllocator::instance(device);
  return std::make_shared<TypedTensor<T>>(allocator, shape);
}

template <typename T>
inline Tensor make_tensor(std::vector<size_t> shape, const dptr &data,
                          const Device &device = getCPU()) {
  auto &allocator = DeviceAllocator::instance(device);
  return std::make_shared<TypedTensor<T>>(allocator, shape, data);
}

template <typename T>
inline Tensor make_tensor(std::initializer_list<size_t> shape = {},
                          const Device &device = getCPU()) {
  auto &allocator = DeviceAllocator::instance(device);
  return std::make_shared<TypedTensor<T>>(allocator, shape);
}

template <typename T>
inline Tensor make_tensor(std::initializer_list<size_t> shape, const dptr &data,
                          const Device &device = getCPU()) {
  auto &allocator = DeviceAllocator::instance(device);
  return std::make_shared<TypedTensor<T>>(allocator, shape, data);
}

inline Tensor make_tensor(DType_t dtype, std::vector<size_t> shape,
                          const Device &device = getCPU()) {
  switch (dtype) {
    case DType_t::FP16:
      return make_tensor<fp16>(shape, device);
    case DType_t::BF16:
      return make_tensor<bf16>(shape, device);
    case DType_t::FP32:
      return make_tensor<float>(shape, device);
    case DType_t::FP64:
      return make_tensor<double>(shape, device);
    default:
      throw std::runtime_error("Unsupported data type for make_tensor");
  }
}

inline Tensor make_tensor(DType_t dtype, std::initializer_list<size_t> shape = {},
                          const Device &device = getCPU()) {
  switch (dtype) {
    case DType_t::FP16:
      return make_tensor<fp16>(shape, device);
    case DType_t::BF16:
      return make_tensor<bf16>(shape, device);
    case DType_t::FP32:
      return make_tensor<float>(shape, device);
    case DType_t::FP64:
      return make_tensor<double>(shape, device);
    default:
      throw std::runtime_error("Unsupported data type for make_tensor");
  }
}

template <typename T>
inline Tensor make_tensor(IAllocator &allocator, dptr &&data, std::vector<size_t> shape) {
  return std::make_shared<TypedTensor<T>>(allocator, shape, std::move(data));
}

inline Tensor make_tensor(IAllocator &allocator, DType_t dtype, dptr &&data,
                          std::vector<size_t> shape) {
  switch (dtype) {
    case DType_t::FP16:
      return make_tensor<fp16>(allocator, std::move(data), shape);
    case DType_t::BF16:
      return make_tensor<bf16>(allocator, std::move(data), shape);
    case DType_t::FP32:
      return make_tensor<float>(allocator, std::move(data), shape);
    case DType_t::FP64:
      return make_tensor<double>(allocator, std::move(data), shape);
    default:
      throw std::runtime_error("Unsupported data type for make_tensor");
  }
}

template <typename T>
inline Tensor make_tensor(IAllocator &allocator, std::vector<size_t> shape) {
  return std::make_shared<TypedTensor<T>>(allocator, shape);
}

template <typename T>
inline Tensor make_tensor(IAllocator &allocator, std::initializer_list<size_t> shape = {}) {
  return std::make_shared<TypedTensor<T>>(allocator, std::vector<size_t>(shape));
}

inline Tensor make_tensor(IAllocator &allocator, DType_t dtype, std::vector<size_t> shape) {
  switch (dtype) {
    case DType_t::FP16:
      return make_tensor<fp16>(allocator, shape);
    case DType_t::BF16:
      return make_tensor<bf16>(allocator, shape);
    case DType_t::FP32:
      return make_tensor<float>(allocator, shape);
    case DType_t::FP64:
      return make_tensor<double>(allocator, shape);
    default:
      throw std::runtime_error("Unsupported data type for make_tensor");
  }
}

inline Tensor make_tensor(IAllocator &allocator, DType_t dtype,
                          std::initializer_list<size_t> shape = {}) {
  switch (dtype) {
    case DType_t::BF16:
      return make_tensor<bf16>(allocator, shape);
    case DType_t::FP16:
      return make_tensor<fp16>(allocator, shape);
    case DType_t::FP32:
      return make_tensor<float>(allocator, shape);
    case DType_t::FP64:
      return make_tensor<double>(allocator, shape);
    default:
      throw std::runtime_error("Unsupported data type for make_tensor");
  }
}

inline Tensor dtype_cast(const Tensor &input, DType_t target_dtype) {
  if (input->data_type() == target_dtype) {
    return input->clone();
  }

  const dptr &input_data = input->data_ptr();
  size_t input_size = input->size();
  dptr output_data = make_dptr(input->device(), input_size * get_dtype_size(target_dtype));
  DISPATCH_ON_DTYPE(input->data_type(), A_T,
                    DISPATCH_ON_DTYPE(target_dtype, B_T,
                                      ops::cast<A_T, B_T>(input_data, output_data, input_size)));
  return make_tensor(input->allocator(), target_dtype, std::move(output_data), input->shape());
}

template <typename T>
inline Tensor load(std::ifstream &in, const Device &device) {
  if (!in.is_open()) {
    throw std::runtime_error("File is not open for reading");
  }
  size_t dims;
  in.read(reinterpret_cast<char *>(&dims), sizeof(size_t));
  std::vector<size_t> shape(dims);
  in.read(reinterpret_cast<char *>(shape.data()), dims * sizeof(size_t));
  if (in.gcount() != static_cast<std::streamsize>(dims * sizeof(size_t))) {
    throw std::runtime_error("Failed to read tensor shape from file");
  }

  auto tensor = std::make_shared<TypedTensor<T>>(shape, device);
  if (device.device_type() == DeviceType::CPU) {
    in.read(reinterpret_cast<char *>(tensor->data()), tensor->size() * sizeof(T));
    if (in.gcount() != static_cast<std::streamsize>(tensor->size() * sizeof(T))) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
  } else {
    std::vector<T> host_buffer(tensor->size());
    in.read(reinterpret_cast<char *>(host_buffer.data()), tensor->size() * sizeof(T));
    if (in.gcount() != static_cast<std::streamsize>(tensor->size() * sizeof(T))) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
    device.copyToDevice(tensor->data(), host_buffer.data(), tensor->size() * sizeof(T));
  }
  return tensor;
}

inline void load_into(std::ifstream &in, Tensor &target) {
  if (!in.is_open()) {
    throw std::runtime_error("File is not open for reading");
  }
  // read dtype, dims, shape, and data
  DType_t dtype;
  in.read(reinterpret_cast<char *>(&dtype), sizeof(DType_t));
  size_t dims;
  in.read(reinterpret_cast<char *>(&dims), sizeof(size_t));
  std::vector<size_t> shape(dims);
  in.read(reinterpret_cast<char *>(shape.data()), dims * sizeof(size_t));
  if (in.gcount() != static_cast<std::streamsize>(dims * sizeof(size_t))) {
    throw std::runtime_error("Failed to read tensor shape from file");
  }

  target = make_tensor(target->allocator(), dtype, target->data_ptr(), shape);

  if (target->device_type() == DeviceType::CPU) {
    in.read(reinterpret_cast<char *>(target->data()), target->size() * get_dtype_size(dtype));
    if (in.gcount() != static_cast<std::streamsize>(target->size() * get_dtype_size(dtype))) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
  } else {
    void *host_buffer = malloc(target->size() * get_dtype_size(dtype));
    in.read(reinterpret_cast<char *>(host_buffer), target->size() * get_dtype_size(dtype));
    if (in.gcount() != static_cast<std::streamsize>(target->size() * get_dtype_size(dtype))) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
    target->device().copyToDevice(target->data(), host_buffer,
                                  target->size() * get_dtype_size(dtype));
    free(host_buffer);
  }
}

}  // namespace tnn