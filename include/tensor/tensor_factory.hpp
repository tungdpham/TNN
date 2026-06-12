#pragma once

#include <istream>

#include "device/iallocator.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace synet {

inline void load_into(std::istream &in, Tensor &tensor) {
  if (!in) {
    throw std::runtime_error("Stream is not ready for reading");
  }
  DType_t dtype;
  in.read(reinterpret_cast<char *>(&dtype), sizeof(DType_t));
  if (dtype != tensor.data_type()) {
    throw std::runtime_error("Tensor dtype does not match data in file");
  }
  size_t dims;
  in.read(reinterpret_cast<char *>(&dims), sizeof(size_t));
  Vec<size_t> shape(dims);
  in.read(reinterpret_cast<char *>(shape.data()), dims * sizeof(size_t));
  if (in.gcount() != static_cast<std::streamsize>(dims * sizeof(size_t))) {
    throw std::runtime_error("Failed to read tensor shape from file");
  }
  if (shape != tensor.shape()) {
    throw std::runtime_error("Tensor shape does not match data in file");
  }
  size_t byte_size = tensor.size() * get_dtype_size(dtype);
  if (tensor.device().device_type() == DeviceType::CPU) {
    in.read(reinterpret_cast<char *>(tensor.data()), byte_size);
    if (in.gcount() != static_cast<std::streamsize>(byte_size)) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
  } else {
    Vec<char> host_buffer(byte_size);
    in.read(reinterpret_cast<char *>(host_buffer.data()), byte_size);
    if (in.gcount() != static_cast<std::streamsize>(byte_size)) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
    tensor.device().copyToDevice(tensor.data(), host_buffer.data(), byte_size);
  }
}

inline Tensor load(std::istream &in, IAllocator &allocator) {
  DType_t dtype;
  in.read(reinterpret_cast<char *>(&dtype), sizeof(DType_t));
  size_t dims;
  in.read(reinterpret_cast<char *>(&dims), sizeof(size_t));
  Vec<size_t> shape(dims);
  in.read(reinterpret_cast<char *>(shape.data()), dims * sizeof(size_t));
  if (in.gcount() != static_cast<std::streamsize>(dims * sizeof(size_t))) {
    throw std::runtime_error("Failed to read tensor shape from file");
  }
  auto tensor = Tensor(shape, dtype, allocator);
  size_t byte_size = tensor.size() * get_dtype_size(dtype);
  if (allocator.device().device_type() == DeviceType::CPU) {
    in.read(reinterpret_cast<char *>(tensor.data()), byte_size);
    if (in.gcount() != static_cast<std::streamsize>(byte_size)) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
  } else {
    Vec<char> host_buffer(byte_size);
    in.read(reinterpret_cast<char *>(host_buffer.data()), byte_size);
    if (in.gcount() != static_cast<std::streamsize>(byte_size)) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
    allocator.device().copyToDevice(tensor.data(), host_buffer.data(), byte_size);
  }
  return tensor;
}

}  // namespace synet