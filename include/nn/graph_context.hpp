#pragma once

#include <vector>

#include "device/device.hpp"
#include "device/iallocator.hpp"
#include "nn/layer.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tnn {

inline size_t get_bytes_size(const std::vector<size_t> &shape, DType_t dtype) {
  return std::accumulate(shape.begin(), shape.end(), get_dtype_size(dtype),
                         std::multiplies<size_t>());
}

struct GraphContextDescriptor {
  std::vector<ParamDescriptor> param_descs;
  size_t param_bytes = 0;
  size_t grad_bytes = 0;

  void register_desc(const ParamDescriptor &param_desc) {
    param_descs.push_back(param_desc);

    size_t bytes_size = get_bytes_size(param_desc.shape, param_desc.dtype);
    // round up to 256 bytes for better memory access pattern, can be tuned later
    bytes_size = (bytes_size + 255) & ~255;

    // assuming param and grad have the same shape and dtype for simplicity
    param_bytes += bytes_size;
    grad_bytes += bytes_size;
  }
};

class GraphContext {
public:
  GraphContext(IAllocator &allocator, GraphContextDescriptor ctx_desc)
      : ctx_desc_(std::move(ctx_desc)),
        allocator_(allocator) {
    allocator.reserve(ctx_desc.param_bytes + ctx_desc.grad_bytes);
    param_slab_ = allocator.allocate(ctx_desc_.param_bytes);
    grad_slab_ = allocator.allocate(ctx_desc_.grad_bytes);

    size_t param_offset = 0;
    size_t grad_offset = 0;
    for (auto &param_desc : ctx_desc_.param_descs) {
      size_t bytes_size = get_bytes_size(param_desc.shape, param_desc.dtype);
      // round up to 256 bytes for better memory access pattern, can be tuned later
      bytes_size = (bytes_size + 255) & ~255;
      dptr param_buffer = param_slab_.span(param_offset, bytes_size);
      dptr grad_buffer = grad_slab_.span(grad_offset, bytes_size);

      Tensor param =
          make_tensor(allocator_, param_desc.dtype, param_desc.shape, std::move(param_buffer));
      Tensor grad =
          make_tensor(allocator_, param_desc.dtype, param_desc.shape, std::move(grad_buffer));

      *param_desc.data_ptr = param;
      *param_desc.grad_ptr = grad;

      params_.push_back(param);
      grads_.push_back(grad);

      param_offset += bytes_size;
      grad_offset += bytes_size;
    }
  }

  GraphContextDescriptor descriptor() const { return ctx_desc_; }

  std::vector<Tensor> &parameters() { return params_; }
  std::vector<Tensor> &gradients() { return grads_; }

  const std::vector<Tensor> &parameters() const { return params_; }
  const std::vector<Tensor> &gradients() const { return grads_; }

  void zero_grads() { ops::set_scalar<uchar>(grad_slab_, 0, grad_slab_.capacity()); }

  IAllocator &allocator() { return allocator_; }

  const Device &device() const { return allocator_.device(); }

private:
  GraphContextDescriptor ctx_desc_;
  IAllocator &allocator_;
  std::vector<Tensor> params_, grads_;
  dptr param_slab_, grad_slab_;
};
}  // namespace tnn