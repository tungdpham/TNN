#include "nn/blocks_impl/cuda/softmax.hpp"

#ifdef USE_CUDNN

#include "type/type.hpp"

#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace tnn {
namespace cuda {

namespace fe = cudnn_frontend;

static void ensure_ok(fe::error_t status, const char *stage) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string("cuDNN frontend error at ") + stage + ": " +
                             status.get_message());
  }
}

static fe::DataType_t to_fe_data_type(cudnnDataType_t data_type) {
  switch (data_type) {
  case CUDNN_DATA_HALF:
    return fe::DataType_t::HALF;
  case CUDNN_DATA_FLOAT:
    return fe::DataType_t::FLOAT;
  case CUDNN_DATA_DOUBLE:
    return fe::DataType_t::DOUBLE;
  case CUDNN_DATA_BFLOAT16:
    return fe::DataType_t::BFLOAT16;
  default:
    throw std::runtime_error("Unsupported cuDNN data type for softmax");
  }
}

static fe::DataType_t to_fe_compute_type(cudnnDataType_t data_type) {
  if (data_type == CUDNN_DATA_HALF || data_type == CUDNN_DATA_BFLOAT16) {
    return fe::DataType_t::FLOAT;
  }
  return to_fe_data_type(data_type);
}

template <typename T> cudnnDataType_t get_cudnn_data_type();

template <> inline cudnnDataType_t get_cudnn_data_type<float>() { return CUDNN_DATA_FLOAT; }

template <> inline cudnnDataType_t get_cudnn_data_type<double>() { return CUDNN_DATA_DOUBLE; }

template <> inline cudnnDataType_t get_cudnn_data_type<fp16>() { return CUDNN_DATA_HALF; }

struct SoftmaxKey {
  size_t rows = 0;
  size_t cols = 0;
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;

  bool operator==(const SoftmaxKey &other) const {
    return rows == other.rows && cols == other.cols && data_type == other.data_type;
  }
};

struct SoftmaxKeyHash {
  size_t operator()(const SoftmaxKey &key) const {
    size_t h1 = std::hash<size_t>()(key.rows);
    size_t h2 = std::hash<size_t>()(key.cols);
    size_t h3 = std::hash<int>()(static_cast<int>(key.data_type));
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

struct SoftmaxHandle {
  cudnnHandle_t cudnn_handle = nullptr;
  size_t rows = 0;
  size_t cols = 0;
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;

  bool use_legacy = false;

  std::shared_ptr<fe::KernelCache> kernel_cache;

  std::shared_ptr<fe::graph::Graph> fwd_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_p;
  std::shared_ptr<fe::graph::Tensor_attributes> fwd_s;
  size_t fwd_workspace = 0;
  void *fwd_workspace_ptr = nullptr;
  size_t fwd_workspace_capacity = 0;

  std::shared_ptr<fe::graph::Graph> bwd_graph;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_s;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_dy;
  std::shared_ptr<fe::graph::Tensor_attributes> bwd_dx;
  size_t bwd_workspace = 0;
  void *bwd_workspace_ptr = nullptr;
  size_t bwd_workspace_capacity = 0;
};

static void ensure_workspace(void **ptr, size_t *capacity, size_t required_bytes) {
  if (required_bytes == 0) {
    return;
  }
  if (*capacity >= required_bytes && *ptr) {
    return;
  }
  if (*ptr) {
    cudaFree(*ptr);
    *ptr = nullptr;
    *capacity = 0;
  }
  cudaMalloc(ptr, required_bytes);
  *capacity = required_bytes;
}

static void build_forward_graph(SoftmaxHandle *handle) {
  const int64_t rows = static_cast<int64_t>(handle->rows);
  const int64_t cols = static_cast<int64_t>(handle->cols);

  auto io_type = to_fe_data_type(handle->data_type);
  auto compute_type = to_fe_compute_type(handle->data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type);

  std::vector<int64_t> dim = {rows, 1, 1, cols};
  std::vector<int64_t> stride = {cols, cols, cols, 1};

  auto P =
      graph->tensor(fe::graph::Tensor_attributes().set_name("P").set_dim(dim).set_stride(stride));

  // Manually implement softmax using basic operators: exp(x - max(x)) / sum(exp(x - max(x)))
  // Step 1: Find max along last dimension
  std::vector<int64_t> red_dim = {rows, 1, 1, 1};
  std::vector<int64_t> red_stride = {1, 1, 1, 1};

  auto max_attributes = fe::graph::Reduction_attributes()
                            .set_name("max")
                            .set_mode(fe::ReductionMode_t::MAX)
                            .set_compute_data_type(compute_type);
  auto max_val = graph->reduction(P, max_attributes);
  max_val->set_dim(red_dim).set_stride(red_stride);

  // Step 2: Subtract max (for numerical stability)
  auto sub_attributes =
      fe::graph::Pointwise_attributes().set_name("sub_max").set_mode(fe::PointwiseMode_t::SUB);
  auto p_minus_max = graph->pointwise(P, max_val, sub_attributes);

  // Step 3: Exponentiate
  auto exp_attributes =
      fe::graph::Pointwise_attributes().set_name("exp").set_mode(fe::PointwiseMode_t::EXP);
  auto exp_p = graph->pointwise(p_minus_max, exp_attributes);

  // Step 4: Sum exponents
  auto sum_attributes = fe::graph::Reduction_attributes()
                            .set_name("sum_exp")
                            .set_mode(fe::ReductionMode_t::ADD)
                            .set_compute_data_type(compute_type);
  auto sum_exp = graph->reduction(exp_p, sum_attributes);
  sum_exp->set_dim(red_dim).set_stride(red_stride);

  // Step 5: Divide by sum
  auto div_attributes =
      fe::graph::Pointwise_attributes().set_name("div").set_mode(fe::PointwiseMode_t::DIV);
  auto S = graph->pointwise(exp_p, sum_exp, div_attributes);

  S->set_output(true).set_data_type(io_type).set_name("S");

  ensure_ok(graph->validate(), "softmax fwd validate");
  ensure_ok(graph->build_operation_graph(handle->cudnn_handle), "softmax fwd build op graph");
  ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}),
            "softmax fwd create plans");
  ensure_ok(graph->check_support(), "softmax fwd check support");
  ensure_ok(graph->build_plans(), "softmax fwd build plans");

  int64_t workspace_size = 0;
  ensure_ok(graph->get_workspace_size(workspace_size), "softmax fwd workspace");

  handle->fwd_graph = graph;
  handle->fwd_p = P;
  handle->fwd_s = S;
  handle->fwd_workspace = static_cast<size_t>(workspace_size);
}

static void build_backward_graph(SoftmaxHandle *handle) {
  const int64_t rows = static_cast<int64_t>(handle->rows);
  const int64_t cols = static_cast<int64_t>(handle->cols);

  auto io_type = to_fe_data_type(handle->data_type);
  auto compute_type = to_fe_compute_type(handle->data_type);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(io_type)
      .set_intermediate_data_type(compute_type)
      .set_compute_data_type(compute_type)
      .set_kernel_cache(handle->kernel_cache);

  std::vector<int64_t> dim = {rows, 1, 1, cols};
  std::vector<int64_t> stride = {cols, cols, cols, 1};
  std::vector<int64_t> red_dim = {rows, 1, 1, 1};
  std::vector<int64_t> red_stride = {1, 1, 1, 1};

  auto S =
      graph->tensor(fe::graph::Tensor_attributes().set_name("S").set_dim(dim).set_stride(stride));

  auto dY =
      graph->tensor(fe::graph::Tensor_attributes().set_name("dY").set_dim(dim).set_stride(stride));

  auto dX =
      graph->tensor(fe::graph::Tensor_attributes().set_name("dX").set_dim(dim).set_stride(stride));

  auto mul_attributes =
      fe::graph::Pointwise_attributes().set_name("mul").set_mode(fe::PointwiseMode_t::MUL);

  auto mul = graph->pointwise(S, dY, mul_attributes);
  mul->set_name("S_mul_dY");

  auto reduce_attributes = fe::graph::Reduction_attributes()
                               .set_name("sum")
                               .set_mode(fe::ReductionMode_t::ADD)
                               .set_compute_data_type(compute_type);

  auto sum = graph->reduction(mul, reduce_attributes);
  sum->set_dim(red_dim).set_stride(red_stride).set_name("sum");

  auto sub_attributes =
      fe::graph::Pointwise_attributes().set_name("sub").set_mode(fe::PointwiseMode_t::SUB);

  auto sub = graph->pointwise(dY, sum, sub_attributes);
  sub->set_name("dY_minus_sum");

  auto out_attributes =
      fe::graph::Pointwise_attributes().set_name("mul_out").set_mode(fe::PointwiseMode_t::MUL);

  auto dX_out = graph->pointwise(S, sub, out_attributes);
  dX_out->set_output(true).set_data_type(io_type).set_name("dX");
  dX = dX_out;

  ensure_ok(graph->validate(), "softmax bwd validate");
  ensure_ok(graph->build_operation_graph(handle->cudnn_handle), "softmax bwd build op graph");
  ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}),
            "softmax bwd create plans");
  ensure_ok(graph->check_support(), "softmax bwd check support");
  ensure_ok(graph->build_plans(), "softmax bwd build plans");

  int64_t workspace_size = 0;
  ensure_ok(graph->get_workspace_size(workspace_size), "softmax bwd workspace");

  handle->bwd_graph = graph;
  handle->bwd_s = S;
  handle->bwd_dy = dY;
  handle->bwd_dx = dX;
  handle->bwd_workspace = static_cast<size_t>(workspace_size);
}

static SoftmaxHandle *get_softmax_handle(cudnnHandle_t cudnn_handle, size_t rows, size_t cols,
                                         cudnnDataType_t data_type) {
  static std::unordered_map<SoftmaxKey, std::unique_ptr<SoftmaxHandle>, SoftmaxKeyHash> cache;
  static std::mutex cache_mutex;

  SoftmaxKey key{rows, cols, data_type};
  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = cache.find(key);
  if (it != cache.end()) {
    it->second->cudnn_handle = cudnn_handle;
    return it->second.get();
  }

  auto handle = std::make_unique<SoftmaxHandle>();
  handle->cudnn_handle = cudnn_handle;
  handle->rows = rows;
  handle->cols = cols;
  handle->data_type = data_type;
  handle->kernel_cache = std::make_shared<fe::KernelCache>();

  try {
    build_forward_graph(handle.get());
    build_backward_graph(handle.get());
  } catch (const std::exception &) {
    handle->use_legacy = true;
  }

  auto *ptr = handle.get();
  cache.emplace(key, std::move(handle));
  return ptr;
}

template <typename T>
void softmax_forward(cudnnHandle_t handle, const T *input, T *output, size_t rows, size_t cols,
                     cudaStream_t stream) {
  auto *softmax_handle = get_softmax_handle(handle, rows, cols, get_cudnn_data_type<T>());
  cudnnSetStream(handle, stream);

  if (softmax_handle->use_legacy) {
    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, get_cudnn_data_type<T>(),
                               static_cast<int>(rows), static_cast<int>(cols), 1, 1);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc,
                        input, &beta, desc, output);
    cudnnDestroyTensorDescriptor(desc);
    return;
  }

  ensure_workspace(&softmax_handle->fwd_workspace_ptr, &softmax_handle->fwd_workspace_capacity,
                   softmax_handle->fwd_workspace);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
      {softmax_handle->fwd_p, const_cast<T *>(input)}, {softmax_handle->fwd_s, output}};

  auto status =
      softmax_handle->fwd_graph->execute(handle, variant_pack, softmax_handle->fwd_workspace_ptr);
  ensure_ok(status, "softmax forward execute");
}

template <typename T>
void softmax_backward(cudnnHandle_t handle, const T *output, const T *grad_output, T *grad_input,
                      size_t rows, size_t cols, cudaStream_t stream) {
  auto *softmax_handle = get_softmax_handle(handle, rows, cols, get_cudnn_data_type<T>());
  cudnnSetStream(handle, stream);

  if (softmax_handle->use_legacy) {
    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, get_cudnn_data_type<T>(),
                               static_cast<int>(rows), static_cast<int>(cols), 1, 1);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc,
                         output, desc, grad_output, &beta, desc, grad_input);
    cudnnDestroyTensorDescriptor(desc);
    return;
  }

  ensure_workspace(&softmax_handle->bwd_workspace_ptr, &softmax_handle->bwd_workspace_capacity,
                   softmax_handle->bwd_workspace);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
      {softmax_handle->bwd_s, const_cast<T *>(output)},
      {softmax_handle->bwd_dy, const_cast<T *>(grad_output)},
      {softmax_handle->bwd_dx, grad_input}};

  auto status =
      softmax_handle->bwd_graph->execute(handle, variant_pack, softmax_handle->bwd_workspace_ptr);
  ensure_ok(status, "softmax backward execute");
}

#define INSTANTIATE_SOFTMAX(T)                                                                     \
  template void softmax_forward<T>(cudnnHandle_t handle, const T *input, T *output, size_t rows,   \
                                   size_t cols, cudaStream_t stream);                              \
  template void softmax_backward<T>(cudnnHandle_t handle, const T *output, const T *grad_output,   \
                                    T *grad_input, size_t rows, size_t cols, cudaStream_t stream);
INSTANTIATE_SOFTMAX(fp16);
INSTANTIATE_SOFTMAX(float);
INSTANTIATE_SOFTMAX(double);
#undef INSTANTIATE_SOFTMAX

} // namespace cuda
} // namespace tnn

#endif