#include "cuda/cudnn/common.hpp"

namespace tnn {
namespace cuda {
namespace cudnn {
cudnnDataType_t to_cudnn_datatype(DType_t dtype) {
  switch (dtype) {
    case DType_t::FP16:
      return CUDNN_DATA_HALF;
    case DType_t::BF16:
      return CUDNN_DATA_BFLOAT16;
    case DType_t::FP32:
      return CUDNN_DATA_FLOAT;
    case DType_t::FP64:
      return CUDNN_DATA_DOUBLE;
    default:
      throw std::runtime_error("Unsupported data type for cudnn data type conversion");
  }
}

}  // namespace cudnn
}  // namespace cuda
}  // namespace tnn