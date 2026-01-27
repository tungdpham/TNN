#pragma once

#ifdef USE_CUDNN

#include "type/type.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_graph.h>

namespace tnn {
namespace cuda {
namespace cudnn {
cudnnDataType_t to_cudnn_datatype(DType_t dtype);
}
} // namespace cuda
} // namespace tnn
#endif