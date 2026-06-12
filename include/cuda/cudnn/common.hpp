#pragma once

#ifdef USE_CUDNN

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_graph.h>

#include <cstring>

#include "type/type.hpp"

namespace synet {
namespace cuda {
namespace cudnn {
cudnnDataType_t to_cudnn_datatype(DType_t dtype);
}
}  // namespace cuda
}  // namespace synet
#endif