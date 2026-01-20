#pragma once

#ifdef USE_CUDNN

#define CUDNN_FRONTEND_SKIP_JSON_LIB
#undef NLOHMAN_JSON_SERIALIZE_ENUM
#undef NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE
#undef NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE
#undef NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT

#include <cstring>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#endif