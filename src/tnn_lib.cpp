// entry point for the main TNN shared library (libtnn.so/tnn.dll)

namespace tnn {

// lib ver
const char *get_version() { return "0.1.0"; }

const char *get_build_info() { return "TNN - Tensor Neural Network Library"; }

}  // namespace tnn
