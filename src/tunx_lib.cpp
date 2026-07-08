// entry point for the main tunx shared library (libtunx.so/tunx.dll)

namespace tunx {

// lib ver
const char *get_version() { return "0.1.0"; }

const char *get_build_info() { return "tunx - Graph-centric Deep Learning Library"; }

}  // namespace tunx
