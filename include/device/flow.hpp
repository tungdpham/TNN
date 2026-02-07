#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <stdexcept>
#include <string>

namespace tnn {
class Flow {
public:
  virtual void synchronize() = 0;
};

// Does nothing for now, but task handler per device can put it into different threads in threadpool
// in the future
class CPUFlow : public Flow {
public:
  void synchronize() override {
    // No-op for CPU
  }
};

#ifdef USE_CUDA
// Manages CUDA streams for asynchronous execution, the id should be put into device's
// hashmap for lookup
class CUDAFlow : public Flow {
private:
  cudaStream_t stream_;

public:
  explicit CUDAFlow() {
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to create CUDA stream: " +
                               std::string(cudaGetErrorString(err)));
    }
  }

  ~CUDAFlow() { cudaStreamDestroy(stream_); }

  cudaStream_t get_stream() { return stream_; }

  void synchronize() override {
    cudaError_t err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to synchronize CUDA stream: " +
                               std::string(cudaGetErrorString(err)));
    }
  }
};

#endif
}  // namespace tnn