/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "device/device_ptr.hpp"

#ifdef USE_CUDA
#include "cuda/error_handler.hpp"
#endif

#include <iostream>
#include <vector>

using namespace tnn;
using namespace std;

#ifdef USE_CUDA
// CUDA kernel to add two arrays
__global__ void vectorAddKernel(const float *a, const float *b, float *result, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    result[idx] = a[idx] + b[idx];
  }
}

// Helper function to launch the kernel with device_ptr
void vectorAdd(const device_ptr<float[]> &a, const device_ptr<float[]> &b,
               device_ptr<float[]> &result, int n) {
  const int blockSize = 256;
  const int gridSize = (n + blockSize - 1) / blockSize;

  vectorAddKernel<<<gridSize, blockSize>>>(a.get(), b.get(), result.get(), n);

  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());

  // Wait for kernel to complete
  CUDA_CHECK(cudaDeviceSynchronize());
}
#endif

int main() {
  try {
    cout << "=== Device Pointer Array Addition Example ===" << endl;

    // Initialize devices
    initializeDefaultDevices();
    DeviceManager &manager = DeviceManager::getInstance();

    string gpu_device_index = "";
    for (const string &id : manager.getAvailableDeviceIDs()) {
      if (manager.getDevice(id).getDeviceType() == DeviceType::GPU) {
        gpu_device_index = id;
        break;
      }
    }

    if (gpu_device_index.empty()) {
      throw runtime_error("No GPU device found. This example requires a GPU device.");
    }

    cout << "Using device: " << manager.getDevice(gpu_device_index).getName() << endl;

    Device *device = &const_cast<Device &>(manager.getDevice(gpu_device_index));

    // Array size for demonstration
    const int n = 1000000;
    cout << "Creating arrays of size: " << n << endl;

    // Create device arrays using device_ptr
    auto a_ptr = make_array_ptr<float[]>(device, n);
    auto b_ptr = make_array_ptr<float[]>(device, n);
    auto result_ptr = make_array_ptr<float[]>(device, n);

    cout << "Arrays created successfully" << endl;

    // Create host arrays for initialization and verification
    vector<float> h_a(n), h_b(n), h_result(n);

    // Initialize host arrays
    for (int i = 0; i < n; i++) {
      h_a[i] = static_cast<float>(i);
      h_b[i] = static_cast<float>(i * 2);
    }

    cout << "Host arrays initialized" << endl;

#ifdef USE_CUDA
    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(a_ptr.get(), h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_ptr.get(), h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cout << "Data copied to device" << endl;

    // Perform vector addition using our kernel
    vectorAdd(a_ptr, b_ptr, result_ptr, n);

    cout << "Vector addition completed on GPU" << endl;

    // Copy result back to host
    CUDA_CHECK(
        cudaMemcpy(h_result.data(), result_ptr.get(), n * sizeof(float), cudaMemcpyDeviceToHost));

    cout << "Result copied back to host" << endl;

    // Verify results
    bool success = true;
    for (int i = 0; i < min(10, n); i++) {
      float expected = h_a[i] + h_b[i];
      if (abs(h_result[i] - expected) > 1e-5) {
        cout << "Verification failed at index " << i << ": expected " << expected << ", got "
             << h_result[i] << endl;
        success = false;
        break;
      }
    }

    if (success) {
      cout << "Vector addition verification successful!" << endl;
      cout << "Sample results:" << endl;
      for (int i = 0; i < 5; i++) {
        cout << "  " << h_a[i] << " + " << h_b[i] << " = " << h_result[i] << endl;
      }
    }
#else
    cout << "CUDA not available - skipping kernel execution" << endl;
#endif

  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  return 0;
}