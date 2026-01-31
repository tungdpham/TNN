/*
 * Example of using type-erased tensors for mixed precision operations
 */

#include <iostream>

#include "tensor/tensor.hpp"

using namespace tnn;

int main() {
  // Create type-erased tensors
  Tensor tensor_f32 = Tensor::create<float>({2, 3}, getCPU());
  Tensor tensor_f64 = make_tensor<double>({2, 3}, getCPU());

  // Fill with values
  tensor_f32->fill_scalar(1.0);
  tensor_f64->fill_scalar(2.0);

  // Use operator overloads
  Tensor result1 = tensor_f32 + tensor_f32;  // Works!
  Tensor result2 = tensor_f32 * 2.5;         // Scalar multiplication
  Tensor result3 = 3.0 * tensor_f32;         // Scalar on left

  // Print results
  std::cout << "Result 1 (f32 + f32): ";
  result1->head(6);

  std::cout << "Result 2 (f32 * 2.5): ";
  result2->head(6);

  std::cout << "Result 3 (3.0 * f32): ";
  result3->head(6);

  // ===== DATA ACCESS OPTIONS =====

  // Option 1: Type-unsafe void* access (use with caution)
  void *raw_ptr = tensor_f32->data();
  float *float_ptr = static_cast<float *>(raw_ptr);
  std::cout << "\nOption 1 - void* cast: First element = " << float_ptr[0] << std::endl;

  // Option 2: Use tensor_cast for type-safe access (RECOMMENDED)
  auto typed_f32 = Tensor::cast<float>(tensor_f32);
  float *typed_ptr = typed_f32->data();  // Now you have float*
  float value = (*typed_f32)(0, 0);      // Direct element access
  std::cout << "Option 2 - tensor_cast: First element = " << value << std::endl;

  // Option 3: Access dptr through tensor_cast
  dptr &dev_ptr = typed_f32->data_ptr();
  float *from_dev_ptr = dev_ptr.get();
  std::cout << "Option 3 - dptr: First element = " << from_dev_ptr[0] << std::endl;

  // Modify data through typed tensor
  (*typed_f32)(0, 0) = 42.0f;
  (*typed_f32)(0, 1) = 99.0f;
  std::cout << "After modification: ";
  tensor_f32->head(6);

  // Option 4: Use typed pointer directly with data_ptr
  float *data_array = typed_f32->data_ptr().get();
  for (size_t i = 0; i < 6; ++i) {
    data_array[i] = static_cast<float>(i * 10);
  }
  std::cout << "After array modification: ";
  tensor_f32->head(6);

  // Clone and device transfer
  Tensor cloned = tensor_f32->clone();
  std::cout << "\nCloned tensor: ";
  cloned->head(6);

  // Statistics work through interface
  std::cout << "\nStatistics:";
  std::cout << "\n  Mean: " << tensor_f32->mean_value();
  std::cout << "\n  Min: " << tensor_f32->min_value();
  std::cout << "\n  Max: " << tensor_f32->max_value();
  std::cout << std::endl;

  return 0;
}
