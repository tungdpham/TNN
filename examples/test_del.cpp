#include <cassert>

#include "device/del_allocator_v2.hpp"
#include "device/device_manager.hpp"

using namespace tnn;

int main() {
  auto allocator = DELAllocatorV2::create(getGPU(), defaultFlowHandle);
  allocator->reserve(100);

  auto ptr1 = allocator->allocate(10);
  auto ptr2 = allocator->allocate(20);
  allocator->flip();
  auto ptr3 = allocator->allocate(15);
  auto ptr4 = allocator->allocate(25);

  // ptr1 [0-10]
  // ptr2 [10-30]
  // ptr3 [85-100]
  // ptr4 [60-85]

  // unallocated [30-60]

  // Now we force them out of scope and ensure they are reclaimed
  ptr2 = nullptr;

  ptr3 = nullptr;

  ptr4 = nullptr;

  auto ptrX = allocator->allocate(30);

  return 0;
}
