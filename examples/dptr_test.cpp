#include "device/dptr.hpp"

#include <iostream>

#include "device/device_manager.hpp"
using namespace tnn;

signed main() {
  dptr view = make_dptr(getCPU(), 1024);
  float *ptr = view.get<float>();
  ptr[0] = 1.0f;

  std::cout << "Dptr at " << ptr << " has value: " << ptr[0] << std::endl;
  return 0;
}