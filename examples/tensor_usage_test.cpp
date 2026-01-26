#include "tensor/tensor.hpp"

using namespace tnn;

using namespace std;

void test_tensor_assignment(Tensor &a, Tensor &b) {
  Tensor temp;
  temp = a;
  a = b;
  b = temp;
}

void test_tensor_swap(Tensor &a, Tensor &b) { std::swap(a, b); }

signed main() {
  Tensor a = Tensor::create<float>({64, 32, 32, 3});
  Tensor b = Tensor::create<float>({128, 224, 224, 3});

  std::cout << "Tensor a shape: " << a->shape_str() << std::endl;
  std::cout << "Tensor b shape: " << b->shape_str() << std::endl;

  test_tensor_assignment(a, b);

  std::cout << "After assignment:" << std::endl;
  std::cout << "Tensor a shape: " << a->shape_str() << std::endl;
  std::cout << "Tensor b shape: " << b->shape_str() << std::endl;

  test_tensor_swap(a, b);
  std::cout << "After swap:" << std::endl;
  std::cout << "Tensor a shape: " << a->shape_str() << std::endl;
  std::cout << "Tensor b shape: " << b->shape_str() << std::endl;
  return 0;
}