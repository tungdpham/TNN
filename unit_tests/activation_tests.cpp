#include "../matrix/matrix.hpp"
#include "layers/activations.hpp"
using namespace std;

ActivationFactory<double> factory;

void relu() {
  cout << "===================== ReLU Activation Test ===============" << endl;
  Matrix mat(3, 3, 3);
  mat.fill_random_normal(1.0);
  cout << "Matrix after filling with random normal 1.0:" << endl;
  mat.print();
  Matrix bias(3, 3, 3);
  bias.fill(-1.2);

  auto relu = factory.create("relu");
  if (relu) {
    relu->apply(mat, &bias);
    cout << "Matrix after ReLU activation:" << endl;
    mat.print();
  } else {
    cout << "ReLU activation function not found." << endl;
  }
  cout << '\n' << endl;
}

void sigmoid() {
  cout << "================= Sigmoid Activation Test ===============" << endl;
  Matrix mat(3, 3, 3);
  mat.fill_random_normal(1.0);
  cout << "Matrix after filling with random normal 1.0:" << endl;
  mat.print();
  Matrix bias(3, 3, 3);
  bias.fill(-1.2);

  auto sigmoid = factory.create("sigmoid");
  if (sigmoid) {
    sigmoid->apply(mat, &bias);
    cout << "Matrix after Sigmoid activation:" << endl;
    mat.print();
  } else {
    cout << "Sigmoid activation function not found." << endl;
  }
  cout << '\n' << endl;
}

void softmax() {
  cout << "================= Softmax Activation Test ===============" << endl;
  Matrix mat(3, 3, 3);
  mat.fill_random_normal(1.0);
  cout << "Matrix after filling with random normal 1.0:" << endl;
  mat.print();
  Matrix bias(3, 3, 3);
  bias.fill(-1.2);

  auto softmax = factory.create("softmax");
  if (softmax) {
    softmax->apply(mat, &bias);
    cout << "Matrix after Softmax activation:" << endl;
    mat.print();
  } else {
    cout << "Softmax activation function not found." << endl;
  }
  cout << '\n' << endl;
}

void softmax_channel() {
  Matrix mat(3, 3, 3);
  mat.fill_random_normal(1.0);
  cout << "Matrix after filling with random normal 1.0:" << endl;
  mat.print();
  Matrix bias(3, 3, 3);

  auto softmax = factory.create("softmax");
  if (softmax) {
    softmax->apply_channel(mat, 0, 0.1);
    cout << "Matrix after Softmax activation on channel 0:" << endl;
    mat.print();
  } else {
    cout << "Softmax activation function not found." << endl;
  }
  cout << '\n' << endl;
}

int main() {
  srand(time(NULL));

  factory = ActivationFactory<double>();
  factory.register_defaults();

  relu();
  sigmoid();
  softmax();
  softmax_channel();
  return 0;
}