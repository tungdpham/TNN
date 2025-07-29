#pragma once
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../matrix/matrix.hpp"

// Forward declaration of your Matrix<T> class

template <typename T> class ActivationFunction {
public:
  virtual ~ActivationFunction() = default;

  // Core methods that work with Matrix<T> objects
  virtual void apply(Matrix<T> &matrix,
                     const Matrix<T> *bias = nullptr) const = 0;
  virtual void apply(Matrix<T> &matrix, T scalar_bias) const = 0;
  virtual void apply_inplace(Matrix<T> &matrix,
                             const Matrix<T> &bias) const = 0;

  // For backpropagation - compute derivatives
  virtual Matrix<T>
  derivative(const Matrix<T> &activated_values,
             const Matrix<T> *upstream_grad = nullptr) const = 0;
  virtual void
  derivative_inplace(Matrix<T> &activated_values,
                     const Matrix<T> *upstream_grad = nullptr) const = 0;

  // Utility methods
  virtual std::string name() const = 0;
  virtual std::unique_ptr<ActivationFunction<T>> clone() const = 0;

  // Channel-wise operations (useful for CNNs)
  virtual void apply_channel(Matrix<T> &matrix, int channel,
                             T bias = T(0)) const = 0;
  virtual void apply_channel(Matrix<T> &matrix, int channel,
                             const std::vector<T> &channel_bias) const = 0;
};

template <typename T> class Linear : public ActivationFunction<T> {
public:
  void apply(Matrix<T> &matrix,
             const Matrix<T> *bias = nullptr) const override {
    if (bias) {
      int size = matrix.size();
      for (int i = 0; i < size; ++i) {
        matrix.data[i] += bias->data[i];
      }
    }
  }

  void apply(Matrix<T> &matrix, T scalar_bias) const override {
    if (scalar_bias != T(0)) {
      int size = matrix.size();
      for (int i = 0; i < size; ++i) {
        matrix.data[i] += scalar_bias;
      }
    }
  }

  void apply_inplace(Matrix<T> &matrix, const Matrix<T> &bias) const override {
    apply(matrix, &bias);
  }

  Matrix<T>
  derivative(const Matrix<T> &activated_values,
             const Matrix<T> *upstream_grad = nullptr) const override {
    Matrix<T> result(activated_values.rows, activated_values.cols,
                     activated_values.channels);
    result.fill(T(1)); // Derivative of linear function is always 1
    return result;
  }

  void
  derivative_inplace(Matrix<T> &activated_values,
                     const Matrix<T> *upstream_grad = nullptr) const override {
    activated_values.fill(T(1));
  }

  void apply_channel(Matrix<T> &matrix, int channel,
                     T bias = T(0)) const override {
    if (channel >= matrix.channels || channel < 0) {
      throw std::invalid_argument("Channel index out of bounds");
    }
    if (bias != T(0)) {
      int channel_size = matrix.rows * matrix.cols;
      int offset = channel * channel_size;

      for (int i = 0; i < channel_size; ++i) {
        matrix.data[offset + i] += bias;
      }
    }
  }

  void apply_channel(Matrix<T> &matrix, int channel,
                     const std::vector<T> &channel_bias) const override {
    if (channel >= matrix.channels || channel < 0) {
      throw std::invalid_argument("Channel index out of bounds");
    }
    int channel_size = matrix.rows * matrix.cols;
    int offset = channel * channel_size;

    for (int i = 0; i < channel_size; ++i) {
      matrix.data[offset + i] += channel_bias[i];
    }
  }

  std::string name() const override { return "linear"; }

  std::unique_ptr<ActivationFunction<T>> clone() const override {
    return std::make_unique<Linear<T>>();
  }
};

template <typename T> class ReLU : public ActivationFunction<T> {
private:
  T negative_slope;

public:
  explicit ReLU(T slope = T(0)) : negative_slope(slope) {}

  void apply(Matrix<T> &matrix,
             const Matrix<T> *bias = nullptr) const override {
    int size = matrix.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      T bias_val = bias ? bias->data[i] : T(0);
      T val = matrix.data[i] + bias_val;
      matrix.data[i] = val > T(0) ? val : negative_slope * val;
    }
  }

  void apply(Matrix<T> &matrix, T scalar_bias) const override {
    int size = matrix.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      T val = matrix.data[i] + scalar_bias;
      matrix.data[i] = val > T(0) ? val : negative_slope * val;
    }
  }

  void apply_inplace(Matrix<T> &matrix, const Matrix<T> &bias) const override {
    if (matrix.rows != bias.rows || matrix.cols != bias.cols ||
        matrix.channels != bias.channels) {
      throw std::invalid_argument("Matrix<T> and bias dimensions must match");
    }
    apply(matrix, &bias);
  }

  Matrix<T>
  derivative(const Matrix<T> &activated_values,
             const Matrix<T> *upstream_grad = nullptr) const override {
    Matrix<T> result(activated_values.rows, activated_values.cols,
                     activated_values.channels);
    int size = activated_values.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      result.data[i] = activated_values.data[i] > T(0) ? T(1) : negative_slope;
    }
    return result;
  }

  void
  derivative_inplace(Matrix<T> &activated_values,
                     const Matrix<T> *upstream_grad = nullptr) const override {
    int size = activated_values.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      activated_values.data[i] =
          activated_values.data[i] > T(0) ? T(1) : negative_slope;
    }
  }

  void apply_channel(Matrix<T> &matrix, int channel,
                     T bias = T(0)) const override {
    if (channel >= matrix.channels || channel < 0) {
      throw std::invalid_argument("Channel index out of bounds");
    }

    int channel_size = matrix.rows * matrix.cols;
    int offset = channel * channel_size;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < channel_size; ++i) {
      T val = matrix.data[offset + i] + bias;
      matrix.data[offset + i] = val > T(0) ? val : negative_slope * val;
    }
  }

  void apply_channel(Matrix<T> &matrix, int channel,
                     const std::vector<T> &channel_bias) const override {
    if (channel >= matrix.channels || channel < 0) {
      throw std::invalid_argument("Channel index out of bounds");
    }

    int channel_size = matrix.rows * matrix.cols;
    if ((int)channel_bias.size() != channel_size) {
      throw std::invalid_argument("Channel bias size must match channel size");
    }
    int offset = channel * channel_size;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < channel_size; ++i) {
      T val = matrix.data[offset + i] + channel_bias[i];
      matrix.data[offset + i] = val > T(0) ? val : negative_slope * val;
    }
  }

  std::string name() const override {
    return negative_slope == T(0) ? "relu" : "leaky_relu";
  }

  std::unique_ptr<ActivationFunction<T>> clone() const override {
    return std::make_unique<ReLU<T>>(negative_slope);
  }
};

template <typename T> class Sigmoid : public ActivationFunction<T> {
public:
  void apply(Matrix<T> &matrix,
             const Matrix<T> *bias = nullptr) const override {
    int size = matrix.size();
    for (int i = 0; i < size; ++i) {
      T bias_val = bias ? bias->data[i] : T(0);
      T val = matrix.data[i] + bias_val;
      matrix.data[i] = T(1) / (T(1) + std::exp(-val));
    }
  }

  void apply(Matrix<T> &matrix, T scalar_bias) const override {
    int size = matrix.size();
    for (int i = 0; i < size; ++i) {
      T val = matrix.data[i] + scalar_bias;
      matrix.data[i] = T(1) / (T(1) + std::exp(-val));
    }
  }

  void apply_inplace(Matrix<T> &matrix, const Matrix<T> &bias) const override {
    apply(matrix, &bias);
  }

  Matrix<T>
  derivative(const Matrix<T> &activated_values,
             const Matrix<T> *upstream_grad = nullptr) const override {
    Matrix<T> result(activated_values.rows, activated_values.cols,
                     activated_values.channels);
    int size = activated_values.size();
    for (int i = 0; i < size; ++i) {
      T val = activated_values.data[i];
      result.data[i] = val * (T(1) - val);
    }
    return result;
  }

  void
  derivative_inplace(Matrix<T> &activated_values,
                     const Matrix<T> *upstream_grad = nullptr) const override {
    int size = activated_values.size();
    for (int i = 0; i < size; ++i) {
      T val = activated_values.data[i];
      activated_values.data[i] = val * (T(1) - val);
    }
  }

  void apply_channel(Matrix<T> &matrix, int channel,
                     T bias = T(0)) const override {
    if (channel >= matrix.channels || channel < 0) {
      throw std::invalid_argument("Channel index out of bounds");
    }
    int channel_size = matrix.rows * matrix.cols;
    int offset = channel * channel_size;

    for (int i = 0; i < channel_size; ++i) {
      T val = matrix.data[offset + i] + bias;
      matrix.data[offset + i] = T(1) / (T(1) + std::exp(-val));
    }
  }

  void apply_channel(Matrix<T> &matrix, int channel,
                     const std::vector<T> &channel_bias) const override {
    if (channel >= matrix.channels || channel < 0) {
      throw std::invalid_argument("Channel index out of bounds");
    }
    int channel_size = matrix.rows * matrix.cols;
    int offset = channel * channel_size;

    for (int i = 0; i < channel_size; ++i) {
      T val = matrix.data[offset + i] + channel_bias[i];
      matrix.data[offset + i] = T(1) / (T(1) + std::exp(-val));
    }
  }

  std::string name() const override { return "sigmoid"; }

  std::unique_ptr<ActivationFunction<T>> clone() const override {
    return std::make_unique<Sigmoid<T>>();
  }
};

// Softmax incompatible with Matrix<T>, designed for Tensor

template <typename T> class ActivationFactory {
private:
  static std::unordered_map<
      std::string, std::function<std::unique_ptr<ActivationFunction<T>>()>>
      creators;

public:
  static void register_activation(
      const std::string &name,
      std::function<std::unique_ptr<ActivationFunction<T>>()> creator) {
    creators[name] = creator;
  }

  static std::unique_ptr<ActivationFunction<T>>
  create(const std::string &name) {
    auto it = creators.find(name);
    if (it != creators.end()) {
      return it->second();
    }
    return nullptr;
  }

  static void register_defaults() {
    register_activation("relu", []() { return std::make_unique<ReLU<T>>(); });
    register_activation("leaky_relu",
                        []() { return std::make_unique<ReLU<T>>(T(0.01)); });
    register_activation("sigmoid",
                        []() { return std::make_unique<Sigmoid<T>>(); });
    register_activation("linear",
                        []() { return std::make_unique<Linear<T>>(); });
  }
};

template <typename T>
std::unordered_map<std::string,
                   std::function<std::unique_ptr<ActivationFunction<T>>()>>
    ActivationFactory<T>::creators;

namespace Activations {

template <typename T = double>
void apply_relu(Matrix<T> &matrix, const Matrix<T> *bias = nullptr,
                T negative_slope = T(0)) {
  ReLU<T> relu(negative_slope);
  relu.apply(matrix, bias);
}

template <typename T = double>
void apply_sigmoid(Matrix<T> &matrix, const Matrix<T> *bias = nullptr) {
  Sigmoid<T> sigmoid;
  sigmoid.apply(matrix, bias);
}

template <typename T = double>
void apply_linear(Matrix<T> &matrix, const Matrix<T> *bias = nullptr) {
  Linear<T> linear;
  linear.apply(matrix, bias);
}

// Channel-wise operations for CNNs
template <typename T = double>
void apply_relu_channel(Matrix<T> &matrix, int channel, T bias = T(0),
                        T negative_slope = T(0)) {
  ReLU<T> relu(negative_slope);
  relu.apply_channel(matrix, channel, bias);
}
} // namespace Activations

// Usage example:
/*
Matrix<T> feature_map(32, 32, 3);  // 32x32 RGB image
Matrix<T> bias(32, 32, 3);

// Method 1: Using factory
auto factory = ActivationFactory<double>();
factory.register_defaults();
auto relu = factory.create("relu");
relu->apply(feature_map, &bias);

// Method 2: Direct functions
Activations::apply_relu(feature_map, &bias);

// Method 3: Channel-wise (useful for CNNs)
Activations::apply_relu_channel(feature_map, 0, 0.1); // Apply to red channel
with bias 0.1
*/