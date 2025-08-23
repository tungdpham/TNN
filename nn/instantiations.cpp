// This file explicitly instantiates all commonly used template classes
// to avoid recompiling them in every translation unit

#include "nn/sequential.hpp"
#include "nn/layers.hpp"
#include "nn/activations.hpp"
#include "nn/optimizers.hpp"
#include "nn/loss.hpp"
#include "tensor/tensor.hpp"
#include "matrix/matrix.hpp"

namespace tnn {
  
// Sequential model instantiations
template class Sequential<float>;
template class Sequential<double>;

// Sequential builder instantiations
template class SequentialBuilder<float>;
template class SequentialBuilder<double>;

// Layer instantiations
template class Layer<float>;
template class Layer<double>;
template class StatelessLayer<float>;
template class StatelessLayer<double>;
template class ParameterizedLayer<float>;
template class ParameterizedLayer<double>;

// Specific layer instantiations
template class DenseLayer<float>;
template class DenseLayer<double>;
template class ActivationLayer<float>;
template class ActivationLayer<double>;
template class Conv2DLayer<float>;
template class Conv2DLayer<double>;
template class MaxPool2DLayer<float>;
template class MaxPool2DLayer<double>;
template class DropoutLayer<float>;
template class DropoutLayer<double>;
template class FlattenLayer<float>;
template class FlattenLayer<double>;
template class BatchNormLayer<float>;
template class BatchNormLayer<double>;

// Factory instantiations
template class LayerFactory<float>;
template class LayerFactory<double>;
template class ActivationFactory<float>;
template class ActivationFactory<double>;
template class OptimizerFactory<float>;
template class OptimizerFactory<double>;
template class LossFactory<float>;
template class LossFactory<double>;

// Activation function instantiations
template class ActivationFunction<float>;
template class ActivationFunction<double>;
template class ReLU<float>;
template class ReLU<double>;
template class Sigmoid<float>;
template class Sigmoid<double>;
template class Linear<float>;
template class Linear<double>;
template class Softmax<float>;
template class Softmax<double>;

// Optimizer instantiations
template class Optimizer<float>;
template class Optimizer<double>;
template class SGD<float>;
template class SGD<double>;
template class Adam<float>;
template class Adam<double>;

// Loss function instantiations
template class Loss<float>;
template class Loss<double>;
template class CrossEntropyLoss<float>;
template class CrossEntropyLoss<double>;
template class MSELoss<float>;
template class MSELoss<double>;
template class MAELoss<float>;
template class MAELoss<double>;
template class HuberLoss<float>;
template class HuberLoss<double>;

// Convenience function instantiations
template std::unique_ptr<Layer<float>> dense<float>(size_t, size_t, const std::string&, bool, const std::string&);
template std::unique_ptr<Layer<double>> dense<double>(size_t, size_t, const std::string&, bool, const std::string&);

template std::unique_ptr<Layer<float>> conv2d<float>(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const std::string&, bool, const std::string&);
template std::unique_ptr<Layer<double>> conv2d<double>(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const std::string&, bool, const std::string&);

template std::unique_ptr<Layer<float>> activation<float>(const std::string&, const std::string&);
template std::unique_ptr<Layer<double>> activation<double>(const std::string&, const std::string&);

template std::unique_ptr<Layer<float>> maxpool2d<float>(size_t, size_t, size_t, size_t, size_t, size_t, const std::string&);
template std::unique_ptr<Layer<double>> maxpool2d<double>(size_t, size_t, size_t, size_t, size_t, size_t, const std::string&);

template std::unique_ptr<Layer<float>> dropout<float>(float, const std::string&);
template std::unique_ptr<Layer<double>> dropout<double>(double, const std::string&);

template std::unique_ptr<Layer<float>> batchnorm<float>(size_t, float, float, bool, const std::string&);
template std::unique_ptr<Layer<double>> batchnorm<double>(size_t, double, double, bool, const std::string&);

template std::unique_ptr<Layer<float>> flatten<float>(const std::string&);
template std::unique_ptr<Layer<double>> flatten<double>(const std::string&);

template std::unique_ptr<ActivationFunction<float>> create_activation<float>(const std::string&);
template std::unique_ptr<ActivationFunction<double>> create_activation<double>(const std::string&);

} // namespace tnn