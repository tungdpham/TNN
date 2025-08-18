#include "parameterized_layer.hpp"

namespace tnn {

template <typename T>
std::vector<Tensor<T> *> ParameterizedLayer<T>::parameters() {
    std::vector<Tensor<T> *> params;
    collect_parameters(params);
    return params;
}

template <typename T>
std::vector<Tensor<T> *> ParameterizedLayer<T>::gradients() {
    std::vector<Tensor<T> *> grads;
    collect_gradients(grads);
    return grads;
}

template <typename T>
void ParameterizedLayer<T>::update_parameters(Optimizer<T> &optimizer) {
    update_parameters_impl(optimizer);
}

} // namespace tnn
