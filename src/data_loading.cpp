/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */


#include "data_loading/cifar100_data_loader.hpp"
#include "data_loading/cifar10_data_loader.hpp"
#include "data_loading/data_loader.hpp"
#include "data_loading/mnist_data_loader.hpp"

namespace data_loading {
template class BaseDataLoader<float>;
template class BaseDataLoader<double>;

template class MNISTDataLoader<float>;
template class MNISTDataLoader<double>;

template class CIFAR10DataLoader<float>;
template class CIFAR10DataLoader<double>;

template class CIFAR100DataLoader<float>;
template class CIFAR100DataLoader<double>;

} // namespace data_loading
