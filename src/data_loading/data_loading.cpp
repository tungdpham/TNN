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
#include "data_loading/open_webtext_data_loader.hpp"
#include "data_loading/tiny_imagenet_data_loader.hpp"

namespace tnn {

template <typename T>
DataLoaderPair<T> DataLoaderFactory<T>::create(const std::string &dataset_type,
                                               const std::string &dataset_path) {
  DataLoaderPair<T> pair;

  if (dataset_type == "mnist") {
    auto train = std::make_unique<MNISTDataLoader<T>>();
    auto val = std::make_unique<MNISTDataLoader<T>>();

    if (train->load_data(dataset_path + "/train.csv") ||
        train->load_data(dataset_path + "/mnist_train.csv")) {
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path + "/test.csv") ||
        val->load_data(dataset_path + "/mnist_test.csv")) {
      pair.val = std::move(val);
    }
  } else if (dataset_type == "cifar10") {
    auto train = std::make_unique<CIFAR10DataLoader<T>>();
    auto val = std::make_unique<CIFAR10DataLoader<T>>();

    std::vector<std::string> train_files = {
        dataset_path + "/data_batch_1.bin", dataset_path + "/data_batch_2.bin",
        dataset_path + "/data_batch_3.bin", dataset_path + "/data_batch_4.bin",
        dataset_path + "/data_batch_5.bin"};

    if (train->load_multiple_files(train_files)) {
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path + "/test_batch.bin")) {
      pair.val = std::move(val);
    }
  } else if (dataset_type == "cifar100") {
    auto train = std::make_unique<CIFAR100DataLoader<T>>();
    auto val = std::make_unique<CIFAR100DataLoader<T>>();

    if (train->load_data(dataset_path + "/train.bin")) {
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path + "/test.bin")) {
      pair.val = std::move(val);
    }
  } else if (dataset_type == "tiny_imagenet") {
    auto train = std::make_unique<TinyImageNetDataLoader<T>>();
    auto val = std::make_unique<TinyImageNetDataLoader<T>>();

    if (train->load_data(dataset_path, true)) {
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path, false)) {
      pair.val = std::move(val);
    }
  } else if (dataset_type == "open_webtext") {
    auto train = std::make_unique<OpenWebTextDataLoader<T>>(512);
    auto val = std::make_unique<OpenWebTextDataLoader<T>>(512);

    if (train->load_data(dataset_path + "/train.bin")) {
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path + "/train.bin")) {
      pair.val = std::move(val);
    }
  } else {
    std::cerr << "Error: Unknown dataset type: " << dataset_type << std::endl;
  }

  // If we only have test/val, or we want to use test for val if val is missing
  if (!pair.val && pair.train) {
    // This case usually doesn't happen with the logic above, but per request:
    // "just take the test for val if it only has test"
    // (Though usually we want the opposite: if we only have one set, use it for both or split)
  }

  return pair;
}

template class DataLoaderFactory<float>;
template class DataLoaderFactory<double>;

template class BaseDataLoader<float>;
template class BaseDataLoader<double>;

template class MNISTDataLoader<float>;
template class MNISTDataLoader<double>;

template class CIFAR10DataLoader<float>;
template class CIFAR10DataLoader<double>;

template class CIFAR100DataLoader<float>;
template class CIFAR100DataLoader<double>;

} // namespace tnn
