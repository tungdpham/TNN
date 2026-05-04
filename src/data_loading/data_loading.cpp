/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "data_augmentation/augmentation.hpp"
#include "data_loading/cifar100_data_loader.hpp"
#include "data_loading/cifar10_data_loader.hpp"
#include "data_loading/data_loader_factory.hpp"
#include "data_loading/imagenet100_data_loader.hpp"
#include "data_loading/mnist_data_loader.hpp"
#include "data_loading/open_webtext_data_loader.hpp"
#include "data_loading/tiny_imagenet_data_loader.hpp"
#include "type/type.hpp"

#include <cstdlib>
#include <string>

namespace tnn {

namespace {
bool env_flag_enabled(const char *primary, const char *fallback, bool default_value) {
  const char *raw = std::getenv(primary);
  if (!raw && fallback) {
    raw = std::getenv(fallback);
  }
  if (!raw) {
    return default_value;
  }
  std::string v(raw);
  return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "YES" || v == "on" ||
         v == "ON";
}
}  // namespace

DataLoaderPair DataLoaderFactory::create(const std::string &dataset_type,
                                         const std::string &dataset_path, DType_t io_dtype_) {
  DataLoaderPair pair;

  if (dataset_type == "mnist") {
    auto train = std::make_unique<MNISTDataLoader>(io_dtype_);
    auto val = std::make_unique<MNISTDataLoader>(io_dtype_);

    if (train->load_data(dataset_path + "/train.csv") ||
        train->load_data(dataset_path + "/mnist_train.csv")) {
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path + "/test.csv") ||
        val->load_data(dataset_path + "/mnist_test.csv")) {
      pair.val = std::move(val);
    }
  } else if (dataset_type == "cifar10") {
    auto train = std::make_unique<CIFAR10DataLoader>(io_dtype_);
    auto val = std::make_unique<CIFAR10DataLoader>(io_dtype_);

    Vec<std::string> train_files = {
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
    auto train = std::make_unique<CIFAR100DataLoader>(false, io_dtype_);
    auto val = std::make_unique<CIFAR100DataLoader>(false, io_dtype_);

    if (train->load_data(dataset_path + "/train.bin")) {
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path + "/test.bin")) {
      pair.val = std::move(val);
    }
  } else if (dataset_type == "tiny_imagenet") {
    auto train = std::make_unique<TinyImageNetDataLoader>(io_dtype_);
    auto val = std::make_unique<TinyImageNetDataLoader>(io_dtype_);

    if (train->load_data(dataset_path, true)) {
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path, false)) {
      pair.val = std::move(val);
    }
  } else if (dataset_type == "imagenet100") {
    auto train = std::make_unique<ImageNet100DataLoader>(io_dtype_);
    auto val = std::make_unique<ImageNet100DataLoader>(io_dtype_);

    if (train->load_data(dataset_path, true)) {
      const bool use_aug = env_flag_enabled("TNN_AUGMENTATION", "AUGMENTATION", true);
      if (use_aug) {
        train->set_augmentation(AugmentationBuilder()
                                    // RandomResizedCrop(224) is performed inside
                                    // ImageNet100DataLoader from the original JPEG.
                                    .horizontal_flip(0.5f)
                                    .brightness(1.0f, 0.10f)
                                    .contrast(1.0f, 0.10f)
                                    .saturation(1.0f, 0.10f)
                                    .normalize({0.485f, 0.456f, 0.406f},
                                               {0.229f, 0.224f, 0.225f})
                                    .build());
        std::cout << "[Augmentation] ImageNet100 train: "
                  << "loader RandomResizedCrop(224) + Flip(0.5) + "
                  << "ColorJitter(b=0.1,c=0.1,s=0.1) + Normalize" << std::endl;
      } else {
        train->set_augmentation(AugmentationBuilder()
                                    .normalize({0.485f, 0.456f, 0.406f},
                                               {0.229f, 0.224f, 0.225f})
                                    .build());
        std::cout << "[Augmentation] ImageNet100 train: disabled | "
                  << "Normalize only after loader spatial preprocessing"
                  << std::endl;
      }
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path, false)) {
      val->set_augmentation(AugmentationBuilder()
                                // Resize(256)+CenterCrop(224) is performed inside
                                // ImageNet100DataLoader from the original JPEG.
                                .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
                                .build());
      std::cout << "[Augmentation] ImageNet100 val: "
                << "loader Resize(256)+CenterCrop(224) + Normalize" << std::endl;
      pair.val = std::move(val);
    }
  } else if (dataset_type == "open_webtext") {
    auto train = std::make_unique<OpenWebTextDataLoader>(1024, io_dtype_);
    auto val = std::make_unique<OpenWebTextDataLoader>(1024, io_dtype_);

    if (train->load_data(dataset_path + "/train.bin")) {
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path + "/val.bin")) {
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
}  // namespace tnn
