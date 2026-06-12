/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <cstdlib>
#include <string>

#include "data_augmentation/augmentation.hpp"
#include "data_loading/cifar100_data_loader.hpp"
#include "data_loading/cifar10_data_loader.hpp"
#include "data_loading/data_loader_factory.hpp"
#include "data_loading/imagenet100_data_loader.hpp"
#include "data_loading/mnist_data_loader.hpp"
#include "data_loading/open_webtext_data_loader.hpp"
#include "data_loading/tiny_imagenet_data_loader.hpp"
#include "type/type.hpp"

namespace synet {

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
      const bool use_aug = env_flag_enabled("SYNET_AUGMENTATION", "AUGMENTATION", true);
      if (use_aug) {
        train->set_augmentation(
            AugmentationBuilder()
                // .random_resized_crop(224, 224, 0.08f, 1.0f, 3.0f / 4.0f, 4.0f / 3.0f)
                .horizontal_flip(0.5f)
                .brightness(0.5f, 0.10f)
                .contrast(0.5f, 0.10f)
                .saturation(0.5f, 0.10f)
                .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
                .build());
      } else {
        train->set_augmentation(AugmentationBuilder()
                                    // .resize_center_crop(256, 224, 224)
                                    .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
                                    .build());
      }
      pair.train = std::move(train);
    }

    if (val->load_data(dataset_path, false)) {
      val->set_augmentation(AugmentationBuilder()
                                .resize_center_crop(256, 224, 224)
                                .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
                                .build());
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
}  // namespace synet
