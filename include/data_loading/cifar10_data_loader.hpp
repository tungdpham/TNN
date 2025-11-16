/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "data_augmentation/augmentation.hpp"
#include "image_data_loader.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <string>
#include <vector>

namespace cifar10_constants {
constexpr size_t IMAGE_HEIGHT = 32;
constexpr size_t IMAGE_WIDTH = 32;
constexpr size_t IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3;
constexpr size_t NUM_CLASSES = 10;
constexpr size_t NUM_CHANNELS = 3;
constexpr float NORMALIZATION_FACTOR = 255.0f;
constexpr size_t RECORD_SIZE = 1 + IMAGE_SIZE;
} // namespace cifar10_constants

namespace tnn {

/**
 * Enhanced CIFAR-10 data loader for binary format adapted for CNN (2D RGB
 * images) Extends ImageDataLoader for proper inheritance
 */
template <typename T = float> class CIFAR10DataLoader : public ImageDataLoader<T> {
private:
  std::vector<std::vector<T>> data_;
  std::vector<int> labels_;

  std::vector<Tensor<T>> batched_data_;
  std::vector<Tensor<T>> batched_labels_;
  bool batches_prepared_;

  std::vector<std::string> class_names_ = {"airplane", "automobile", "bird",  "cat",  "deer",
                                           "dog",      "frog",       "horse", "ship", "truck"};

  std::unique_ptr<AugmentationStrategy<T>> augmentation_strategy_;

public:
  CIFAR10DataLoader() : ImageDataLoader<T>(), batches_prepared_(false) {

    data_.reserve(50000);
    labels_.reserve(50000);
  }

  virtual ~CIFAR10DataLoader() = default;

  /**
   * Load CIFAR-10 data from binary file(s)
   * @param source Path to binary file or directory containing multiple files
   * @return true if successful, false otherwise
   */
  bool load_data(const std::string &source) override {

    std::vector<std::string> filenames;

    if (source.find(".bin") != std::string::npos) {
      filenames.push_back(source);
    } else {

      std::cerr << "Error: For multiple files, use load_multiple_files() method" << std::endl;
      return false;
    }

    return load_multiple_files(filenames);
  }

  /**
   * Load CIFAR-10 data from multiple binary files
   * @param filenames Vector of file paths to load
   * @return true if successful, false otherwise
   */
  bool load_multiple_files(const std::vector<std::string> &filenames) {
    data_.clear();
    labels_.clear();

    for (const auto &filename : filenames) {
      std::ifstream file(filename, std::ios::binary);
      if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
      }

      char buffer[cifar10_constants::RECORD_SIZE];
      size_t records_loaded = 0;

      while (file.read(buffer, cifar10_constants::RECORD_SIZE)) {

        labels_.push_back(static_cast<int>(static_cast<unsigned char>(buffer[0])));

        std::vector<T> image_data;
        image_data.reserve(cifar10_constants::IMAGE_SIZE);

        for (size_t i = 1; i < cifar10_constants::RECORD_SIZE; ++i) {
          image_data.push_back(static_cast<T>(static_cast<unsigned char>(buffer[i]) /
                                              cifar10_constants::NORMALIZATION_FACTOR));
        }

        data_.push_back(std::move(image_data));
        records_loaded++;
      }

      std::cout << "Loaded " << records_loaded << " samples from " << filename << std::endl;
    }

    this->current_index_ = 0;
    std::cout << "Total loaded: " << data_.size() << " samples" << std::endl;
    return !data_.empty();
  }

  /**
   * Get the next batch of data using pre-computed batches
   */
  bool get_next_batch(Tensor<T> &batch_data, Tensor<T> &batch_labels) override {
    if (!batches_prepared_) {
      std::cerr << "Error: Batches not prepared! Call prepare_batches() first." << std::endl;
      return false;
    }

    if (this->current_batch_index_ >= batched_data_.size()) {
      return false;
    }

    batch_data = batched_data_[this->current_batch_index_].clone();
    batch_labels = batched_labels_[this->current_batch_index_].clone();
    ++this->current_batch_index_;

    return true;
  }

  /**
   * Get a specific batch size (supports both pre-computed and on-demand
   * batches)
   */
  bool get_batch(size_t batch_size, Tensor<T> &batch_data, Tensor<T> &batch_labels) override {

    if (batches_prepared_ && batch_size == this->batch_size_) {
      return get_next_batch(batch_data, batch_labels);
    }

    if (this->current_index_ >= data_.size()) {
      return false;
    }

    const size_t actual_batch_size = std::min(batch_size, data_.size() - this->current_index_);

    batch_data = Tensor<T>({actual_batch_size, cifar10_constants::NUM_CHANNELS,
                            cifar10_constants::IMAGE_HEIGHT, cifar10_constants::IMAGE_WIDTH});

    batch_labels = Tensor<T>({actual_batch_size, cifar10_constants::NUM_CLASSES, 1, 1});
    batch_labels.fill(static_cast<T>(0.0));

    for (size_t i = 0; i < actual_batch_size; ++i) {
      const std::vector<T> &image_data = data_[this->current_index_ + i];

      for (int c = 0; c < static_cast<int>(cifar10_constants::NUM_CHANNELS); ++c) {
        for (int h = 0; h < static_cast<int>(cifar10_constants::IMAGE_HEIGHT); ++h) {
          for (int w = 0; w < static_cast<int>(cifar10_constants::IMAGE_WIDTH); ++w) {
            size_t pixel_idx =
                c * cifar10_constants::IMAGE_HEIGHT * cifar10_constants::IMAGE_WIDTH +
                h * cifar10_constants::IMAGE_WIDTH + w;
            batch_data(i, c, h, w) = image_data[pixel_idx];
          }
        }
      }

      const int label = labels_[this->current_index_ + i];
      if (label >= 0 && label < static_cast<int>(cifar10_constants::NUM_CLASSES)) {
        batch_labels(i, label, 0, 0) = static_cast<T>(1.0);
      }
    }

    if (augmentation_strategy_) {
      augmentation_strategy_->apply(batch_data, batch_labels);
    }

    this->current_index_ += actual_batch_size;
    return true;
  }

  /**
   * Reset iterator to beginning of dataset
   */
  void reset() override {
    this->current_index_ = 0;
    this->current_batch_index_ = 0;
  }

  /**
   * Shuffle the dataset
   */
  void shuffle() override {
    if (!batches_prepared_) {
      if (data_.empty())
        return;

      std::vector<size_t> indices = this->generate_shuffled_indices(data_.size());

      std::vector<std::vector<T>> shuffled_data;
      std::vector<int> shuffled_labels;
      shuffled_data.reserve(data_.size());
      shuffled_labels.reserve(labels_.size());

      for (const auto &idx : indices) {
        shuffled_data.emplace_back(std::move(data_[idx]));
        shuffled_labels.emplace_back(labels_[idx]);
      }

      data_ = std::move(shuffled_data);
      labels_ = std::move(shuffled_labels);
      this->current_index_ = 0;
    } else {
      this->current_batch_index_ = 0;

      std::vector<size_t> indices = this->generate_shuffled_indices(batched_data_.size());

      std::vector<Tensor<T>> shuffled_data;
      std::vector<Tensor<T>> shuffled_labels;

      for (const auto &idx : indices) {
        shuffled_data.emplace_back(std::move(batched_data_[idx]));
        shuffled_labels.emplace_back(std::move(batched_labels_[idx]));
      }

      batched_data_ = std::move(shuffled_data);
      batched_labels_ = std::move(shuffled_labels);
      this->current_batch_index_ = 0;
    }
  }

  /**
   * Get the total number of samples in the dataset
   */
  size_t size() const override { return data_.size(); }

  /**
   * Get image dimensions (channels, height, width)
   */
  std::vector<size_t> get_image_shape() const override {
    return {cifar10_constants::NUM_CHANNELS, cifar10_constants::IMAGE_HEIGHT,
            cifar10_constants::IMAGE_WIDTH};
  }

  /**
   * Get number of classes
   */
  int get_num_classes() const override { return static_cast<int>(cifar10_constants::NUM_CLASSES); }

  /**
   * Get class names for CIFAR-10
   */
  std::vector<std::string> get_class_names() const override { return class_names_; }

  /**
   * Pre-compute all batches for efficient training
   */
  void prepare_batches(size_t batch_size) override {
    if (data_.empty()) {
      std::cerr << "Warning: No data loaded, cannot prepare batches!" << std::endl;
      return;
    }

    this->batch_size_ = batch_size;
    this->batches_prepared_ = true;
    batched_data_.clear();
    batched_labels_.clear();

    const size_t num_samples = data_.size();
    const size_t num_batches = (num_samples + batch_size - 1) / batch_size;

    batched_data_.reserve(num_batches);
    batched_labels_.reserve(num_batches);

    std::cout << "Preparing " << num_batches << " batches of size " << batch_size << "..."
              << std::endl;

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      const size_t start_idx = batch_idx * batch_size;
      const size_t end_idx = std::min(start_idx + batch_size, num_samples);
      const size_t actual_batch_size = end_idx - start_idx;

      Tensor<T> batch_data({actual_batch_size, cifar10_constants::NUM_CHANNELS,
                            cifar10_constants::IMAGE_HEIGHT, cifar10_constants::IMAGE_WIDTH});

      Tensor<T> batch_labels({actual_batch_size, cifar10_constants::NUM_CLASSES, 1, 1});
      batch_labels.fill(static_cast<T>(0.0));

      for (size_t i = 0; i < actual_batch_size; ++i) {
        const size_t sample_idx = start_idx + i;
        const std::vector<T> &image_data = data_[sample_idx];

        for (size_t c = 0; c < cifar10_constants::NUM_CHANNELS; ++c) {
          for (size_t h = 0; h < cifar10_constants::IMAGE_HEIGHT; ++h) {
            for (size_t w = 0; w < cifar10_constants::IMAGE_WIDTH; ++w) {
              size_t pixel_idx =
                  c * cifar10_constants::IMAGE_HEIGHT * cifar10_constants::IMAGE_WIDTH +
                  h * cifar10_constants::IMAGE_WIDTH + w;
              batch_data(i, c, h, w) = image_data[pixel_idx];
            }
          }
        }

        const int label = labels_[sample_idx];
        if (label >= 0 && label < static_cast<int>(cifar10_constants::NUM_CLASSES)) {
          batch_labels(i, label, 0, 0) = static_cast<T>(1.0);
        }
      }

      if (augmentation_strategy_) {
        augmentation_strategy_->apply(batch_data, batch_labels);
      }

      batched_data_.emplace_back(std::move(batch_data));
      batched_labels_.emplace_back(std::move(batch_labels));
    }

    this->current_batch_index_ = 0;
    batches_prepared_ = true;
    std::cout << "Batch preparation completed!" << std::endl;
  }

  /**
   * Get number of batches when using prepared batches
   */
  size_t num_batches() const override {
    return batches_prepared_ ? batched_data_.size() : BaseDataLoader<T>::num_batches();
  }

  /**
   * Check if batches are prepared
   */
  bool are_batches_prepared() const override { return batches_prepared_; }

  /**
   * Set augmentation strategy to apply during batch preparation and retrieval
   */
  void set_augmentation_strategy(std::unique_ptr<AugmentationStrategy<T>> strategy) {
    augmentation_strategy_ = std::move(strategy);
  }

  /**
   * Set augmentation strategy using a copy
   */
  void set_augmentation_strategy(const AugmentationStrategy<T> &strategy) {
    augmentation_strategy_ = std::make_unique<AugmentationStrategy<T>>();
    for (const auto &aug : strategy.get_augmentations()) {
      augmentation_strategy_->add_augmentation(aug->clone());
    }
  }

  /**
   * Clear the augmentation strategy
   */
  void clear_augmentation_strategy() { augmentation_strategy_.reset(); }

  /**
   * Check if augmentation is enabled
   */
  bool has_augmentation() const { return augmentation_strategy_ != nullptr; }

  /**
   * Get data statistics for debugging
   */
  void print_data_stats() const {
    if (data_.empty()) {
      std::cout << "No data loaded" << std::endl;
      return;
    }

    std::vector<int> label_counts(cifar10_constants::NUM_CLASSES, 0);
    for (const auto &label : labels_) {
      if (label >= 0 && label < static_cast<int>(cifar10_constants::NUM_CLASSES)) {
        label_counts[label]++;
      }
    }

    std::cout << "CIFAR-10 Dataset Statistics:" << std::endl;
    std::cout << "Total samples: " << data_.size() << std::endl;
    std::cout << "Image shape: " << cifar10_constants::NUM_CHANNELS << "x"
              << cifar10_constants::IMAGE_HEIGHT << "x" << cifar10_constants::IMAGE_WIDTH
              << std::endl;
    std::cout << "Class distribution:" << std::endl;
    for (int i = 0; i < static_cast<int>(cifar10_constants::NUM_CLASSES); ++i) {
      std::cout << "  " << class_names_[i] << " (" << i << "): " << label_counts[i] << " samples"
                << std::endl;
    }

    if (!data_.empty()) {
      T min_val = *std::min_element(data_[0].begin(), data_[0].end());
      T max_val = *std::max_element(data_[0].begin(), data_[0].end());
      T sum = std::accumulate(data_[0].begin(), data_[0].end(), static_cast<T>(0.0));
      T mean = sum / data_[0].size();

      std::cout << "Pixel value range: [" << min_val << ", " << max_val << "]" << std::endl;
      std::cout << "First image mean pixel value: " << mean << std::endl;
    }
  }
};

template <typename T>
void create_cifar10_dataloader(const std::string &data_path, CIFAR10DataLoader<T> &train_loader,
                               CIFAR10DataLoader<T> &test_loader) {
  if (!train_loader.load_multiple_files({data_path + "/cifar-10-batches-bin/data_batch_1.bin",
                                         data_path + "/cifar-10-batches-bin/data_batch_2.bin",
                                         data_path + "/cifar-10-batches-bin/data_batch_3.bin",
                                         data_path + "/cifar-10-batches-bin/data_batch_4.bin",
                                         data_path + "/cifar-10-batches-bin/data_batch_5.bin"})) {
    throw std::runtime_error("Failed to load training data!");
  }

  if (!test_loader.load_data(data_path + "/cifar-10-batches-bin/test_batch.bin")) {
    throw std::runtime_error("Failed to load test data!");
  }
}

using CIFAR10DataLoaderFloat = CIFAR10DataLoader<float>;
using CIFAR10DataLoaderDouble = CIFAR10DataLoader<double>;

} // namespace tnn
