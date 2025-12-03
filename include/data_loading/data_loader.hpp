/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "data_augmentation/augmentation.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace tnn {

/**
 * Abstract base class for all data loaders
 * Provides common interface and functionality for training neural networks
 */
template <typename T = float> class BaseDataLoader {
public:
  virtual ~BaseDataLoader() = default;

  /**
   * Load data from file(s)
   * @param source Data source (file path, directory, etc.)
   * @return true if successful, false otherwise
   */
  virtual bool load_data(const std::string &source) = 0;

  /**
   * Get the next batch of data
   * @param batch_data Output tensor for features/input data
   * @param batch_labels Output tensor for labels/targets
   * @return true if batch was retrieved, false if no more data
   */
  virtual bool get_next_batch(Tensor<T> &batch_data, Tensor<T> &batch_labels) = 0;

  /**
   * Get a specific batch size
   * @param batch_size Number of samples per batch
   * @param batch_data Output tensor for features/input data
   * @param batch_labels Output tensor for labels/targets
   * @return true if batch was retrieved, false if no more data
   */
  virtual bool get_batch(size_t batch_size, Tensor<T> &batch_data, Tensor<T> &batch_labels) = 0;

  /**
   * Reset iterator to beginning of dataset
   */
  virtual void reset() = 0;

  /**
   * Shuffle the dataset
   */
  virtual void shuffle() = 0;

  /**
   * Get the total number of samples in the dataset
   */
  virtual size_t size() const = 0;

  /**
   * Prepare batches for efficient training
   * @param batch_size Size of each batch
   */
  virtual void prepare_batches(size_t batch_size) {
    if (size() == 0) {
      std::cerr << "Warning: Cannot prepare batches - no data loaded" << std::endl;
      return;
    }

    batch_size_ = batch_size;
    batches_prepared_ = true;
    current_batch_index_ = 0;

    std::cout << "Preparing batches with size " << batch_size << " for " << size() << " samples..."
              << std::endl;
  }

  /**
   * Get number of batches when using prepared batches
   */
  virtual size_t num_batches() const {
    if (!batches_prepared_ || size() == 0)
      return 0;
    return (size() + batch_size_ - 1) / batch_size_;
  }

  /**
   * Check if batches are prepared
   */
  virtual bool are_batches_prepared() const { return batches_prepared_; }

  /**
   * Get current batch size
   */
  virtual int get_batch_size() const { return static_cast<int>(batch_size_); }

  /**
   * Set random seed for reproducible shuffling
   */
  virtual void set_seed(unsigned int seed) { rng_.seed(seed); }

  /**
   * Get random number generator for derived classes
   */
  std::mt19937 &get_rng() { return rng_; }
  const std::mt19937 &get_rng() const { return rng_; }

  /**
   * Set augmentation strategy
   * @param aug Unique pointer to augmentation strategy (takes ownership)
   *
   * This allows clean separation between data loading and augmentation.
   * Different datasets can use different augmentation strategies without
   * coupling the augmentation logic to the data loader.
   *
   * Example usage:
   *   auto aug = std::make_unique<CIFAR10Augmentation<float>>(0.1f, true);
   *   loader.set_augmentation(std::move(aug));
   */
  void set_augmentation(std::unique_ptr<AugmentationStrategy<T>> aug) {
    augmentation_ = std::move(aug);
  }

  /**
   * Remove augmentation strategy (useful for validation/test sets)
   */
  void clear_augmentation() { augmentation_.reset(); }

  /**
   * Check if augmentation is enabled
   */
  bool has_augmentation() const { return augmentation_ != nullptr; }

protected:
  size_t current_index_ = 0;
  size_t current_batch_index_ = 0;
  size_t batch_size_ = 32;
  bool batches_prepared_ = false;
  mutable std::mt19937 rng_{std::random_device{}()};
  std::unique_ptr<AugmentationStrategy<T>> augmentation_;

  /**
   * Apply augmentation to batch if augmentation strategy is set
   * Called internally by derived classes after loading batch data
   */
  void apply_augmentation(Tensor<T> &batch_data, Tensor<T> &batch_labels) {
    if (augmentation_) {
      augmentation_->apply(batch_data, batch_labels);
    }
  }

  /**
   * Utility function to shuffle indices
   */
  std::vector<size_t> generate_shuffled_indices(size_t data_size) const {
    std::vector<size_t> indices(data_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_);
    return indices;
  }

  /**
   * Utility function to create one-hot encoded labels
   */
  void create_one_hot_label(Tensor<T> &label_tensor, int batch_idx, int class_idx,
                            int num_classes) {

    label_tensor(batch_idx, class_idx, 0, 0) = static_cast<T>(1.0);
  }

  /**
   * Utility function to normalize data to [0, 1] range
   */
  void normalize_to_unit_range(std::vector<T> &data, T max_value = static_cast<T>(255.0)) {
    for (auto &value : data) {
      value /= max_value;
    }
  }
};

/**
 * Factory function to create appropriate data loader based on dataset type
 */
template <typename T = float>
std::unique_ptr<BaseDataLoader<T>> create_data_loader(const std::string &dataset_type) {

  return nullptr;
}

/**
 * Utility functions for common data loading operations
 */
namespace tnn {
/**
 * Split dataset into train/validation sets
 */
template <typename T>
std::pair<std::vector<size_t>, std::vector<size_t>>
train_val_split(size_t dataset_size, float val_ratio = 0.2f, unsigned int seed = 42) {
  std::vector<size_t> indices(dataset_size);
  std::iota(indices.begin(), indices.end(), 0);

  std::mt19937 rng(seed);
  std::shuffle(indices.begin(), indices.end(), rng);

  size_t val_size = static_cast<size_t>(dataset_size * val_ratio);
  size_t train_size = dataset_size - val_size;

  std::vector<size_t> train_indices(indices.begin(), indices.begin() + train_size);
  std::vector<size_t> val_indices(indices.begin() + train_size, indices.end());

  return {train_indices, val_indices};
}

} // namespace tnn

} // namespace tnn
