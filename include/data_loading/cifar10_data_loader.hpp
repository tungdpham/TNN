/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "image_data_loader.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
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

namespace data_loading {

/**
 * Data augmentation parameters for CIFAR-10
 */
struct CIFAR10AugmentationConfig {
  bool enable_horizontal_flip = true;
  bool enable_rotation = true;
  bool enable_brightness_contrast = true;
  bool enable_noise = false;
  bool enable_random_crop = true;

  float horizontal_flip_prob = 0.5f;
  float rotation_prob = 0.5f;
  float brightness_contrast_prob = 0.5f;
  float noise_prob = 0.3f;
  float random_crop_prob = 0.5f;

  float max_rotation_degrees = 15.0f;
  float brightness_range = 0.2f; // ±20% brightness
  float contrast_range = 0.2f;   // ±20% contrast
  float noise_std = 0.05f;       // Standard deviation for Gaussian noise
  int crop_padding = 4;          // Padding for random crop

  CIFAR10AugmentationConfig() = default;
};

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

  // Data augmentation configuration
  CIFAR10AugmentationConfig aug_config_;
  bool augmentation_enabled_ = false;

  /**
   * Apply horizontal flip augmentation to image data
   */
  void apply_horizontal_flip(std::vector<T> &image_data) const {
    std::vector<T> flipped(image_data.size());

    for (int c = 0; c < static_cast<int>(cifar10_constants::NUM_CHANNELS); ++c) {
      for (int h = 0; h < static_cast<int>(cifar10_constants::IMAGE_HEIGHT); ++h) {
        for (int w = 0; w < static_cast<int>(cifar10_constants::IMAGE_WIDTH); ++w) {
          int src_idx = c * cifar10_constants::IMAGE_HEIGHT * cifar10_constants::IMAGE_WIDTH +
                        h * cifar10_constants::IMAGE_WIDTH + w;
          int dst_idx = c * cifar10_constants::IMAGE_HEIGHT * cifar10_constants::IMAGE_WIDTH +
                        h * cifar10_constants::IMAGE_WIDTH +
                        (cifar10_constants::IMAGE_WIDTH - 1 - w);
          flipped[dst_idx] = image_data[src_idx];
        }
      }
    }
    image_data = std::move(flipped);
  }

  /**
   * Apply rotation augmentation to image data
   */
  void apply_rotation(std::vector<T> &image_data, float angle_degrees) const {
    const float angle_rad = angle_degrees * M_PI / 180.0f;
    const float cos_angle = std::cos(angle_rad);
    const float sin_angle = std::sin(angle_rad);
    const int center_x = cifar10_constants::IMAGE_WIDTH / 2;
    const int center_y = cifar10_constants::IMAGE_HEIGHT / 2;

    std::vector<T> rotated(image_data.size(), static_cast<T>(0));

    for (int c = 0; c < static_cast<int>(cifar10_constants::NUM_CHANNELS); ++c) {
      for (int y = 0; y < static_cast<int>(cifar10_constants::IMAGE_HEIGHT); ++y) {
        for (int x = 0; x < static_cast<int>(cifar10_constants::IMAGE_WIDTH); ++x) {
          // Transform coordinates to center origin
          float src_x = (x - center_x) * cos_angle - (y - center_y) * sin_angle + center_x;
          float src_y = (x - center_x) * sin_angle + (y - center_y) * cos_angle + center_y;

          // Bilinear interpolation
          int x1 = static_cast<int>(std::floor(src_x));
          int y1 = static_cast<int>(std::floor(src_y));
          int x2 = x1 + 1;
          int y2 = y1 + 1;

          if (x1 >= 0 && x2 < static_cast<int>(cifar10_constants::IMAGE_WIDTH) && y1 >= 0 &&
              y2 < static_cast<int>(cifar10_constants::IMAGE_HEIGHT)) {

            float wx = src_x - x1;
            float wy = src_y - y1;

            int base_idx = c * cifar10_constants::IMAGE_HEIGHT * cifar10_constants::IMAGE_WIDTH;
            T val1 = image_data[base_idx + y1 * cifar10_constants::IMAGE_WIDTH + x1];
            T val2 = image_data[base_idx + y1 * cifar10_constants::IMAGE_WIDTH + x2];
            T val3 = image_data[base_idx + y2 * cifar10_constants::IMAGE_WIDTH + x1];
            T val4 = image_data[base_idx + y2 * cifar10_constants::IMAGE_WIDTH + x2];

            T interpolated = val1 * (1 - wx) * (1 - wy) + val2 * wx * (1 - wy) +
                             val3 * (1 - wx) * wy + val4 * wx * wy;

            rotated[c * cifar10_constants::IMAGE_HEIGHT * cifar10_constants::IMAGE_WIDTH +
                    y * cifar10_constants::IMAGE_WIDTH + x] = interpolated;
          }
        }
      }
    }
    image_data = std::move(rotated);
  }

  /**
   * Apply brightness and contrast adjustments
   */
  void apply_brightness_contrast(std::vector<T> &image_data, float brightness_factor,
                                 float contrast_factor) const {
    for (auto &pixel : image_data) {
      pixel = std::clamp(pixel * contrast_factor + brightness_factor, static_cast<T>(0),
                         static_cast<T>(1));
    }
  }

  /**
   * Apply Gaussian noise augmentation
   */
  void apply_noise(std::vector<T> &image_data, float noise_std) const {
    std::normal_distribution<float> noise_dist(0.0f, noise_std);

    for (auto &pixel : image_data) {
      float noise = noise_dist(this->rng_);
      pixel = std::clamp(pixel + static_cast<T>(noise), static_cast<T>(0), static_cast<T>(1));
    }
  }

  /**
   * Apply random crop with padding
   */
  void apply_random_crop(std::vector<T> &image_data, int padding) const {
    if (padding <= 0)
      return;

    // Create padded image
    const int padded_size = cifar10_constants::IMAGE_WIDTH + 2 * padding;
    std::vector<T> padded(cifar10_constants::NUM_CHANNELS * padded_size * padded_size,
                          static_cast<T>(0));

    // Copy original image to center of padded image
    for (int c = 0; c < static_cast<int>(cifar10_constants::NUM_CHANNELS); ++c) {
      for (int h = 0; h < static_cast<int>(cifar10_constants::IMAGE_HEIGHT); ++h) {
        for (int w = 0; w < static_cast<int>(cifar10_constants::IMAGE_WIDTH); ++w) {
          int src_idx = c * cifar10_constants::IMAGE_HEIGHT * cifar10_constants::IMAGE_WIDTH +
                        h * cifar10_constants::IMAGE_WIDTH + w;
          int dst_idx = c * padded_size * padded_size + (h + padding) * padded_size + (w + padding);
          padded[dst_idx] = image_data[src_idx];
        }
      }
    }

    // Random crop from padded image
    std::uniform_int_distribution<int> crop_dist(0, 2 * padding);
    int start_x = crop_dist(this->rng_);
    int start_y = crop_dist(this->rng_);

    std::vector<T> cropped(image_data.size());
    for (int c = 0; c < static_cast<int>(cifar10_constants::NUM_CHANNELS); ++c) {
      for (int h = 0; h < static_cast<int>(cifar10_constants::IMAGE_HEIGHT); ++h) {
        for (int w = 0; w < static_cast<int>(cifar10_constants::IMAGE_WIDTH); ++w) {
          int src_idx = c * padded_size * padded_size + (start_y + h) * padded_size + (start_x + w);
          int dst_idx = c * cifar10_constants::IMAGE_HEIGHT * cifar10_constants::IMAGE_WIDTH +
                        h * cifar10_constants::IMAGE_WIDTH + w;
          cropped[dst_idx] = padded[src_idx];
        }
      }
    }
    image_data = std::move(cropped);
  }

  /**
   * Apply all enabled augmentations to a single image
   */
  void apply_augmentations(std::vector<T> &image_data) const {
    if (!augmentation_enabled_)
      return;

    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    // Apply horizontal flip
    if (aug_config_.enable_horizontal_flip &&
        prob_dist(this->rng_) < aug_config_.horizontal_flip_prob) {
      apply_horizontal_flip(image_data);
    }

    // Apply rotation
    if (aug_config_.enable_rotation && prob_dist(this->rng_) < aug_config_.rotation_prob) {
      std::uniform_real_distribution<float> angle_dist(-aug_config_.max_rotation_degrees,
                                                       aug_config_.max_rotation_degrees);
      float angle = angle_dist(this->rng_);
      apply_rotation(image_data, angle);
    }

    // Apply random crop
    if (aug_config_.enable_random_crop && prob_dist(this->rng_) < aug_config_.random_crop_prob) {
      apply_random_crop(image_data, aug_config_.crop_padding);
    }

    // Apply brightness and contrast adjustments
    if (aug_config_.enable_brightness_contrast &&
        prob_dist(this->rng_) < aug_config_.brightness_contrast_prob) {
      std::uniform_real_distribution<float> brightness_dist(-aug_config_.brightness_range,
                                                            aug_config_.brightness_range);
      std::uniform_real_distribution<float> contrast_dist(1.0f - aug_config_.contrast_range,
                                                          1.0f + aug_config_.contrast_range);
      float brightness = brightness_dist(this->rng_);
      float contrast = contrast_dist(this->rng_);
      apply_brightness_contrast(image_data, brightness, contrast);
    }

    // Apply noise
    if (aug_config_.enable_noise && prob_dist(this->rng_) < aug_config_.noise_prob) {
      apply_noise(image_data, aug_config_.noise_std);
    }
  }

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

    batch_data = Tensor<T>(actual_batch_size, cifar10_constants::NUM_CHANNELS,
                           cifar10_constants::IMAGE_HEIGHT, cifar10_constants::IMAGE_WIDTH);

    batch_labels = Tensor<T>(actual_batch_size, cifar10_constants::NUM_CLASSES, 1, 1);
    batch_labels.fill(static_cast<T>(0.0));

    for (size_t i = 0; i < actual_batch_size; ++i) {
      // Make a copy of the image data for augmentation
      std::vector<T> image_data = data_[this->current_index_ + i];

      // Apply augmentations if enabled
      apply_augmentations(image_data);

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

      Tensor<T> batch_data(actual_batch_size, cifar10_constants::NUM_CHANNELS,
                           cifar10_constants::IMAGE_HEIGHT, cifar10_constants::IMAGE_WIDTH);

      Tensor<T> batch_labels(actual_batch_size, cifar10_constants::NUM_CLASSES, 1, 1);
      batch_labels.fill(static_cast<T>(0.0));

      for (size_t i = 0; i < actual_batch_size; ++i) {
        const size_t sample_idx = start_idx + i;

        // Make a copy of the image data for augmentation
        std::vector<T> image_data = data_[sample_idx];

        // Apply augmentations if enabled
        apply_augmentations(image_data);

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
   * Enable or disable data augmentation
   */
  void set_augmentation_enabled(bool enabled) {
    augmentation_enabled_ = enabled;
    if (enabled) {
      std::cout << "Data augmentation enabled" << std::endl;
    } else {
      std::cout << "Data augmentation disabled" << std::endl;
    }
  }

  /**
   * Get current augmentation status
   */
  bool is_augmentation_enabled() const { return augmentation_enabled_; }

  /**
   * Set augmentation configuration
   */
  void set_augmentation_config(const CIFAR10AugmentationConfig &config) {
    aug_config_ = config;
    std::cout << "Updated augmentation configuration" << std::endl;
  }

  /**
   * Get current augmentation configuration
   */
  const CIFAR10AugmentationConfig &get_augmentation_config() const { return aug_config_; }

  /**
   * Enable specific augmentation with custom probability
   */
  void enable_horizontal_flip(float probability = 0.5f) {
    aug_config_.enable_horizontal_flip = true;
    aug_config_.horizontal_flip_prob = probability;
  }

  void enable_rotation(float probability = 0.5f, float max_degrees = 15.0f) {
    aug_config_.enable_rotation = true;
    aug_config_.rotation_prob = probability;
    aug_config_.max_rotation_degrees = max_degrees;
  }

  void enable_brightness_contrast(float probability = 0.5f, float brightness_range = 0.2f,
                                  float contrast_range = 0.2f) {
    aug_config_.enable_brightness_contrast = true;
    aug_config_.brightness_contrast_prob = probability;
    aug_config_.brightness_range = brightness_range;
    aug_config_.contrast_range = contrast_range;
  }

  void enable_noise(float probability = 0.3f, float std_dev = 0.05f) {
    aug_config_.enable_noise = true;
    aug_config_.noise_prob = probability;
    aug_config_.noise_std = std_dev;
  }

  void enable_random_crop(float probability = 0.5f, int padding = 4) {
    aug_config_.enable_random_crop = true;
    aug_config_.random_crop_prob = probability;
    aug_config_.crop_padding = padding;
  }

  /**
   * Disable specific augmentations
   */
  void disable_horizontal_flip() { aug_config_.enable_horizontal_flip = false; }
  void disable_rotation() { aug_config_.enable_rotation = false; }
  void disable_brightness_contrast() { aug_config_.enable_brightness_contrast = false; }
  void disable_noise() { aug_config_.enable_noise = false; }
  void disable_random_crop() { aug_config_.enable_random_crop = false; }

  /**
   * Print current augmentation settings
   */
  void print_augmentation_config() const {
    std::cout << "\nData Augmentation Configuration:" << std::endl;
    std::cout << "Augmentation enabled: " << (augmentation_enabled_ ? "Yes" : "No") << std::endl;
    if (augmentation_enabled_) {
      std::cout << "Horizontal flip: " << (aug_config_.enable_horizontal_flip ? "Yes" : "No")
                << " (prob: " << aug_config_.horizontal_flip_prob << ")" << std::endl;
      std::cout << "Rotation: " << (aug_config_.enable_rotation ? "Yes" : "No")
                << " (prob: " << aug_config_.rotation_prob
                << ", max_degrees: " << aug_config_.max_rotation_degrees << ")" << std::endl;
      std::cout << "Brightness/Contrast: "
                << (aug_config_.enable_brightness_contrast ? "Yes" : "No")
                << " (prob: " << aug_config_.brightness_contrast_prob << ", brightness_range: ±"
                << aug_config_.brightness_range << ", contrast_range: ±"
                << aug_config_.contrast_range << ")" << std::endl;
      std::cout << "Noise: " << (aug_config_.enable_noise ? "Yes" : "No")
                << " (prob: " << aug_config_.noise_prob << ", std: " << aug_config_.noise_std << ")"
                << std::endl;
      std::cout << "Random crop: " << (aug_config_.enable_random_crop ? "Yes" : "No")
                << " (prob: " << aug_config_.random_crop_prob
                << ", padding: " << aug_config_.crop_padding << ")" << std::endl;
    }
    std::cout << std::endl;
  }

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

using CIFAR10DataLoaderFloat = CIFAR10DataLoader<float>;
using CIFAR10DataLoaderDouble = CIFAR10DataLoader<double>;

} // namespace data_loading
