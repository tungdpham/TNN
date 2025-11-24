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
#include "threading/thread_handler.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

// Forward declare stb_image functions
extern "C" {
unsigned char *stbi_load(const char *filename, int *x, int *y, int *channels_in_file,
                         int desired_channels);
void stbi_image_free(void *retval_from_stbi_load);
}

namespace tiny_imagenet_constants {
constexpr size_t IMAGE_HEIGHT = 64;
constexpr size_t IMAGE_WIDTH = 64;
constexpr size_t NUM_CLASSES = 200;
constexpr size_t NUM_CHANNELS = 3;
constexpr float NORMALIZATION_FACTOR = 255.0f;
constexpr size_t TRAIN_IMAGES_PER_CLASS = 500;
constexpr size_t VAL_IMAGES = 10000;
} // namespace tiny_imagenet_constants

namespace tnn {

/**
 * Tiny ImageNet-200 data loader for JPEG format adapted for CNN (2D RGB images)
 * Extends ImageDataLoader for proper inheritance
 *
 * Dataset structure:
 * - 200 classes
 * - 500 training images per class (100,000 total)
 * - 50 validation images per class (10,000 total)
 * - Images are 64x64 RGB
 * - Training images organized in directories by class ID (wnid)
 * - Validation images in a single directory with annotations file
 */
template <typename T = float> class TinyImageNetDataLoader : public ImageDataLoader<T> {
private:
  std::vector<std::vector<T>> data_;
  std::vector<int> labels_;

  std::vector<Tensor<T>> batched_data_;
  std::vector<Tensor<T>> batched_labels_;
  bool batches_prepared_;

  std::vector<std::string> class_ids_;                  // WordNet IDs (wnids)
  std::map<std::string, int> class_id_to_index_;        // Map wnid to class index
  std::map<std::string, std::string> class_id_to_name_; // Map wnid to human-readable name

  std::unique_ptr<AugmentationStrategy<T>> augmentation_strategy_;

  /**
   * Load class IDs from wnids.txt
   */
  bool load_class_ids(const std::string &dataset_dir) {
    std::string wnids_file = dataset_dir + "/wnids.txt";
    std::ifstream file(wnids_file);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open " << wnids_file << std::endl;
      return false;
    }

    class_ids_.clear();
    class_id_to_index_.clear();

    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
      // Remove trailing whitespace
      line.erase(line.find_last_not_of(" \n\r\t") + 1);
      if (!line.empty()) {
        class_ids_.push_back(line);
        class_id_to_index_[line] = index++;
      }
    }

    std::cout << "Loaded " << class_ids_.size() << " class IDs" << std::endl;
    return class_ids_.size() == tiny_imagenet_constants::NUM_CLASSES;
  }

  /**
   * Load class names from words.txt
   */
  void load_class_names(const std::string &dataset_dir) {
    std::string words_file = dataset_dir + "/words.txt";
    std::ifstream file(words_file);
    if (!file.is_open()) {
      std::cerr << "Warning: Could not open " << words_file << std::endl;
      return;
    }

    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string wnid, name;
      if (iss >> wnid) {
        // Only load names for the 200 classes we're actually using
        if (class_id_to_index_.find(wnid) != class_id_to_index_.end()) {
          // Get the rest of the line as the class name
          std::getline(iss, name);
          // Remove leading whitespace
          name.erase(0, name.find_first_not_of(" \t"));
          class_id_to_name_[wnid] = name;
        }
      }
    }

    std::cout << "Loaded " << class_id_to_name_.size() << " class names" << std::endl;
  }

  /**
   * Load a JPEG image and convert to normalized float data
   */
  bool load_jpeg_image(const std::string &image_path, std::vector<T> &image_data) {
    int width, height, channels;
    unsigned char *img = stbi_load(image_path.c_str(), &width, &height, &channels, 3);

    if (!img) {
      std::cerr << "Error loading image: " << image_path << std::endl;
      return false;
    }

    if (width != static_cast<int>(tiny_imagenet_constants::IMAGE_WIDTH) ||
        height != static_cast<int>(tiny_imagenet_constants::IMAGE_HEIGHT)) {
      std::cerr << "Warning: Image " << image_path << " has unexpected dimensions: " << width << "x"
                << height << std::endl;
      stbi_image_free(img);
      return false;
    }

    // Convert from HWC (Height, Width, Channels) to CHW (Channels, Height, Width) format
    // and normalize to [0, 1]
    image_data.resize(tiny_imagenet_constants::NUM_CHANNELS *
                      tiny_imagenet_constants::IMAGE_HEIGHT * tiny_imagenet_constants::IMAGE_WIDTH);

    for (size_t c = 0; c < tiny_imagenet_constants::NUM_CHANNELS; ++c) {
      for (size_t h = 0; h < tiny_imagenet_constants::IMAGE_HEIGHT; ++h) {
        for (size_t w = 0; w < tiny_imagenet_constants::IMAGE_WIDTH; ++w) {
          size_t src_idx = (h * tiny_imagenet_constants::IMAGE_WIDTH + w) * 3 + c;
          size_t dst_idx =
              c * tiny_imagenet_constants::IMAGE_HEIGHT * tiny_imagenet_constants::IMAGE_WIDTH +
              h * tiny_imagenet_constants::IMAGE_WIDTH + w;
          image_data[dst_idx] =
              static_cast<T>(img[src_idx]) / tiny_imagenet_constants::NORMALIZATION_FACTOR;
        }
      }
    }

    stbi_image_free(img);
    return true;
  }

  /**
   * Load training data from directory structure
   */
  bool load_train_data(const std::string &dataset_dir) {
    std::string train_dir = dataset_dir + "/train";

    if (!std::filesystem::exists(train_dir)) {
      std::cerr << "Error: Training directory not found: " << train_dir << std::endl;
      return false;
    }

    size_t total_loaded = 0;

    for (const auto &class_id : class_ids_) {
      std::string class_dir = train_dir + "/" + class_id + "/images";

      if (!std::filesystem::exists(class_dir)) {
        std::cerr << "Warning: Class directory not found: " << class_dir << std::endl;
        continue;
      }

      int class_index = class_id_to_index_[class_id];
      size_t class_count = 0;

      for (const auto &entry : std::filesystem::directory_iterator(class_dir)) {
        if (entry.path().extension() == ".JPEG") {
          std::vector<T> image_data;
          if (load_jpeg_image(entry.path().string(), image_data)) {
            data_.push_back(std::move(image_data));
            labels_.push_back(class_index);
            class_count++;
            total_loaded++;
          }
        }
      }

      if (class_count > 0 && total_loaded % 10000 == 0) {
        std::cout << "Loaded " << total_loaded << " images..." << std::endl;
      }
    }

    std::cout << "Loaded " << total_loaded << " training images" << std::endl;
    return total_loaded > 0;
  }

  /**
   * Load validation data from annotations file
   */
  bool load_val_data(const std::string &dataset_dir) {
    std::string val_dir = dataset_dir + "/val";
    std::string val_annotations = val_dir + "/val_annotations.txt";
    std::string val_images_dir = val_dir + "/images";

    if (!std::filesystem::exists(val_annotations)) {
      std::cerr << "Error: Validation annotations not found: " << val_annotations << std::endl;
      return false;
    }

    std::ifstream file(val_annotations);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open validation annotations file" << std::endl;
      return false;
    }

    std::string line;
    size_t total_loaded = 0;

    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string image_name, class_id;

      // Format: image_name class_id x y w h
      if (iss >> image_name >> class_id) {
        std::string image_path = val_images_dir + "/" + image_name;

        if (class_id_to_index_.find(class_id) == class_id_to_index_.end()) {
          std::cerr << "Warning: Unknown class ID: " << class_id << std::endl;
          continue;
        }

        int class_index = class_id_to_index_[class_id];
        std::vector<T> image_data;

        if (load_jpeg_image(image_path, image_data)) {
          data_.push_back(std::move(image_data));
          labels_.push_back(class_index);
          total_loaded++;
        }
      }
    }

    std::cout << "Loaded " << total_loaded << " validation images" << std::endl;
    return total_loaded > 0;
  }

public:
  TinyImageNetDataLoader() : ImageDataLoader<T>(), batches_prepared_(false) {
    data_.reserve(100000); // Reserve for training set
    labels_.reserve(100000);
  }

  virtual ~TinyImageNetDataLoader() = default;

  /**
   * Load Tiny ImageNet-200 data
   * @param source Path to dataset directory containing train/, val/, wnids.txt, and words.txt
   * @param is_train If true, load training data; if false, load validation data
   * @return true if successful, false otherwise
   */
  bool load_data(const std::string &source, bool is_train = true) {
    data_.clear();
    labels_.clear();

    // Load class IDs and names
    if (!load_class_ids(source)) {
      return false;
    }

    load_class_names(source);

    // Load the appropriate dataset
    bool success;
    if (is_train) {
      success = load_train_data(source);
    } else {
      success = load_val_data(source);
    }

    if (success) {
      this->current_index_ = 0;
      std::cout << "Total loaded: " << data_.size() << " samples" << std::endl;
    }

    return success;
  }

  /**
   * Overload for compatibility with BaseDataLoader interface
   */
  bool load_data(const std::string &source) override {
    return load_data(source, true); // Default to training data
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
   * Get a specific batch size (supports both pre-computed and on-demand batches)
   */
  bool get_batch(size_t batch_size, Tensor<T> &batch_data, Tensor<T> &batch_labels) override {
    // If batches are prepared and match requested size, use them
    if (batches_prepared_ && batch_size == this->batch_size_) {
      return get_next_batch(batch_data, batch_labels);
    }

    // Otherwise, create batch on demand
    if (this->current_index_ >= data_.size()) {
      return false;
    }

    const size_t actual_batch_size = std::min(batch_size, data_.size() - this->current_index_);

    batch_data =
        Tensor<T>({actual_batch_size, tiny_imagenet_constants::NUM_CHANNELS,
                   tiny_imagenet_constants::IMAGE_HEIGHT, tiny_imagenet_constants::IMAGE_WIDTH});

    batch_labels = Tensor<T>({actual_batch_size, tiny_imagenet_constants::NUM_CLASSES, 1, 1});
    batch_labels.fill(static_cast<T>(0.0));
    for (size_t i = 0; i < actual_batch_size; ++i) {
      const auto &image_data = data_[this->current_index_ + i];

      // Copy image data in CHW format
      for (size_t c = 0; c < tiny_imagenet_constants::NUM_CHANNELS; ++c) {
        for (size_t h = 0; h < tiny_imagenet_constants::IMAGE_HEIGHT; ++h) {
          for (size_t w = 0; w < tiny_imagenet_constants::IMAGE_WIDTH; ++w) {
            size_t pixel_idx =
                c * tiny_imagenet_constants::IMAGE_HEIGHT * tiny_imagenet_constants::IMAGE_WIDTH +
                h * tiny_imagenet_constants::IMAGE_WIDTH + w;
            batch_data(i, c, h, w) = image_data[pixel_idx];
          }
        }
      }

      // Set one-hot label
      const int label = labels_[this->current_index_ + i];
      if (label >= 0 && label < static_cast<int>(tiny_imagenet_constants::NUM_CLASSES)) {
        batch_labels(i, label, 0, 0) = static_cast<T>(1.0);
      }
    }

    // Apply augmentation if strategy is set
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
      // Shuffle raw data
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
      // Shuffle batched data
      this->current_batch_index_ = 0;

      std::vector<size_t> indices = this->generate_shuffled_indices(batched_data_.size());

      std::vector<Tensor<T>> shuffled_data;
      std::vector<Tensor<T>> shuffled_labels;
      shuffled_data.reserve(batched_data_.size());
      shuffled_labels.reserve(batched_labels_.size());

      for (const auto &idx : indices) {
        shuffled_data.emplace_back(std::move(batched_data_[idx]));
        shuffled_labels.emplace_back(std::move(batched_labels_[idx]));
      }

      batched_data_ = std::move(shuffled_data);
      batched_labels_ = std::move(shuffled_labels);
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
    return {tiny_imagenet_constants::NUM_CHANNELS, tiny_imagenet_constants::IMAGE_HEIGHT,
            tiny_imagenet_constants::IMAGE_WIDTH};
  }

  /**
   * Get number of classes
   */
  int get_num_classes() const override {
    return static_cast<int>(tiny_imagenet_constants::NUM_CLASSES);
  }

  /**
   * Get class names for Tiny ImageNet-200
   */
  std::vector<std::string> get_class_names() const override {
    std::vector<std::string> names;
    names.reserve(class_ids_.size());

    for (const auto &class_id : class_ids_) {
      auto it = class_id_to_name_.find(class_id);
      if (it != class_id_to_name_.end()) {
        names.push_back(it->second);
      } else {
        names.push_back(class_id); // Fall back to wnid if name not found
      }
    }

    return names;
  }

  /**
   * Get class IDs (WordNet IDs)
   */
  std::vector<std::string> get_class_ids() const { return class_ids_; }

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

    // Generate a shuffled list of sample indices to improve within-batch label diversity
    std::vector<size_t> shuffled_indices = this->generate_shuffled_indices(num_samples);

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      const size_t start_idx = batch_idx * batch_size;
      const size_t end_idx = std::min(start_idx + batch_size, num_samples);
      const size_t actual_batch_size = end_idx - start_idx;

      Tensor<T> batch_data({actual_batch_size, tiny_imagenet_constants::NUM_CHANNELS,
                            tiny_imagenet_constants::IMAGE_HEIGHT,
                            tiny_imagenet_constants::IMAGE_WIDTH});

      Tensor<T> batch_labels({actual_batch_size, tiny_imagenet_constants::NUM_CLASSES, 1, 1});
      batch_labels.fill(static_cast<T>(0.0));

      for (size_t i = 0; i < actual_batch_size; ++i) {
        const size_t sample_idx = shuffled_indices[start_idx + i];
        const auto &image_data = data_[sample_idx];

        // Copy image data
        for (size_t c = 0; c < tiny_imagenet_constants::NUM_CHANNELS; ++c) {
          for (size_t h = 0; h < tiny_imagenet_constants::IMAGE_HEIGHT; ++h) {
            for (size_t w = 0; w < tiny_imagenet_constants::IMAGE_WIDTH; ++w) {
              size_t pixel_idx =
                  c * tiny_imagenet_constants::IMAGE_HEIGHT * tiny_imagenet_constants::IMAGE_WIDTH +
                  h * tiny_imagenet_constants::IMAGE_WIDTH + w;
              batch_data(i, c, h, w) = image_data[pixel_idx];
            }
          }
        }

        // Set one-hot label
        const int label = labels_[sample_idx];
        if (label >= 0 && label < static_cast<int>(tiny_imagenet_constants::NUM_CLASSES)) {
          batch_labels(i, label, 0, 0) = static_cast<T>(1.0);
        }
      }

      // Apply augmentation if strategy is set
      if (augmentation_strategy_) {
        augmentation_strategy_->apply(batch_data, batch_labels);
      }

      batched_data_.emplace_back(std::move(batch_data));
      batched_labels_.emplace_back(std::move(batch_labels));

      if ((batch_idx + 1) % 100 == 0) {
        std::cout << "Prepared " << (batch_idx + 1) << "/" << num_batches << " batches..."
                  << std::endl;
      }
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

    std::vector<int> label_counts(tiny_imagenet_constants::NUM_CLASSES, 0);
    for (const auto &label : labels_) {
      if (label >= 0 && label < static_cast<int>(tiny_imagenet_constants::NUM_CLASSES)) {
        label_counts[label]++;
      }
    }

    std::cout << "Tiny ImageNet-200 Dataset Statistics:" << std::endl;
    std::cout << "Total samples: " << data_.size() << std::endl;
    std::cout << "Image shape: " << tiny_imagenet_constants::NUM_CHANNELS << "x"
              << tiny_imagenet_constants::IMAGE_HEIGHT << "x"
              << tiny_imagenet_constants::IMAGE_WIDTH << std::endl;
    std::cout << "Number of classes: " << tiny_imagenet_constants::NUM_CLASSES << std::endl;

    // Show distribution for first 10 classes
    std::cout << "Class distribution (first 10 classes):" << std::endl;
    for (int i = 0; i < std::min(10, static_cast<int>(class_ids_.size())); ++i) {
      std::string class_name = class_ids_[i];
      auto it = class_id_to_name_.find(class_name);
      if (it != class_id_to_name_.end()) {
        class_name = it->second;
      }
      std::cout << "  Class " << i << " (" << class_ids_[i] << " - " << class_name
                << "): " << label_counts[i] << " samples" << std::endl;
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

void create_tiny_image_loader(std::string data_path, TinyImageNetDataLoader<float> &train_loader,
                              TinyImageNetDataLoader<float> &val_loader) {
  if (!train_loader.load_data(data_path, true)) {
    throw std::runtime_error("Failed to load training data!");
  }
  if (!val_loader.load_data(data_path, false)) {
    throw std::runtime_error("Failed to load validation data!");
  }
}

using TinyImageNetDataLoaderFloat = TinyImageNetDataLoader<float>;
using TinyImageNetDataLoaderDouble = TinyImageNetDataLoader<double>;

} // namespace tnn
