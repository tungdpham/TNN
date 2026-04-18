/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>

#include "data_loading/image_data_loader.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_handler.hpp"

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
constexpr size_t IMAGE_SIZE = NUM_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH;
}  // namespace tiny_imagenet_constants

namespace tnn {
/**
 * Tiny ImageNet-200 data loader for JPEG format adapted for CNN (2D RGB images)
 * NHWC format: (Batch, Height, Width, Channels)
 *
 * Uses lazy loading: only the file paths and labels are stored in memory.
 * JPEG images are decoded on-demand during get_batch, so the full ~4.7 GB
 * pixel buffer is never resident in RAM.
 *
 * Dataset structure:
 * - 200 classes
 * - 500 training images per class (100,000 total)
 * - 50 validation images per class (10,000 total)
 * - Images are 64x64 RGB
 */
class TinyImageNetDataLoader : public ImageDataLoader {
private:
  // (image_path, class_index) — the only persistent storage per sample
  Vec<std::pair<std::string, int>> sample_list_;
  // Access order — shuffled in-place; current_index_ indexes into this
  Vec<size_t> access_order_;

  DType_t dtype_ = DType_t::FP32;

  Vec<std::string> class_ids_;                           // WordNet IDs (wnids)
  std::map<std::string, int> class_id_to_index_;         // Map wnid to class index
  std::map<std::string, std::string> class_id_to_name_;  // Map wnid to human-readable name

  template <typename T>
  bool get_batch_impl(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) {
    if (this->current_index_ >= access_order_.size()) return false;

    const size_t actual_batch_size =
        std::min(batch_size, access_order_.size() - this->current_index_);

    // NHWC format: (Batch, Height, Width, Channels)
    batch_data = make_tensor<T>({actual_batch_size, tiny_imagenet_constants::IMAGE_HEIGHT,
                                 tiny_imagenet_constants::IMAGE_WIDTH,
                                 tiny_imagenet_constants::NUM_CHANNELS});
    batch_labels = make_tensor<int>({actual_batch_size});

    parallel_for<size_t>(0, actual_batch_size, [&](size_t i) {
      const size_t sample_idx = access_order_[this->current_index_ + i];
      const auto &[path, class_index] = sample_list_[sample_idx];

      // Temporary CHW buffer for stbi_load output
      float chw_buf[tiny_imagenet_constants::IMAGE_SIZE];
      if (!load_jpeg_image(path, chw_buf)) return;

      // Convert from CHW float to NHWC tensor
      for (size_t c = 0; c < tiny_imagenet_constants::NUM_CHANNELS; ++c) {
        for (size_t h = 0; h < tiny_imagenet_constants::IMAGE_HEIGHT; ++h) {
          for (size_t w = 0; w < tiny_imagenet_constants::IMAGE_WIDTH; ++w) {
            const size_t src_idx =
                c * tiny_imagenet_constants::IMAGE_HEIGHT * tiny_imagenet_constants::IMAGE_WIDTH +
                h * tiny_imagenet_constants::IMAGE_WIDTH + w;
            batch_data->at<T>({i, h, w, c}) = static_cast<T>(chw_buf[src_idx]);
          }
        }
      }

      batch_labels->at<int>({i}) = class_index;
    });

    this->apply_augmentation(batch_data, batch_labels);
    this->current_index_ += actual_batch_size;
    return true;
  }

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
  bool load_jpeg_image(const std::string &image_path, float *image_data_ptr) {
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
    for (size_t c = 0; c < tiny_imagenet_constants::NUM_CHANNELS; ++c) {
      for (size_t h = 0; h < tiny_imagenet_constants::IMAGE_HEIGHT; ++h) {
        for (size_t w = 0; w < tiny_imagenet_constants::IMAGE_WIDTH; ++w) {
          size_t src_idx = (h * tiny_imagenet_constants::IMAGE_WIDTH + w) * 3 + c;
          size_t dst_idx =
              c * tiny_imagenet_constants::IMAGE_HEIGHT * tiny_imagenet_constants::IMAGE_WIDTH +
              h * tiny_imagenet_constants::IMAGE_WIDTH + w;
          image_data_ptr[dst_idx] = img[src_idx] / tiny_imagenet_constants::NORMALIZATION_FACTOR;
        }
      }
    }

    stbi_image_free(img);
    return true;
  }

  /**
   * Enumerate training images from directory structure and build the sample list.
   * No pixel data is loaded here — images are decoded on-demand in get_batch.
   */
  bool load_train_data(const std::string &dataset_dir) {
    std::string train_dir = dataset_dir + "/train";

    if (!std::filesystem::exists(train_dir)) {
      std::cerr << "Error: Training directory not found: " << train_dir << std::endl;
      return false;
    }

    for (const auto &class_id : class_ids_) {
      std::string class_dir = train_dir + "/" + class_id + "/images";

      if (!std::filesystem::exists(class_dir)) {
        std::cerr << "Warning: Class directory not found: " << class_dir << std::endl;
        continue;
      }

      int class_index = class_id_to_index_[class_id];

      for (const auto &entry : std::filesystem::directory_iterator(class_dir)) {
        if (entry.path().extension() == ".JPEG") {
          sample_list_.emplace_back(entry.path().string(), class_index);
        }
      }
    }

    std::cout << "Indexed " << sample_list_.size() << " training images (lazy)" << std::endl;
    return !sample_list_.empty();
  }

  /**
   * Parse the validation annotations file and build the sample list.
   * No pixel data is loaded here — images are decoded on-demand in get_batch.
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
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string image_name, class_id;

      // Format: image_name class_id x y w h
      if (!(iss >> image_name >> class_id)) continue;

      auto it = class_id_to_index_.find(class_id);
      if (it == class_id_to_index_.end()) {
        std::cerr << "Warning: Unknown class ID: " << class_id << std::endl;
        continue;
      }

      sample_list_.emplace_back(val_images_dir + "/" + image_name, it->second);
    }

    std::cout << "Indexed " << sample_list_.size() << " validation images (lazy)" << std::endl;
    return !sample_list_.empty();
  }

public:
  explicit TinyImageNetDataLoader(DType_t dtype = DType_t::FP32)
      : ImageDataLoader(),
        dtype_(dtype) {
    sample_list_.reserve(100000);
  }

  virtual ~TinyImageNetDataLoader() = default;

  /**
   * Load Tiny ImageNet-200 data.
   * Enumerates image paths and labels only — no pixel data is loaded here.
   * @param source Path to dataset directory containing train/, val/, wnids.txt, words.txt
   * @param is_train If true, index training data; if false, index validation data
   */
  bool load_data(const std::string &source, bool is_train = true) {
    sample_list_.clear();

    if (!load_class_ids(source)) return false;
    load_class_names(source);

    bool success = is_train ? load_train_data(source) : load_val_data(source);

    if (success) {
      const size_t n = sample_list_.size();
      access_order_.resize(n);
      std::iota(access_order_.begin(), access_order_.end(), 0);
      this->current_index_ = 0;
      std::cout << "Total indexed: " << n << " samples" << std::endl;
    }

    return success;
  }

  bool load_data(const std::string &source) override { return load_data(source, true); }

  bool get_batch(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) override {
    DISPATCH_DTYPE(dtype_, T, return get_batch_impl<T>(batch_size, batch_data, batch_labels));
  }

  void reset() override { this->current_index_ = 0; }

  /**
   * Shuffle the access order without touching any pixel data.
   */
  void shuffle() override {
    if (access_order_.empty()) return;
    access_order_ = this->generate_shuffled_indices(access_order_.size());
    this->current_index_ = 0;
  }

  size_t size() const override { return sample_list_.size(); }

  Vec<size_t> get_data_shape() const override {
    return {tiny_imagenet_constants::IMAGE_HEIGHT, tiny_imagenet_constants::IMAGE_WIDTH,
            tiny_imagenet_constants::NUM_CHANNELS};
  }

  int get_num_classes() const override {
    return static_cast<int>(tiny_imagenet_constants::NUM_CLASSES);
  }

  Vec<std::string> get_class_names() const override {
    Vec<std::string> names;
    names.reserve(class_ids_.size());
    for (const auto &class_id : class_ids_) {
      auto it = class_id_to_name_.find(class_id);
      names.push_back(it != class_id_to_name_.end() ? it->second : class_id);
    }
    return names;
  }

  Vec<std::string> get_class_ids() const { return class_ids_; }

  void print_data_stats() const override {
    if (sample_list_.empty()) {
      std::cout << "No data indexed" << std::endl;
      return;
    }

    Vec<int> label_counts(tiny_imagenet_constants::NUM_CLASSES, 0);
    for (const auto &[path, label] : sample_list_) {
      if (label >= 0 && label < static_cast<int>(tiny_imagenet_constants::NUM_CLASSES)) {
        label_counts[label]++;
      }
    }

    std::cout << "Tiny ImageNet-200 Dataset Statistics (NHWC format, lazy-loaded):" << std::endl;
    std::cout << "Total samples: " << sample_list_.size() << std::endl;
    std::cout << "Image shape: " << tiny_imagenet_constants::IMAGE_HEIGHT << "x"
              << tiny_imagenet_constants::IMAGE_WIDTH << "x"
              << tiny_imagenet_constants::NUM_CHANNELS << std::endl;
    std::cout << "Number of classes: " << tiny_imagenet_constants::NUM_CLASSES << std::endl;
    std::cout << "Class distribution (first 10 classes):" << std::endl;
    for (int i = 0; i < std::min(10, static_cast<int>(class_ids_.size())); ++i) {
      std::string display_name = class_ids_[i];
      auto it = class_id_to_name_.find(display_name);
      if (it != class_id_to_name_.end()) display_name = it->second;
      std::cout << "  Class " << i << " (" << class_ids_[i] << " - " << display_name
                << "): " << label_counts[i] << " samples" << std::endl;
    }
  }

  static void create(const std::string &data_path, TinyImageNetDataLoader &train_loader,
                     TinyImageNetDataLoader &val_loader) {
    if (!train_loader.load_data(data_path, true)) {
      throw std::runtime_error("Failed to index training data!");
    }
    if (!val_loader.load_data(data_path, false)) {
      throw std::runtime_error("Failed to index validation data!");
    }
  }
};
}  // namespace tnn
