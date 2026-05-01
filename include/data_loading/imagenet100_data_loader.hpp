/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>

#include "data_loading/image_data_loader.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_handler.hpp"

// Include nlohmann/json for parsing Labels.json
#include <nlohmann/json.hpp>

// Forward declare stb_image functions
extern "C" {
unsigned char *stbi_load(const char *filename, int *x, int *y, int *channels_in_file,
                         int desired_channels);
void stbi_image_free(void *retval_from_stbi_load);
unsigned char *stbi_load_from_memory(unsigned char const *buffer, int len, int *x, int *y,
                                     int *channels_in_file, int desired_channels);
}

// Forward declare stb_image_resize functions
extern "C" {
unsigned char *stbir_resize_uint8_linear(const unsigned char *input_pixels, int input_w,
                                         int input_h, int input_stride_in_bytes,
                                         unsigned char *output_pixels, int output_w, int output_h,
                                         int output_stride_in_bytes, int num_channels);
}

namespace imagenet100_constants {
constexpr size_t IMAGE_HEIGHT = 224;
constexpr size_t IMAGE_WIDTH = 224;
constexpr size_t NUM_CLASSES = 100;
constexpr size_t NUM_CHANNELS = 3;
constexpr float NORMALIZATION_FACTOR = 255.0f;
constexpr size_t IMAGE_SIZE = NUM_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH;
}  // namespace imagenet100_constants

namespace tnn {
/**
 * ImageNet-100 data loader for JPEG format adapted for CNN (2D RGB images)
 * NHWC format: (Batch, Height, Width, Channels)
 *
 * Uses lazy loading: only the file paths and labels are stored in memory.
 * JPEG images are decoded and resized on-demand during get_batch, so the full
 * pixel buffer is never resident in RAM.
 *
 * Dataset structure:
 * - 100 classes
 * - Training images split across train.X1, train.X2, train.X3, train.X4
 * - Validation images in val.X
 * - Images are resized to 224x224 RGB
 * - Labels.json maps class IDs (wnids) to human-readable names
 */
class ImageNet100DataLoader : public ImageDataLoader {
private:
  // (image_path, class_index) — the only persistent storage per sample
  Vec<std::pair<std::string, int>> sample_list_;
  // Access order — shuffled in-place; current_index_ indexes into this
  Vec<size_t> access_order_;

  DType_t dtype_ = DType_t::FP32;
  bool is_train_ = true;

  Vec<std::string> class_ids_;                           // WordNet IDs (wnids)
  std::map<std::string, int> class_id_to_index_;         // Map wnid to class index
  std::map<std::string, std::string> class_id_to_name_;  // Map wnid to human-readable name

  template <typename T>
  bool get_batch_impl(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) {
    if (this->current_index_ >= access_order_.size()) return false;

    const size_t actual_batch_size =
        std::min(batch_size, access_order_.size() - this->current_index_);

    // NHWC format: (Batch, Height, Width, Channels)
    batch_data =
        make_tensor<T>({actual_batch_size, imagenet100_constants::IMAGE_HEIGHT,
                        imagenet100_constants::IMAGE_WIDTH, imagenet100_constants::NUM_CHANNELS});
    batch_labels = make_tensor<int>({actual_batch_size});

    parallel_for<size_t>(0, actual_batch_size, [&](size_t i) {
      const size_t sample_idx = access_order_[this->current_index_ + i];
      const auto &[path, class_index] = sample_list_[sample_idx];

      // Temporary CHW buffer for image data
      float chw_buf[imagenet100_constants::IMAGE_SIZE];
      if (!load_jpeg_image(path, chw_buf, is_train_)) return;

      // Convert from CHW float to NHWC tensor
      for (size_t c = 0; c < imagenet100_constants::NUM_CHANNELS; ++c) {
        for (size_t h = 0; h < imagenet100_constants::IMAGE_HEIGHT; ++h) {
          for (size_t w = 0; w < imagenet100_constants::IMAGE_WIDTH; ++w) {
            const size_t src_idx =
                c * imagenet100_constants::IMAGE_HEIGHT * imagenet100_constants::IMAGE_WIDTH +
                h * imagenet100_constants::IMAGE_WIDTH + w;
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
   * Load class IDs and names from Labels.json
   */
  bool load_class_labels(const std::string &dataset_dir) {
    std::string labels_file = dataset_dir + "/Labels.json";
    std::ifstream file(labels_file);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open " << labels_file << std::endl;
      return false;
    }

    try {
      nlohmann::json labels_json;
      file >> labels_json;

      class_ids_.clear();
      class_id_to_index_.clear();
      class_id_to_name_.clear();

      int index = 0;
      for (auto it = labels_json.begin(); it != labels_json.end(); ++it) {
        std::string class_id = it.key();
        std::string class_name = it.value().get<std::string>();

        class_ids_.push_back(class_id);
        class_id_to_index_[class_id] = index++;
        class_id_to_name_[class_id] = class_name;
      }

      // Sort class IDs for consistent ordering
      std::sort(class_ids_.begin(), class_ids_.end());

      // Rebuild the index mapping after sorting
      class_id_to_index_.clear();
      for (size_t i = 0; i < class_ids_.size(); ++i) {
        class_id_to_index_[class_ids_[i]] = static_cast<int>(i);
      }

      std::cout << "Loaded " << class_ids_.size() << " class labels from JSON" << std::endl;
      return class_ids_.size() == imagenet100_constants::NUM_CLASSES;
    } catch (const std::exception &e) {
      std::cerr << "Error parsing Labels.json: " << e.what() << std::endl;
      return false;
    }
  }

  /**
   * Load a JPEG image and convert it to CHW float data in [0, 1].
   * Train: torchvision-like RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3)).
   * Val: torchvision-like Resize(256) + CenterCrop(224).
   */
  bool load_jpeg_image(const std::string &image_path,
                       float *image_data_ptr,
                       bool is_train) {
    int width, height, channels;
    unsigned char *img = stbi_load(image_path.c_str(), &width, &height, &channels, 3);

    if (!img) {
      std::cerr << "Error loading image: " << image_path << std::endl;
      return false;
    }

    constexpr int OUT_H = static_cast<int>(imagenet100_constants::IMAGE_HEIGHT);
    constexpr int OUT_W = static_cast<int>(imagenet100_constants::IMAGE_WIDTH);

    unsigned char *final_img = new unsigned char[OUT_H * OUT_W * 3];

    if (is_train) {
      // Torch-like RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3)).
      static thread_local std::mt19937 rng(std::random_device{}());
      std::uniform_real_distribution<float> scale_dist(0.08f, 1.0f);
      std::uniform_real_distribution<float> ratio_dist(std::log(3.0f / 4.0f),
                                                       std::log(4.0f / 3.0f));

      int crop_x = 0;
      int crop_y = 0;
      int crop_w = width;
      int crop_h = height;

      const int area = width * height;
      bool found = false;

      for (int attempt = 0; attempt < 10; ++attempt) {
        const float target_area = scale_dist(rng) * static_cast<float>(area);
        const float aspect = std::exp(ratio_dist(rng));

        const int w = static_cast<int>(std::round(std::sqrt(target_area * aspect)));
        const int h = static_cast<int>(std::round(std::sqrt(target_area / aspect)));

        if (w > 0 && h > 0 && w <= width && h <= height) {
          std::uniform_int_distribution<int> x_dist(0, width - w);
          std::uniform_int_distribution<int> y_dist(0, height - h);
          crop_x = x_dist(rng);
          crop_y = y_dist(rng);
          crop_w = w;
          crop_h = h;
          found = true;
          break;
        }
      }

      if (!found) {
        // Torch fallback: center crop with valid aspect-ratio bounds.
        const float in_ratio = static_cast<float>(width) / static_cast<float>(height);
        if (in_ratio < 3.0f / 4.0f) {
          crop_w = width;
          crop_h = static_cast<int>(std::round(crop_w / (3.0f / 4.0f)));
        } else if (in_ratio > 4.0f / 3.0f) {
          crop_h = height;
          crop_w = static_cast<int>(std::round(crop_h * (4.0f / 3.0f)));
        } else {
          crop_w = width;
          crop_h = height;
        }
        crop_w = std::min(crop_w, width);
        crop_h = std::min(crop_h, height);
        crop_x = std::max(0, (width - crop_w) / 2);
        crop_y = std::max(0, (height - crop_h) / 2);
      }

      const unsigned char *crop_ptr = img + (crop_y * width + crop_x) * 3;
      unsigned char *result = stbir_resize_uint8_linear(
          crop_ptr, crop_w, crop_h, width * 3,
          final_img, OUT_W, OUT_H, 0, 3);

      if (!result) {
        std::cerr << "Error random-resized-cropping image: " << image_path << std::endl;
        delete[] final_img;
        stbi_image_free(img);
        return false;
      }
    } else {
      // Torch-like validation: Resize(256) preserving aspect ratio, then CenterCrop(224).
      constexpr int RESIZE_SHORT_SIDE = 256;
      int resized_w = 0;
      int resized_h = 0;

      if (width < height) {
        resized_w = RESIZE_SHORT_SIDE;
        resized_h = static_cast<int>(std::round(
            static_cast<float>(height) * RESIZE_SHORT_SIDE / static_cast<float>(width)));
      } else {
        resized_h = RESIZE_SHORT_SIDE;
        resized_w = static_cast<int>(std::round(
            static_cast<float>(width) * RESIZE_SHORT_SIDE / static_cast<float>(height)));
      }

      resized_w = std::max(resized_w, OUT_W);
      resized_h = std::max(resized_h, OUT_H);

      unsigned char *resized_img = new unsigned char[resized_h * resized_w * 3];
      unsigned char *result = stbir_resize_uint8_linear(
          img, width, height, 0,
          resized_img, resized_w, resized_h, 0, 3);

      if (!result) {
        std::cerr << "Error resizing validation image: " << image_path << std::endl;
        delete[] resized_img;
        delete[] final_img;
        stbi_image_free(img);
        return false;
      }

      const int crop_x = std::max(0, (resized_w - OUT_W) / 2);
      const int crop_y = std::max(0, (resized_h - OUT_H) / 2);

      for (int y = 0; y < OUT_H; ++y) {
        const unsigned char *src_row = resized_img + ((crop_y + y) * resized_w + crop_x) * 3;
        unsigned char *dst_row = final_img + y * OUT_W * 3;
        std::copy(src_row, src_row + OUT_W * 3, dst_row);
      }

      delete[] resized_img;
    }

    stbi_image_free(img);

    // Convert from HWC [0,255] to CHW [0,1].
    for (size_t c = 0; c < imagenet100_constants::NUM_CHANNELS; ++c) {
      for (size_t h = 0; h < imagenet100_constants::IMAGE_HEIGHT; ++h) {
        for (size_t w = 0; w < imagenet100_constants::IMAGE_WIDTH; ++w) {
          const size_t src_idx = (h * imagenet100_constants::IMAGE_WIDTH + w) * 3 + c;
          const size_t dst_idx =
              c * imagenet100_constants::IMAGE_HEIGHT * imagenet100_constants::IMAGE_WIDTH +
              h * imagenet100_constants::IMAGE_WIDTH + w;
          image_data_ptr[dst_idx] =
              final_img[src_idx] / imagenet100_constants::NORMALIZATION_FACTOR;
        }
      }
    }

    delete[] final_img;
    return true;
  }

  /**
   * Enumerate training images from directory structure and build the sample list.
   * Training images are split across train.X1, train.X2, train.X3, train.X4
   * No pixel data is loaded here — images are decoded on-demand in get_batch.
   */
  bool load_train_data(const std::string &dataset_dir) {
    Vec<std::string> train_dirs = {"train.X1", "train.X2", "train.X3", "train.X4"};

    for (const auto &train_subdir : train_dirs) {
      std::string train_dir = dataset_dir + "/" + train_subdir;

      if (!std::filesystem::exists(train_dir)) {
        std::cerr << "Warning: Training directory not found: " << train_dir << std::endl;
        continue;
      }

      for (const auto &class_id : class_ids_) {
        std::string class_dir = train_dir + "/" + class_id;

        if (!std::filesystem::exists(class_dir)) {
          // Not all classes may be in each training subdirectory
          continue;
        }

        int class_index = class_id_to_index_[class_id];

        for (const auto &entry : std::filesystem::directory_iterator(class_dir)) {
          if (entry.path().extension() == ".JPEG" || entry.path().extension() == ".jpeg") {
            sample_list_.emplace_back(entry.path().string(), class_index);
          }
        }
      }
    }

    std::cout << "Indexed " << sample_list_.size() << " training images (lazy)" << std::endl;
    return !sample_list_.empty();
  }

  /**
   * Enumerate validation images from val.X directory.
   * No pixel data is loaded here — images are decoded on-demand in get_batch.
   */
  bool load_val_data(const std::string &dataset_dir) {
    std::string val_dir = dataset_dir + "/val.X";

    if (!std::filesystem::exists(val_dir)) {
      std::cerr << "Error: Validation directory not found: " << val_dir << std::endl;
      return false;
    }

    for (const auto &class_id : class_ids_) {
      std::string class_dir = val_dir + "/" + class_id;

      if (!std::filesystem::exists(class_dir)) {
        std::cerr << "Warning: Validation class directory not found: " << class_dir << std::endl;
        continue;
      }

      int class_index = class_id_to_index_[class_id];

      for (const auto &entry : std::filesystem::directory_iterator(class_dir)) {
        if (entry.path().extension() == ".JPEG" || entry.path().extension() == ".jpeg") {
          sample_list_.emplace_back(entry.path().string(), class_index);
        }
      }
    }

    std::cout << "Indexed " << sample_list_.size() << " validation images (lazy)" << std::endl;
    return !sample_list_.empty();
  }

public:
  explicit ImageNet100DataLoader(DType_t dtype = DType_t::FP32)
      : ImageDataLoader(),
        dtype_(dtype) {
    sample_list_.reserve(150000);  // ImageNet-100 has ~130k training images
  }

  virtual ~ImageNet100DataLoader() = default;

  /**
   * Load ImageNet-100 data.
   * Enumerates image paths and labels only — no pixel data is loaded here.
   * @param source Path to dataset directory containing train.X1-X4/, val.X/, Labels.json
   * @param is_train If true, index training data; if false, index validation data
   */
  bool load_data(const std::string &source, bool is_train = true) {
    is_train_ = is_train;
    sample_list_.clear();

    if (!load_class_labels(source)) return false;

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
    return {imagenet100_constants::IMAGE_HEIGHT, imagenet100_constants::IMAGE_WIDTH,
            imagenet100_constants::NUM_CHANNELS};
  }

  int get_num_classes() const override {
    return static_cast<int>(imagenet100_constants::NUM_CLASSES);
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

    Vec<int> label_counts(imagenet100_constants::NUM_CLASSES, 0);
    for (const auto &[path, label] : sample_list_) {
      if (label >= 0 && label < static_cast<int>(imagenet100_constants::NUM_CLASSES)) {
        label_counts[label]++;
      }
    }

    std::cout << "ImageNet-100 Dataset Statistics (NHWC format, lazy-loaded):" << std::endl;
    std::cout << "Total samples: " << sample_list_.size() << std::endl;
    std::cout << "Image shape: " << imagenet100_constants::IMAGE_HEIGHT << "x"
              << imagenet100_constants::IMAGE_WIDTH << "x" << imagenet100_constants::NUM_CHANNELS
              << std::endl;
    std::cout << "Number of classes: " << imagenet100_constants::NUM_CLASSES << std::endl;
    std::cout << "Class distribution (first 10 classes):" << std::endl;
    for (int i = 0; i < std::min(10, static_cast<int>(class_ids_.size())); ++i) {
      std::string display_name = class_ids_[i];
      auto it = class_id_to_name_.find(display_name);
      if (it != class_id_to_name_.end()) display_name = it->second;
      std::cout << "  Class " << i << " (" << class_ids_[i] << " - " << display_name
                << "): " << label_counts[i] << " samples" << std::endl;
    }
  }

  static void create(const std::string &data_path, ImageNet100DataLoader &train_loader,
                     ImageNet100DataLoader &val_loader) {
    if (!train_loader.load_data(data_path, true)) {
      throw std::runtime_error("Failed to index training data!");
    }
    if (!val_loader.load_data(data_path, false)) {
      throw std::runtime_error("Failed to index validation data!");
    }
  }
};
}  // namespace tnn
