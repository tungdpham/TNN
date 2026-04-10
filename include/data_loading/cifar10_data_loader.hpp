/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

#include "data_loading/image_data_loader.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_handler.hpp"

namespace cifar10_constants {
constexpr size_t IMAGE_HEIGHT = 32;
constexpr size_t IMAGE_WIDTH = 32;
constexpr size_t IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3;
constexpr size_t NUM_CLASSES = 10;
constexpr size_t NUM_CHANNELS = 3;
constexpr float NORMALIZATION_FACTOR = 255.0f;
constexpr size_t RECORD_SIZE = 1 + IMAGE_SIZE;
}  // namespace cifar10_constants

namespace tnn {
/**
 *  CIFAR-10 data loader for binary format adapted for CNN (2D RGB images)
 *  NHWC format: (Batch, Height, Width, Channels)
 *  Uses memory-mapped I/O for lazy loading — only pages that are actually
 *  accessed are brought into RAM, keeping the full dataset off-heap.
 */
class CIFAR10DataLoader : public ImageDataLoader {
private:
  struct MappedFile {
    int fd = -1;
    const uint8_t *data = nullptr;
    size_t size = 0;
    size_t num_records = 0;

    void unmap() {
      if (data && data != reinterpret_cast<const uint8_t *>(MAP_FAILED)) {
        munmap(const_cast<uint8_t *>(data), size);
      }
      if (fd != -1) close(fd);
      data = nullptr;
      fd = -1;
    }
  };

  Vec<MappedFile> mapped_files_;
  // sample_map_[i] = (file_idx, record_idx within that file)
  Vec<std::pair<size_t, size_t>> sample_map_;
  // Access order — shuffled in-place; current_index_ indexes into this
  Vec<size_t> access_order_;

  DType_t dtype_ = DType_t::FP32;

  Vec<std::string> class_names_ = {"airplane", "automobile", "bird",  "cat",  "deer",
                                   "dog",      "frog",       "horse", "ship", "truck"};

  void cleanup_maps() {
    for (auto &mf : mapped_files_) mf.unmap();
    mapped_files_.clear();
  }

  bool map_file(const std::string &filename) {
    MappedFile mf;
    mf.fd = open(filename.c_str(), O_RDONLY);
    if (mf.fd == -1) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      return false;
    }

    struct stat sb;
    if (fstat(mf.fd, &sb) == -1) {
      close(mf.fd);
      return false;
    }
    mf.size = static_cast<size_t>(sb.st_size);

    void *ptr = mmap(nullptr, mf.size, PROT_READ, MAP_PRIVATE, mf.fd, 0);
    if (ptr == MAP_FAILED) {
      std::cerr << "Error: mmap failed for " << filename << std::endl;
      close(mf.fd);
      return false;
    }
    mf.data = static_cast<const uint8_t *>(ptr);
    mf.num_records = mf.size / cifar10_constants::RECORD_SIZE;

    const size_t file_idx = mapped_files_.size();
    for (size_t r = 0; r < mf.num_records; ++r) {
      sample_map_.emplace_back(file_idx, r);
    }

    mapped_files_.push_back(mf);
    std::cout << "Mapped " << mf.num_records << " samples from " << filename << std::endl;
    return true;
  }

  bool load_files_impl(const Vec<std::string> &filenames) {
    cleanup_maps();
    sample_map_.clear();

    for (const auto &fn : filenames) {
      if (!map_file(fn)) return false;
    }

    const size_t n = sample_map_.size();
    access_order_.resize(n);
    std::iota(access_order_.begin(), access_order_.end(), 0);

    this->current_index_ = 0;
    std::cout << "Total lazy-mapped: " << n << " samples" << std::endl;
    return n > 0;
  }

  template <typename T>
  bool get_batch_impl(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) {
    if (this->current_index_ >= access_order_.size()) return false;

    const size_t actual_batch_size =
        std::min(batch_size, access_order_.size() - this->current_index_);
    const size_t height = cifar10_constants::IMAGE_HEIGHT;
    const size_t width = cifar10_constants::IMAGE_WIDTH;
    const size_t channels = cifar10_constants::NUM_CHANNELS;
    const size_t num_classes = cifar10_constants::NUM_CLASSES;

    batch_data = make_tensor<T>({actual_batch_size, height, width, channels});
    batch_labels = make_tensor<T>({actual_batch_size, num_classes, 1, 1});
    batch_labels->fill(0.0);

    T *data_ptr = batch_data->data_as<T>();
    T *labels_ptr = batch_labels->data_as<T>();

    parallel_for<size_t>(0, actual_batch_size, [&](size_t i) {
      const size_t sample_idx = access_order_[this->current_index_ + i];
      const auto &[file_idx, record_idx] = sample_map_[sample_idx];
      const uint8_t *record =
          mapped_files_[file_idx].data + record_idx * cifar10_constants::RECORD_SIZE;

      const int label = static_cast<int>(record[0]);
      const uint8_t *pixels = record + 1;

      // Source: CHW uint8, Destination: NHWC float
      for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            const size_t src_idx = c * (height * width) + h * width + w;
            const size_t dst_idx =
                i * height * width * channels + h * width * channels + w * channels + c;
            data_ptr[dst_idx] =
                static_cast<T>(pixels[src_idx] / cifar10_constants::NORMALIZATION_FACTOR);
          }
        }
      }

      if (label >= 0 && label < static_cast<int>(num_classes)) {
        labels_ptr[i * num_classes + label] = static_cast<T>(1.0);
      }
    });

    this->apply_augmentation(batch_data, batch_labels);
    this->current_index_ += actual_batch_size;
    return true;
  }

public:
  explicit CIFAR10DataLoader(DType_t dtype = DType_t::FP32)
      : ImageDataLoader(),
        dtype_(dtype) {}

  virtual ~CIFAR10DataLoader() { cleanup_maps(); }

  /**
   * Load CIFAR-10 data from a single binary file
   */
  bool load_data(const std::string &source) override {
    if (source.find(".bin") != std::string::npos) {
      return load_multiple_files({source});
    }
    std::cerr << "Error: For multiple files, use load_multiple_files() method" << std::endl;
    return false;
  }

  /**
   * Lazy-map CIFAR-10 data from multiple binary files.
   * The OS will page in only the records actually accessed during get_batch.
   */
  bool load_multiple_files(const Vec<std::string> &filenames) { return load_files_impl(filenames); }

  bool get_batch(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) override {
    DISPATCH_DTYPE(dtype_, T, return get_batch_impl<T>(batch_size, batch_data, batch_labels));
  }

  void reset() override { this->current_index_ = 0; }

  /**
   * Shuffle the access order without copying any pixel data.
   */
  void shuffle() override {
    if (access_order_.empty()) return;
    access_order_ = this->generate_shuffled_indices(access_order_.size());
    this->current_index_ = 0;
  }

  size_t size() const override { return sample_map_.size(); }

  Vec<size_t> get_data_shape() const override {
    return {cifar10_constants::IMAGE_HEIGHT, cifar10_constants::IMAGE_WIDTH,
            cifar10_constants::NUM_CHANNELS};
  }

  int get_num_classes() const override { return static_cast<int>(cifar10_constants::NUM_CLASSES); }

  Vec<std::string> get_class_names() const override { return class_names_; }

  void print_data_stats() const override {
    const size_t n = sample_map_.size();
    if (n == 0) {
      std::cout << "No data loaded" << std::endl;
      return;
    }

    Vec<int> label_counts(cifar10_constants::NUM_CLASSES, 0);
    for (size_t i = 0; i < n; ++i) {
      const auto &[fi, ri] = sample_map_[i];
      const int label =
          static_cast<int>(mapped_files_[fi].data[ri * cifar10_constants::RECORD_SIZE]);
      if (label >= 0 && label < static_cast<int>(cifar10_constants::NUM_CLASSES)) {
        label_counts[label]++;
      }
    }

    std::cout << "CIFAR-10 Dataset Statistics (NHWC format, lazy-mapped):" << std::endl;
    std::cout << "Total samples: " << n << std::endl;
    std::cout << "Image shape: " << cifar10_constants::IMAGE_HEIGHT << "x"
              << cifar10_constants::IMAGE_WIDTH << "x" << cifar10_constants::NUM_CHANNELS
              << std::endl;
    std::cout << "Class distribution:" << std::endl;
    for (int i = 0; i < static_cast<int>(cifar10_constants::NUM_CLASSES); ++i) {
      std::cout << "  " << class_names_[i] << " (" << i << "): " << label_counts[i] << " samples"
                << std::endl;
    }
  }

  static void create(const std::string &data_path, CIFAR10DataLoader &train_loader,
                     CIFAR10DataLoader &test_loader) {
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
};
}  // namespace tnn
