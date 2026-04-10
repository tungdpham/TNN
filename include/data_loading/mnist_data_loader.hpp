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
#include <iostream>
#include <numeric>
#include <string>

#include "data_loading/image_data_loader.hpp"
#include "tensor/tensor.hpp"

namespace mnist_constants {
constexpr size_t IMAGE_HEIGHT = 28;
constexpr size_t IMAGE_WIDTH = 28;
constexpr size_t IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
constexpr size_t NUM_CLASSES = 10;
constexpr size_t NUM_CHANNELS = 1;
constexpr float NORMALIZATION_FACTOR = 255.0f;
}  // namespace mnist_constants

namespace tnn {
/**
 * Enhanced MNIST data loader for CSV format adapted for CNN (2D images)
 * NHWC format: (Batch, Height, Width, Channels)
 *
 * Uses memory-mapped I/O + a line-position index built at load time.
 * Only the line-start offsets (~480 KB for 60 k rows) are kept in RAM;
 * the raw CSV bytes stay on disk and are paged in by the OS on demand.
 */
class MNISTDataLoader : public ImageDataLoader {
private:
  int fd_ = -1;
  const char *mapped_ = nullptr;
  size_t file_size_ = 0;

  // Byte offset of the start of each sample line (after the header)
  Vec<size_t> line_offsets_;
  // Access order — shuffled in-place; current_index_ indexes into this
  Vec<size_t> access_order_;

  DType_t dtype_ = DType_t::FP32;

  void cleanup_map() {
    if (mapped_ && mapped_ != reinterpret_cast<const char *>(MAP_FAILED)) {
      munmap(const_cast<char *>(mapped_), file_size_);
    }
    if (fd_ != -1) close(fd_);
    mapped_ = nullptr;
    fd_ = -1;
    file_size_ = 0;
  }

  /**
   * Parse a single CSV sample line starting at buf[0..len).
   * Fills image_data (IMAGE_SIZE floats) and returns the integer label,
   * or -1 on parse error.
   */
  static int parse_sample_line(const char *buf, size_t len, float *image_data) {
    size_t pos = 0;

    // Parse label
    int label = 0;
    bool got_digit = false;
    while (pos < len && buf[pos] != ',') {
      if (buf[pos] >= '0' && buf[pos] <= '9') {
        label = label * 10 + (buf[pos] - '0');
        got_digit = true;
      }
      ++pos;
    }
    if (!got_digit || pos >= len) return -1;
    ++pos;  // skip comma

    // Parse pixel values
    for (size_t p = 0; p < mnist_constants::IMAGE_SIZE; ++p) {
      int val = 0;
      bool got = false;
      while (pos < len && buf[pos] != ',' && buf[pos] != '\r' && buf[pos] != '\n') {
        if (buf[pos] >= '0' && buf[pos] <= '9') {
          val = val * 10 + (buf[pos] - '0');
          got = true;
        }
        ++pos;
      }
      if (!got) return -1;
      image_data[p] = static_cast<float>(val) / mnist_constants::NORMALIZATION_FACTOR;
      if (pos < len && (buf[pos] == ',' || buf[pos] == '\r')) ++pos;
    }

    return label;
  }

  bool load_data_impl(const std::string &source) {
    cleanup_map();
    line_offsets_.clear();

    fd_ = open(source.c_str(), O_RDONLY);
    if (fd_ == -1) {
      std::cerr << "Error: Could not open file " << source << std::endl;
      return false;
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
      close(fd_);
      fd_ = -1;
      return false;
    }
    file_size_ = static_cast<size_t>(sb.st_size);

    void *ptr = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (ptr == MAP_FAILED) {
      std::cerr << "Error: mmap failed for " << source << std::endl;
      close(fd_);
      fd_ = -1;
      return false;
    }
    mapped_ = static_cast<const char *>(ptr);

    // Skip the header line
    size_t pos = 0;
    while (pos < file_size_ && mapped_[pos] != '\n') ++pos;
    ++pos;  // move past '\n'

    // Record start of each subsequent line
    while (pos < file_size_) {
      if (mapped_[pos] != '\r' && mapped_[pos] != '\n') {
        line_offsets_.push_back(pos);
        while (pos < file_size_ && mapped_[pos] != '\n') ++pos;
      }
      ++pos;
    }

    const size_t n = line_offsets_.size();
    access_order_.resize(n);
    std::iota(access_order_.begin(), access_order_.end(), 0);

    this->current_index_ = 0;
    std::cout << "Lazy-mapped " << n << " samples from " << source << std::endl;
    return n > 0;
  }

  template <typename T>
  bool get_batch_impl(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) {
    if (this->current_index_ >= access_order_.size()) return false;

    const size_t actual_batch_size =
        std::min(batch_size, access_order_.size() - this->current_index_);

    // NHWC format: (Batch, Height, Width, Channels)
    batch_data = make_tensor<T>({actual_batch_size, mnist_constants::IMAGE_HEIGHT,
                                 mnist_constants::IMAGE_WIDTH, mnist_constants::NUM_CHANNELS});
    batch_labels = make_tensor<T>({actual_batch_size, mnist_constants::NUM_CLASSES});
    batch_labels->fill(0.0);

    for (size_t i = 0; i < actual_batch_size; ++i) {
      const size_t sample_idx = access_order_[this->current_index_ + i];
      const size_t line_start = line_offsets_[sample_idx];

      // Compute line length (up to next newline or end-of-file)
      size_t line_end = line_start;
      while (line_end < file_size_ && mapped_[line_end] != '\n') ++line_end;

      float pixel_buf[mnist_constants::IMAGE_SIZE];
      const int label = parse_sample_line(mapped_ + line_start, line_end - line_start, pixel_buf);

      if (label < 0) continue;

      // Store in NHWC (H, W, C=1) order
      for (size_t j = 0; j < mnist_constants::IMAGE_SIZE; ++j) {
        batch_data->at<T>({i, j / mnist_constants::IMAGE_WIDTH, j % mnist_constants::IMAGE_WIDTH,
                           0}) = static_cast<T>(pixel_buf[j]);
      }

      if (label >= 0 && label < static_cast<int>(mnist_constants::NUM_CLASSES)) {
        batch_labels->at<T>({i, static_cast<size_t>(label)}) = static_cast<T>(1.0);
      }
    }

    this->apply_augmentation(batch_data, batch_labels);
    this->current_index_ += actual_batch_size;
    return true;
  }

public:
  explicit MNISTDataLoader(DType_t dtype = DType_t::FP32)
      : ImageDataLoader(),
        dtype_(dtype) {}

  virtual ~MNISTDataLoader() { cleanup_map(); }

  /**
   * Lazy-map MNIST data from a CSV file.
   * Reads only the line-start index into RAM; pixel bytes stay on disk.
   */
  bool load_data(const std::string &source) override { return load_data_impl(source); }

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

  size_t size() const override { return line_offsets_.size(); }

  Vec<size_t> get_data_shape() const override {
    return {mnist_constants::IMAGE_HEIGHT, mnist_constants::IMAGE_WIDTH,
            mnist_constants::NUM_CHANNELS};
  }

  int get_num_classes() const override { return static_cast<int>(mnist_constants::NUM_CLASSES); }

  Vec<std::string> get_class_names() const override {
    return {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
  }

  void print_data_stats() const override {
    if (line_offsets_.empty()) {
      std::cout << "No data loaded" << std::endl;
      return;
    }
    std::cout << "MNIST Dataset Statistics (NHWC format, lazy-mapped):" << std::endl;
    std::cout << "Total samples: " << line_offsets_.size() << std::endl;
    std::cout << "Image shape: " << mnist_constants::IMAGE_HEIGHT << "x"
              << mnist_constants::IMAGE_WIDTH << "x" << mnist_constants::NUM_CHANNELS << std::endl;
  }

  static void create(const std::string &data_path, MNISTDataLoader &train_loader,
                     MNISTDataLoader &test_loader) {
    if (!train_loader.load_data(data_path + "/mnist_train.csv")) {
      throw std::runtime_error("Failed to load training data!");
    }
    if (!test_loader.load_data(data_path + "/mnist_test.csv")) {
      throw std::runtime_error("Failed to load test data!");
    }
  }
};
}  // namespace tnn
