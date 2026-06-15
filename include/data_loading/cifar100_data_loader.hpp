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
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

#include "data_loading/image_data_loader.hpp"
#include "tensor/tensor.hpp"

namespace cifar100_constants {
constexpr size_t IMAGE_HEIGHT = 32;
constexpr size_t IMAGE_WIDTH = 32;
constexpr size_t IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3;
constexpr size_t NUM_CLASSES = 100;
constexpr size_t NUM_COARSE_CLASSES = 20;
constexpr size_t NUM_CHANNELS = 3;
constexpr float NORMALIZATION_FACTOR = 255.0f;
constexpr size_t RECORD_SIZE = 1 + 1 + IMAGE_SIZE;
}  // namespace cifar100_constants

namespace synet {
/**
 * Enhanced CIFAR-100 data loader for binary format adapted for CNN (2D RGB images)
 * NHWC format: (Batch, Height, Width, Channels)
 * Uses memory-mapped I/O for lazy loading — only pages that are actually
 * accessed are brought into RAM, keeping the full dataset off-heap.
 */
class CIFAR100DataLoader : public ImageDataLoader {
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
  Vec<std::string> fine_class_names_;
  Vec<std::string> coarse_class_names_;
  bool use_coarse_labels_;
  IAllocator &allocator_;
  DType_t dtype_ = DType_t::FP32;

  static Vec<std::string> load_names_from_file(const std::string &path) {
    Vec<std::string> names;
    std::ifstream f(path);
    if (!f.is_open()) return names;
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty()) names.push_back(line);
    }
    return names;
  }

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
    mf.num_records = mf.size / cifar100_constants::RECORD_SIZE;

    size_t file_idx = mapped_files_.size();
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

    size_t n = sample_map_.size();
    access_order_.resize(n);
    std::iota(access_order_.begin(), access_order_.end(), 0);

    this->current_index_ = 0;
    std::cout << "Total lazy-mapped: " << n << " samples" << std::endl;
    std::cout << "Using " << (use_coarse_labels_ ? "coarse" : "fine") << " labels" << std::endl;
    return n > 0;
  }

  template <typename T>
  bool get_batch_impl(size_t batch_size, Tensor &batch_data, Tensor &batch_labels) {
    if (this->current_index_ >= access_order_.size()) return false;

    size_t actual_batch_size = std::min(batch_size, access_order_.size() - this->current_index_);

    batch_data = Tensor({actual_batch_size, cifar100_constants::IMAGE_HEIGHT,
                         cifar100_constants::IMAGE_WIDTH, cifar100_constants::NUM_CHANNELS},
                        dtype_of<T>());

    batch_labels = Tensor({actual_batch_size}, DType_t::INT32_T);

    for (size_t i = 0; i < actual_batch_size; ++i) {
      size_t sample_idx = access_order_[this->current_index_ + i];
      const auto &[file_idx, record_idx] = sample_map_[sample_idx];
      const uint8_t *record =
          mapped_files_[file_idx].data + record_idx * cifar100_constants::RECORD_SIZE;

      size_t label =
          use_coarse_labels_ ? static_cast<size_t>(record[0]) : static_cast<size_t>(record[1]);
      const uint8_t *pixels = record + 2;

      for (size_t c = 0; c < cifar100_constants::NUM_CHANNELS; ++c) {
        for (size_t h = 0; h < cifar100_constants::IMAGE_HEIGHT; ++h) {
          for (size_t w = 0; w < cifar100_constants::IMAGE_WIDTH; ++w) {
            size_t src_idx =
                c * cifar100_constants::IMAGE_HEIGHT * cifar100_constants::IMAGE_WIDTH +
                h * cifar100_constants::IMAGE_WIDTH + w;
            batch_data.at<T>({i, h, w, c}) =
                static_cast<T>(pixels[src_idx] / cifar100_constants::NORMALIZATION_FACTOR);
          }
        }
      }

      batch_labels.at<int>({i}) = static_cast<int>(label);
    }

    this->apply_augmentation(batch_data, batch_labels);
    this->current_index_ += actual_batch_size;
    return true;
  }

public:
  explicit CIFAR100DataLoader(bool use_coarse_labels = false, DType_t dtype = DType_t::FP32)
      : ImageDataLoader(),
        use_coarse_labels_(use_coarse_labels),
        allocator_(PoolAllocator::instance(getHost(), defaultFlowHandle)),
        dtype_(dtype) {}

  virtual ~CIFAR100DataLoader() { cleanup_maps(); }

  /**
   * Load CIFAR-100 data from a single binary file.
   */
  bool load_data(const std::string &source) override {
    if (source.find(".bin") != std::string::npos) {
      return load_multiple_files({source});
    }
    std::cerr << "Error: For multiple files, use load_multiple_files() method" << std::endl;
    return false;
  }

  /**
   * Lazy-map CIFAR-100 data from multiple binary files.
   */
  bool load_multiple_files(const Vec<std::string> &filenames) { return load_files_impl(filenames); }

  void set_label_files(const std::string &fine_path, const std::string &coarse_path) {
    fine_class_names_ = load_names_from_file(fine_path);
    if (fine_class_names_.empty())
      std::cerr << "Warning: could not read fine class names from " << fine_path << std::endl;
    coarse_class_names_ = load_names_from_file(coarse_path);
    if (coarse_class_names_.empty())
      std::cerr << "Warning: could not read coarse class names from " << coarse_path << std::endl;
  }

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
    return {cifar100_constants::IMAGE_HEIGHT, cifar100_constants::IMAGE_WIDTH,
            cifar100_constants::NUM_CHANNELS};
  }

  int get_num_classes() const override {
    return use_coarse_labels_ ? static_cast<int>(cifar100_constants::NUM_COARSE_CLASSES)
                              : static_cast<int>(cifar100_constants::NUM_CLASSES);
  }

  Vec<std::string> get_class_names() const override {
    return use_coarse_labels_ ? coarse_class_names_ : fine_class_names_;
  }

  void set_use_coarse_labels(bool use_coarse) { use_coarse_labels_ = use_coarse; }

  void print_data_stats() const override {
    size_t n = sample_map_.size();
    if (n == 0) {
      std::cout << "No data loaded" << std::endl;
      return;
    }

    size_t num_classes = use_coarse_labels_ ? cifar100_constants::NUM_COARSE_CLASSES
                                            : cifar100_constants::NUM_CLASSES;
    Vec<int> label_counts(num_classes, 0);
    for (size_t i = 0; i < n; ++i) {
      const auto &[fi, ri] = sample_map_[i];
      const uint8_t *record = mapped_files_[fi].data + ri * cifar100_constants::RECORD_SIZE;
      const int label =
          use_coarse_labels_ ? static_cast<int>(record[0]) : static_cast<int>(record[1]);
      if (label >= 0 && label < static_cast<int>(num_classes)) label_counts[label]++;
    }

    std::cout << "CIFAR-100 Dataset Statistics (NHWC format, lazy-mapped):" << std::endl;
    std::cout << "Total samples: " << n << std::endl;
    std::cout << "Using " << (use_coarse_labels_ ? "coarse" : "fine") << " labels" << std::endl;
    std::cout << "Image shape: " << cifar100_constants::IMAGE_HEIGHT << "x"
              << cifar100_constants::IMAGE_WIDTH << "x" << cifar100_constants::NUM_CHANNELS
              << std::endl;
    std::cout << "Number of classes: " << num_classes << std::endl;
  }

  static void create(const std::string &data_path, CIFAR100DataLoader &train_loader,
                     CIFAR100DataLoader &test_loader) {
    const std::string fine_labels = data_path + "/cifar-100-binary/fine_label_names.txt";
    const std::string coarse_labels = data_path + "/cifar-100-binary/coarse_label_names.txt";
    train_loader.set_label_files(fine_labels, coarse_labels);
    test_loader.set_label_files(fine_labels, coarse_labels);

    if (!train_loader.load_data(data_path + "/cifar-100-binary/train.bin")) {
      throw std::runtime_error("Failed to load training data!");
    }

    if (!test_loader.load_data(data_path + "/cifar-100-binary/test.bin")) {
      throw std::runtime_error("Failed to load test data!");
    }
  }
};
}  // namespace synet
