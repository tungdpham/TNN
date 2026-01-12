#include "data_loader.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tnn {

template <typename T = float> class OpenWebTextDataLoader : public BaseDataLoader<T> {
public:
  OpenWebTextDataLoader(size_t context_length) : context_length_(context_length) {}

  ~OpenWebTextDataLoader() {
    if (mapped_data_ != MAP_FAILED && mapped_data_ != nullptr) {
      munmap(mapped_data_, file_size_);
    }
    if (fd_ != -1) {
      close(fd_);
    }
  }

  bool load_data(const std::string &source) override {
    fd_ = open(source.c_str(), O_RDONLY);
    if (fd_ == -1) {
      perror("Error opening file");
      return false;
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1)
      return false;
    file_size_ = sb.st_size;

    mapped_data_ = (uint16_t *)mmap(NULL, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_data_ == MAP_FAILED) {
      perror("mmap failed");
      return false;
    }

    total_tokens_ = file_size_ / sizeof(uint16_t);

    if (total_tokens_ <= context_length_ + 1)
      return false;
    num_samples_ = total_tokens_ - context_length_ - 1;

    std::cout << "Total tokens in dataset: " << total_tokens_ << std::endl;

    if (num_samples_ > 0) {
      dist_ = std::uniform_int_distribution<size_t>(0, num_samples_ - 1);
    }

    return true;
  }

  bool get_batch(size_t batch_size, Tensor<T> &batch_data, Tensor<T> &batch_labels) override {
    if (this->current_index_ + batch_size > num_samples_) {
      return false;
    }

    batch_data.resize({batch_size, context_length_});
    batch_labels.resize({batch_size, context_length_});

    for (size_t b = 0; b < batch_size; ++b) {
      size_t start_pos;
      if (shuffled_) {
        start_pos = dist_(this->rng_);
        this->current_index_++;
      } else {
        start_pos = this->current_index_++;
      }

      for (size_t i = 0; i < context_length_; ++i) {
        batch_data(b, i) = static_cast<T>(mapped_data_[start_pos + i]);
        batch_labels(b, i) = static_cast<T>(mapped_data_[start_pos + i + 1]);
      }
    }

    return true;
  }

  bool get_next_batch(Tensor<T> &batch_data, Tensor<T> &batch_labels) override {
    return get_batch(this->batch_size_, batch_data, batch_labels);
  }

  void reset() override { this->current_index_ = 0; }

  void shuffle() override { shuffled_ = true; }

  size_t size() const override { return num_samples_; }

  std::vector<size_t> get_data_shape() const override { return {context_length_}; }

private:
  int fd_ = -1;
  uint16_t *mapped_data_ = nullptr;
  size_t file_size_ = 0;
  size_t total_tokens_ = 0;
  size_t num_samples_ = 0;
  size_t context_length_;

  bool shuffled_ = false;
  std::uniform_int_distribution<size_t> dist_;
};

} // namespace tnn