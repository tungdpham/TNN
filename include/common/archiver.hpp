#pragma once

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "common/blob.hpp"

namespace tnn {

template <typename Derived>
class IArchiver;

template <typename T, typename Derived>
concept Archivable = requires(T t, IArchiver<Derived>& archiver) { t.archive(archiver); };

template <typename T>
concept TriviallyArchivable = (std::is_fundamental_v<T> || std::is_enum_v<T>) &&
                              !std::is_pointer_v<T>;  // add more primitive types if needed

template <typename T>
struct is_blob : std::false_type {};

template <typename T>
struct is_blob<Blob<T>> : std::true_type {};

template <typename T>
concept IsBlob = is_blob<std::remove_cvref_t<T>>::value;

template <typename T>
concept always_false = false;

// Derived class should implement archive_impl(const T* data, size_t count, const Device& device)
template <typename Derived>
class IArchiver {
public:
  template <typename... Args>
  Derived& operator()(Args&&... args) {
    (process(args), ...);
    return static_cast<Derived&>(*this);
  }

private:
  template <typename T>
  inline void process(T& data) {
    auto& self = static_cast<Derived&>(*this);
    using RawT = std::remove_cv_t<T>;
    if constexpr (Archivable<RawT, Derived>) {
      data.archive(self);
    } else if constexpr (TriviallyArchivable<RawT>) {
      self.archive_impl(&data, 1, getHost());
    } else if constexpr (IsBlob<RawT>) {
      self.archive_impl(&data.count, 1, getHost());
      self.archive_impl(data.ptr, data.count, data.device);
    } else {
      static_assert(always_false<RawT>, "Type is not archivable");
    }
  }
};

// Examples of concrete archivers
class SizeArchiver : public IArchiver<SizeArchiver> {
private:
  size_t size_ = 0;

public:
  template <typename T>
  void archive_impl(const T* data, size_t count, const Device& device) {
    size_ += sizeof(T) * count;
  }

  size_t size() const { return size_; }
};

class OutArchiver : public IArchiver<OutArchiver> {
private:
  char* buffer_;
  size_t offset_ = 0;

public:
  OutArchiver(char* buffer, size_t size)
      : buffer_(buffer) {}

  template <typename T>
  void archive_impl(const T* data, size_t count, const Device& device) {
    device.copyToHost(buffer_ + offset_, data, sizeof(T) * count);
    offset_ += sizeof(T) * count;
  }

  size_t bytes_written() const { return offset_; }
};

class InArchiver : public IArchiver<InArchiver> {
private:
  const char* buffer_;
  size_t offset_ = 0;

public:
  InArchiver(const char* buffer, size_t size)
      : buffer_(buffer) {}

  template <typename T>
  void archive_impl(T* data, size_t count, const Device& device) {
    device.copyToDevice(data, buffer_ + offset_, sizeof(T) * count);
    offset_ += sizeof(T) * count;
  }

  size_t bytes_read() const { return offset_; }
};

}  // namespace tnn