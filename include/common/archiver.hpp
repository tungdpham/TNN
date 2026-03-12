#pragma once

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "common/blob.hpp"
#include "device/dptr.hpp"

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

// Derived archiver class should implement archive_impl(const T* data, size_t count)
// Optionally implement archive_dptr_impl(dptr& data) for dptr type if supported.
template <typename Derived>
class IArchiver {
public:
  template <typename T>
  Derived& operator&(T& data) {
    process(data);
    return static_cast<Derived&>(*this);
  }

  template <typename T>
  Derived& operator&(const T& data) {
    process(data);
    return static_cast<Derived&>(*this);
  }

  template <typename T>
  Derived& operator&(T&& data) {
    process(data);
    return static_cast<Derived&>(*this);
  }

private:
  template <typename T>
  void process(T& data) {
    auto& self = static_cast<Derived&>(*this);
    using RawT = std::remove_cv_t<T>;
    if constexpr (Archivable<RawT, Derived>) {
      data.archive(self);
    } else if constexpr (TriviallyArchivable<RawT>) {
      self.archive_impl(&data, 1);
    } else if constexpr (IsBlob<RawT>) {
      self.archive_impl(&data.count, 1);
      self.archive_impl(data.ptr, data.count);
    } else if constexpr (std::is_same_v<RawT, dptr>) {
      self.archive_dptr_impl(data);
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
  void archive_impl(const T* data, size_t count) {
    size_ += sizeof(T) * count;
  }

  void archive_dptr_impl(dptr& data) {
    size_ += sizeof(uint64_t);  // for storing the size
    size_ += data.capacity();
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
  void archive_impl(const T* data, size_t count) {
    std::memcpy(buffer_ + offset_, data, sizeof(T) * count);
    offset_ += sizeof(T) * count;
  }

  void archive_dptr_impl(dptr& data) {
    uint64_t byte_size = data.capacity();
    archive_impl(&byte_size, 1);
    data.copy_to_host(buffer_ + offset_, byte_size);
    offset_ += data.capacity();
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
  void archive_impl(T* data, size_t count) {
    std::memcpy(data, buffer_ + offset_, sizeof(T) * count);
    offset_ += sizeof(T) * count;
  }

  void archive_dptr_impl(dptr& data) {
    uint64_t byte_size;
    archive_impl(&byte_size, 1);
    data.copy_from_host(buffer_ + offset_, byte_size);
    offset_ += byte_size;
  }

  size_t bytes_read() const { return offset_; }
};

}  // namespace tnn