#pragma once

#include <cstring>
#include <type_traits>
#include <utility>
#include <variant>

#include "common/blob.hpp"

namespace tnn {

template <typename Derived>
class IArchiver;

template <typename T>
struct ExactType {
  template <typename U>
  operator U() const
    requires(std::is_same_v<T, U>);
};

template <typename T, typename Derived>
concept ModifiableArchivable = requires(T& t, IArchiver<Derived>& a) { archive(a, t); };

template <typename T, typename Derived>
concept ConstArchivable = requires(const T& t, IArchiver<Derived>& a) { archive(a, t); };

template <typename T, typename Derived>
concept Archivable = ModifiableArchivable<T, Derived> || ConstArchivable<T, Derived>;

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
// Archivable types should implement archive(Archiver& archiver, const Derived& obj) method outside
// of the class definition for read only archivers and possibly archive(Archiver& archiver, Derived&
// obj) for write archivers.
template <typename Derived>
class IArchiver {
public:
  template <typename... Args>
  Derived& operator()(const Args&... args) {
    (dispatch<true>(args), ...);  // Force true (const) path
    return static_cast<Derived&>(*this);
  }

  template <typename... Args>
  Derived& operator()(Args&... args) {
    (dispatch<false>(args), ...);
    return static_cast<Derived&>(*this);
  }

  template <typename... Args>
  Derived& operator()(Args&&... args) {
    (dispatch_rvalue(std::forward<Args>(args)), ...);
    return static_cast<Derived&>(*this);
  }

private:
  template <typename T>
  void dispatch_rvalue(T&& data) {
    using RawT = std::remove_cvref_t<T>;
    if constexpr (IsBlob<RawT>) {
      constexpr bool IsConstBlob = std::is_const_v<typename RawT::value_type>;
      dispatch<IsConstBlob>(data);
    } else {
      dispatch<std::is_const_v<std::remove_reference_t<T>>>(data);
    }
  }

  // preserves constness of parent.
  template <bool IsConstContext, typename T>
  void dispatch(T& data) {
    auto& self = static_cast<Derived&>(*this);
    using EffectiveT = std::conditional_t<IsConstContext, const T, T>;
    EffectiveT& ref = static_cast<EffectiveT&>(data);

    using RawT = std::remove_cvref_t<T>;

    if constexpr (Archivable<RawT, Derived>) {
      archive(self, ref);
    } else if constexpr (TriviallyArchivable<RawT>) {
      self.archive_impl(&ref, 1, getHost());
    } else if constexpr (IsBlob<RawT>) {
      if constexpr (TriviallyArchivable<typename RawT::value_type>) {
        self.archive_impl(ref.ptr, ref.count, ref.device);
      } else {
        for (size_t i = 0; i < ref.count; ++i) {
          this->dispatch<IsConstContext>(ref.ptr[i]);
        }
      }
    } else {
      static_assert(always_false<RawT>, "Type is not archivable");
    }
  }
};

// common types
template <typename Archiver>
void archive(Archiver& archiver, const std::monostate&) {
  // No data to archive for std::monostate
}

template <typename Archiver>
void archive(Archiver& archiver, std::monostate&) {
  // No data to archive for std::monostate
}

template <typename Archiver>
void archive(Archiver& archiver, const std::string& str) {
  archiver(static_cast<uint64_t>(str.size()));
  archiver(make_blob(str.data(), str.size()));
}

template <typename Archiver>
void archive(Archiver& archiver, std::string& str) {
  uint64_t str_size = str.size();
  archiver(str_size);
  str.resize(str_size);
  archiver(make_blob(str.data(), str.size()));
}

template <typename Archiver>
void archive(Archiver& archiver, const std::vector<size_t>& vec) {
  archiver(static_cast<uint64_t>(vec.size()));
  if (!vec.empty()) {
    archiver(make_blob(vec.data(), vec.size(), getHost()));
  }
}

template <typename Archiver>
void archive(Archiver& archiver, std::vector<size_t>& vec) {
  uint64_t vec_size = vec.size();
  archiver(vec_size);
  vec.resize(vec_size);
  if (!vec.empty()) {
    archiver(make_blob(vec.data(), vec.size(), getHost()));
  }
}

}  // namespace tnn