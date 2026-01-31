#pragma once

namespace tnn {
// strong ref (pointer that cannot be null)
template <typename T>
class sref {
private:
  T* ptr;

public:
  sref(T& val) noexcept : ptr(&val) {}

  operator T&() const noexcept { return *ptr; }

  T* operator->() const noexcept { return ptr; }
  T& operator*() const noexcept { return *ptr; }
};

// strong const ref (const pointer that cannot be null)
template <typename T>
class csref {
private:
  const T* ptr;

public:
  csref(const T& val) noexcept : ptr(&val) {}

  operator const T&() const noexcept { return *ptr; }

  const T* operator->() const noexcept { return ptr; }
  const T& operator*() const noexcept { return *ptr; }
};
}  // namespace tnn