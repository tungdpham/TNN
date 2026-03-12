#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace tnn {

struct bf16 {
  uint16_t data;

  bf16()
      : data(0) {}
  explicit bf16(uint16_t d)
      : data(d) {}

  // Constructors from other types
  bf16(float f)
      : data(float_to_bf16(f)) {}
  explicit bf16(double d)
      : data(float_to_bf16(static_cast<float>(d))) {}
  explicit bf16(int i)
      : data(float_to_bf16(static_cast<float>(i))) {}
  explicit bf16(size_t s)
      : data(float_to_bf16(static_cast<float>(s))) {}

  // Conversion operators
  operator float() const { return bf16_to_float(data); }

  // Conversion to uint16_t for raw bit access
  explicit operator uint16_t() const { return data; }

  // Explicit conversion to size_t for indexing operations
  explicit operator size_t() const { return static_cast<size_t>(bf16_to_float(data)); }

  static uint16_t float_to_bf16(float f) {
    if (std::isnan(f)) {
      return 0x7FC0;
    }
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    // Round to nearest even
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
  }

  static float bf16_to_float(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
  }
};

// Comparison operators
inline bool operator==(const bf16 &a, const bf16 &b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

inline bool operator!=(const bf16 &a, const bf16 &b) { return !(a == b); }

inline bool operator<(const bf16 &a, const bf16 &b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator>(const bf16 &a, const bf16 &b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator<=(const bf16 &a, const bf16 &b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator>=(const bf16 &a, const bf16 &b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

// Arithmetic operators
inline bf16 operator+(const bf16 &a, const bf16 &b) {
  return bf16(static_cast<float>(a) + static_cast<float>(b));
}

inline bf16 operator-(const bf16 &a, const bf16 &b) {
  return bf16(static_cast<float>(a) - static_cast<float>(b));
}

inline bf16 operator*(const bf16 &a, const bf16 &b) {
  return bf16(static_cast<float>(a) * static_cast<float>(b));
}

inline bf16 operator/(const bf16 &a, const bf16 &b) {
  return bf16(static_cast<float>(a) / static_cast<float>(b));
}

// Compound assignment operators
inline bf16 &operator+=(bf16 &a, const bf16 &b) {
  a = a + b;
  return a;
}

inline bf16 &operator-=(bf16 &a, const bf16 &b) {
  a = a - b;
  return a;
}

inline bf16 &operator*=(bf16 &a, const bf16 &b) {
  a = a * b;
  return a;
}

inline bf16 &operator/=(bf16 &a, const bf16 &b) {
  a = a / b;
  return a;
}

// Unary operators
inline bf16 operator-(const bf16 &a) { return bf16(-static_cast<float>(a)); }

inline bf16 operator+(const bf16 &a) { return a; }

}  // namespace tnn
