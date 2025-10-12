#pragma once

#include <cstdint>

enum Endianness : uint8_t { LITTLE = 0, BIG = 1 };

constexpr Endianness get_system_endianness() {
  union {
    uint32_t i;
    char c[4];
  } u = {0x01020304};
  return (u.c[0] == 1) ? Endianness::BIG : Endianness::LITTLE;
}

template <typename T> void bswap(T &value) {
  if constexpr (sizeof(T) == 2) {
    value = __builtin_bswap16(value);
  } else if constexpr (sizeof(T) == 4) {
    value = __builtin_bswap32(value);
  } else if constexpr (sizeof(T) == 8) {
    value = __builtin_bswap64(value);
  }
}