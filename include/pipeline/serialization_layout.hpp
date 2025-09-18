#pragma once

#include <cstdint>

enum class Endianness { LITTLE, BIG };

constexpr Endianness get_system_endianness() {
  union {
    uint32_t i;
    char c[4];
  } u = {0x01020304};
  return (u.c[0] == 1) ? Endianness::BIG : Endianness::LITTLE;
}