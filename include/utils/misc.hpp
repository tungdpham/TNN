#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace utils {

template <typename EnumType> constexpr std::vector<EnumType> get_enum_vector() {
  static_assert(std::is_enum_v<EnumType>, "Template parameter must be an enum type");
  static_assert(std::is_same_v<decltype(EnumType::_COUNT), EnumType>,
                "Enum type must have a _COUNT member to indicate the number of "
                "enum values");
  static_assert(std::is_same_v<decltype(EnumType::_START), EnumType>,
                "Enum type must have a _START member to indicate the starting "
                "enum value");
  std::vector<EnumType> values;
  for (int i = static_cast<int>(EnumType::_START); i < static_cast<int>(EnumType::_COUNT); ++i) {
    values.push_back(static_cast<EnumType>(i));
  }
  return values;
}

} // namespace utils