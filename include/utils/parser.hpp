#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace tnn {

template <typename T>
T from_str(const std::string &str) {
  if constexpr (std::is_same_v<T, std::string>) {
    return str;
  } else if constexpr (std::is_arithmetic_v<T>) {
    std::istringstream iss(str);
    T value;
    if (iss >> value && iss.eof()) {
      return value;
    }
    throw std::invalid_argument("Cannot convert '" + str + "' to numeric type");
  } else {
    static_assert(std::is_same_v<T, std::string> || std::is_arithmetic_v<T>,
                  "get_env only supports string and arithmetic types");
    return T{};
  }
}

}  // namespace tnn