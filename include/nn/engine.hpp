#pragma once

#include <fmt/core.h>

#include <memory>

#include "nn/engines/iengine.hpp"

namespace tunx {

using Engine = std::shared_ptr<IEngine>;

template <typename Type, typename... Args>
inline Engine make_engine(Args... args) {
  return std::make_shared<Type>(args...);
}

}  // namespace tunx