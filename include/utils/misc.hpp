#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#elif defined(USE_TBB)
#include <memory>
#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/task_arena.h>
#endif

namespace utils {

#if defined(USE_TBB)
inline std::unique_ptr<tbb::global_control> g_tbb_control;
#endif

template <typename EnumType> constexpr std::vector<EnumType> get_enum_vector() {
  static_assert(std::is_enum_v<EnumType>,
                "Template parameter must be an enum type");
  static_assert(std::is_same_v<decltype(EnumType::_COUNT), EnumType>,
                "Enum type must have a _COUNT member to indicate the number of "
                "enum values");
  static_assert(std::is_same_v<decltype(EnumType::_START), EnumType>,
                "Enum type must have a _START member to indicate the starting "
                "enum value");
  std::vector<EnumType> values;
  for (int i = static_cast<int>(EnumType::_START);
       i < static_cast<int>(EnumType::_COUNT); ++i) {
    values.push_back(static_cast<EnumType>(i));
  }
  return values;
}

void set_num_threads(int num_threads, bool cap_cpu = true) {
  if (num_threads < 1) {
    throw std::invalid_argument("Number of threads must be at least 1");
  }
  int max_threads = 1000;
#if defined(USE_TBB)
  if (cap_cpu) {
    max_threads = tbb::info::default_concurrency();
  }
  num_threads = std::min(num_threads, max_threads);
  g_tbb_control.reset();
  g_tbb_control = std::make_unique<tbb::global_control>(
      tbb::global_control::max_allowed_parallelism, num_threads);

  std::cout << "Using up to " << num_threads << " TBB threads" << std::endl;
#elif defined(_OPENMP)
  if (cap_cpu) {
    max_threads = omp_get_max_threads();
  }
  num_threads = std::min(num_threads, max_threads);
  omp_set_num_threads(num_threads);
  std::cout << "Using up to " << num_threads << " OpenMP threads" << std::endl;
#endif
}

} // namespace utils