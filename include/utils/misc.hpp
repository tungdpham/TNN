#pragma once

#include <algorithm>
#include <chrono>
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

template <typename Func> void benchmark(const std::string &name, Func &&func, int bench_runs = 3) {
  std::cout << "Benchmarking: " << name << std::endl;
  std::vector<double> times;
  for (int i = 0; i < bench_runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    times.push_back(duration.count());
  }

  double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  std::cout << name << " average time: " << avg << " ms" << std::endl;
}

} // namespace utils