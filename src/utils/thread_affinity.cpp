/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "utils/thread_affinity.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <unistd.h>

namespace utils {

bool ThreadAffinity::set_current_thread_affinity(const AffinityConfig &config) const {
#ifdef __linux__
  return set_thread_affinity(pthread_self(), config);
#else
  std::cout << "Thread affinity not implemented for this platform" << std::endl;
  return false;
#endif
}

bool ThreadAffinity::set_thread_affinity(pthread_t thread_id, const AffinityConfig &config) const {
#ifdef __linux__
  std::vector<int> cpu_ids = get_recommended_cores(config);
  if (cpu_ids.empty()) {
    std::cout << "Warning: No suitable cores found for affinity configuration" << std::endl;
    return false;
  }

  return set_affinity_impl(thread_id, cpu_ids);
#else
  std::cout << "Thread affinity not implemented for this platform" << std::endl;
  return false;
#endif
}

bool ThreadAffinity::set_thread_affinity(std::thread &thread, const AffinityConfig &config) const {
#ifdef __linux__
  if (!thread.joinable()) {
    std::cout << "Warning: Cannot set affinity for non-joinable thread" << std::endl;
    return false;
  }

  return set_thread_affinity(thread.native_handle(), config);
#else
  std::cout << "Thread affinity not implemented for this platform" << std::endl;
  return false;
#endif
}

std::vector<int> ThreadAffinity::get_recommended_cores(const AffinityConfig &config) const {
  switch (config.core_type) {
  case CoreType::EFFICIENCY_CORES:
    return get_efficiency_core_ids(config.max_threads, config.numa_node);
  case CoreType::PERFORMANCE_CORES:
    return get_performance_core_ids(config.max_threads, config.numa_node);
  case CoreType::ALL_CORES: {
    std::vector<int> all_cores;
    for (int i = 0; i < hw_info_.get_logical_cores(); ++i) {
      all_cores.push_back(i);
    }
    if (config.max_threads > 0 && config.max_threads < static_cast<int>(all_cores.size())) {
      all_cores.resize(config.max_threads);
    }
    return all_cores;
  }
  case CoreType::AUTO:
    // Default to E-cores if available, otherwise P-cores
    if (has_efficiency_cores()) {
      return get_efficiency_core_ids(config.max_threads, config.numa_node);
    } else {
      return get_performance_core_ids(config.max_threads, config.numa_node);
    }
  default:
    return {};
  }
}

std::vector<int> ThreadAffinity::get_efficiency_core_ids(int max_cores, int numa_node) const {
  std::vector<int> ecore_ids;
  const auto &cores = hw_info_.get_cores();

  for (size_t i = 0; i < cores.size(); ++i) {
    const auto &core = cores[i];

    // Check if this is an E-core
    if (!core.is_performance_core) {
      // Check NUMA node constraint
      if (numa_node >= 0) {
        auto numa_cores = hw_info_.get_numa_aware_cores();
        bool core_in_numa_node = false;
        if (numa_cores.find(numa_node) != numa_cores.end()) {
          const auto &node_cores = numa_cores.at(numa_node);
          core_in_numa_node = std::find(node_cores.begin(), node_cores.end(),
                                        static_cast<int>(i)) != node_cores.end();
        }
        if (!core_in_numa_node) {
          continue;
        }
      }

      ecore_ids.push_back(static_cast<int>(i));

      // Check if we've reached the maximum requested cores
      if (max_cores > 0 && static_cast<int>(ecore_ids.size()) >= max_cores) {
        break;
      }
    }
  }

  if (ecore_ids.empty()) {
    std::cout << "Warning: No E-cores found, system may not have hybrid architecture" << std::endl;
  } else {
    std::cout << "Found " << ecore_ids.size() << " E-core(s): ";
    for (size_t i = 0; i < ecore_ids.size(); ++i) {
      std::cout << ecore_ids[i];
      if (i < ecore_ids.size() - 1)
        std::cout << ", ";
    }
    std::cout << std::endl;
  }

  return ecore_ids;
}

std::vector<int> ThreadAffinity::get_performance_core_ids(int max_cores, int numa_node) const {
  std::vector<int> pcore_ids;
  const auto &cores = hw_info_.get_cores();

  for (size_t i = 0; i < cores.size(); ++i) {
    const auto &core = cores[i];

    // Check if this is a P-core
    if (core.is_performance_core) {
      // Check NUMA node constraint
      if (numa_node >= 0) {
        auto numa_cores = hw_info_.get_numa_aware_cores();
        bool core_in_numa_node = false;
        if (numa_cores.find(numa_node) != numa_cores.end()) {
          const auto &node_cores = numa_cores.at(numa_node);
          core_in_numa_node = std::find(node_cores.begin(), node_cores.end(),
                                        static_cast<int>(i)) != node_cores.end();
        }
        if (!core_in_numa_node) {
          continue;
        }
      }

      pcore_ids.push_back(static_cast<int>(i));

      // Check if we've reached the maximum requested cores
      if (max_cores > 0 && static_cast<int>(pcore_ids.size()) >= max_cores) {
        break;
      }
    }
  }

  if (pcore_ids.empty()) {
    std::cout << "Warning: No P-cores found" << std::endl;
  } else {
    std::cout << "Found " << pcore_ids.size() << " P-core(s): ";
    for (size_t i = 0; i < pcore_ids.size(); ++i) {
      std::cout << pcore_ids[i];
      if (i < pcore_ids.size() - 1)
        std::cout << ", ";
    }
    std::cout << std::endl;
  }

  return pcore_ids;
}

bool ThreadAffinity::set_affinity_impl(pthread_t thread_id, const std::vector<int> &cpu_ids) const {
#ifdef __linux__
  if (cpu_ids.empty()) {
    return false;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  // Add each CPU ID to the set
  for (int cpu_id : cpu_ids) {
    if (cpu_id >= 0 && cpu_id < CPU_SETSIZE) {
      CPU_SET(cpu_id, &cpuset);
    } else {
      std::cout << "Warning: Invalid CPU ID " << cpu_id << ", skipping" << std::endl;
    }
  }

  // Set the affinity
  int result = pthread_setaffinity_np(thread_id, sizeof(cpu_set_t), &cpuset);
  if (result != 0) {
    std::cout << "Error setting thread affinity: " << strerror(result) << std::endl;
    return false;
  }

  std::cout << "Successfully set thread affinity to CPUs: ";
  for (size_t i = 0; i < cpu_ids.size(); ++i) {
    std::cout << cpu_ids[i];
    if (i < cpu_ids.size() - 1)
      std::cout << ", ";
  }
  std::cout << std::endl;

  return true;
#else
  return false;
#endif
}

void ThreadAffinity::print_affinity_info() const {
  std::cout << "\n=== CPU Affinity Information ===" << std::endl;
  std::cout << "Total logical cores: " << hw_info_.get_logical_cores() << std::endl;
  std::cout << "Total physical cores: " << hw_info_.get_physical_cores() << std::endl;
  std::cout << "Performance cores (P-cores): " << hw_info_.get_performance_cores() << std::endl;
  std::cout << "Efficiency cores (E-cores): " << hw_info_.get_efficiency_cores() << std::endl;
  std::cout << "Hyperthreading: " << (hw_info_.supports_hyperthreading() ? "Yes" : "No")
            << std::endl;

  const auto &cores = hw_info_.get_cores();
  if (!cores.empty()) {
    std::cout << "\nCore topology:" << std::endl;
    for (size_t i = 0; i < cores.size(); ++i) {
      const auto &core = cores[i];
      std::cout << "  CPU " << i << ": " << (core.is_performance_core ? "P-core" : "E-core")
                << " (Physical ID: " << core.physical_id << ", Core ID: " << core.core_id
                << ", Max Freq: " << core.max_freq_mhz << " MHz)" << std::endl;
    }
  }

  auto numa_cores = hw_info_.get_numa_aware_cores();
  if (numa_cores.size() > 1) {
    std::cout << "\nNUMA topology:" << std::endl;
    for (const auto &[node, core_list] : numa_cores) {
      std::cout << "  NUMA node " << node << ": ";
      for (size_t i = 0; i < core_list.size(); ++i) {
        std::cout << core_list[i];
        if (i < core_list.size() - 1)
          std::cout << ", ";
      }
      std::cout << std::endl;
    }
  }

#ifdef __linux__
  // Show current thread affinity
  cpu_set_t current_mask;
  if (pthread_getaffinity_np(pthread_self(), sizeof(current_mask), &current_mask) == 0) {
    std::cout << "\nCurrent thread affinity: ";
    bool first = true;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
      if (CPU_ISSET(i, &current_mask)) {
        if (!first)
          std::cout << ", ";
        std::cout << i;
        first = false;
      }
    }
    std::cout << std::endl;
  }
#endif

  std::cout << "=================================" << std::endl;
}

} // namespace utils