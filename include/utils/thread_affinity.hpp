/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "utils/hardware_info.hpp"
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <vector>

namespace tnn {

/**
 * @brief CPU Core Types for thread affinity
 */
enum class CoreType {
  PERFORMANCE_CORES, // P-cores (high performance)
  EFFICIENCY_CORES,  // E-cores (energy efficient)
  ALL_CORES,         // All available cores
  AUTO               // Let system decide
};

/**
 * @brief Thread affinity configuration options
 */
struct AffinityConfig {
  CoreType core_type = CoreType::AUTO;
  int max_threads = -1;              // -1 means use all available cores of the type
  bool numa_aware = true;            // Consider NUMA topology
  bool avoid_hyperthreading = false; // Only use physical cores, avoid logical cores
  int numa_node = -1;                // -1 means any NUMA node, >=0 specific node
};

/**
 * @brief Thread affinity utility class
 *
 * Provides functionality to set thread affinity to specific core types
 * based on detected hardware topology including P-core/E-core detection.
 */
class ThreadAffinity {
public:
  explicit ThreadAffinity(const HardwareInfo &hw_info) : hw_info_(hw_info) {}

  /**
   * @brief Set affinity for current thread to E-cores
   * @param config Affinity configuration options
   * @return true if successful, false otherwise
   */
  bool set_current_thread_affinity(const AffinityConfig &config = {}) const;

  /**
   * @brief Set affinity for a specific thread to E-cores
   * @param thread_id Native thread handle
   * @param config Affinity configuration options
   * @return true if successful, false otherwise
   */
  bool set_thread_affinity(pthread_t thread_id, const AffinityConfig &config = {}) const;

  /**
   * @brief Set affinity for a std::thread to E-cores
   * @param thread std::thread reference
   * @param config Affinity configuration options
   * @return true if successful, false otherwise
   */
  bool set_thread_affinity(std::thread &thread, const AffinityConfig &config = {}) const;

  /**
   * @brief Get list of recommended cores for the given configuration
   * @param config Affinity configuration options
   * @return Vector of logical CPU IDs
   */
  std::vector<int> get_recommended_cores(const AffinityConfig &config = {}) const;

  /**
   * @brief Set affinity to E-cores specifically (convenience function)
   * @param max_threads Maximum number of E-cores to use (-1 for all)
   * @return true if successful, false otherwise
   */
  bool set_ecore_affinity(int max_threads = -1) const {
    AffinityConfig config;
    config.core_type = CoreType::EFFICIENCY_CORES;
    config.max_threads = max_threads;
    return set_current_thread_affinity(config);
  }

  /**
   * @brief Set affinity to P-cores specifically (convenience function)
   * @param max_threads Maximum number of P-cores to use (-1 for all)
   * @return true if successful, false otherwise
   */
  bool set_pcore_affinity(int max_threads = -1) const {
    AffinityConfig config;
    config.core_type = CoreType::PERFORMANCE_CORES;
    config.max_threads = max_threads;
    return set_current_thread_affinity(config);
  }

  /**
   * @brief Check if E-cores are available on this system
   * @return true if E-cores are detected
   */
  bool has_efficiency_cores() const { return hw_info_.get_efficiency_cores() > 0; }

  /**
   * @brief Get number of available E-cores
   * @return Number of E-cores
   */
  int get_efficiency_core_count() const { return hw_info_.get_efficiency_cores(); }

  /**
   * @brief Get number of available P-cores
   * @return Number of P-cores
   */
  int get_performance_core_count() const { return hw_info_.get_performance_cores(); }

  /**
   * @brief Print affinity information for debugging
   */
  void print_affinity_info() const;

private:
  const HardwareInfo &hw_info_;

  /**
   * @brief Get E-core logical CPU IDs
   * @param max_cores Maximum number of cores to return (-1 for all)
   * @param numa_node Specific NUMA node (-1 for any)
   * @return Vector of logical CPU IDs that are E-cores
   */
  std::vector<int> get_efficiency_core_ids(int max_cores = -1, int numa_node = -1) const;

  /**
   * @brief Get P-core logical CPU IDs
   * @param max_cores Maximum number of cores to return (-1 for all)
   * @param numa_node Specific NUMA node (-1 for any)
   * @return Vector of logical CPU IDs that are P-cores
   */
  std::vector<int> get_performance_core_ids(int max_cores = -1, int numa_node = -1) const;

  /**
   * @brief Set CPU affinity using cpu_set_t
   * @param thread_id pthread_t handle
   * @param cpu_ids Vector of logical CPU IDs to bind to
   * @return true if successful
   */
  bool set_affinity_impl(pthread_t thread_id, const std::vector<int> &cpu_ids) const;
};

/**
 * @brief RAII helper for temporary thread affinity changes
 */
class ScopedThreadAffinity {
public:
  ScopedThreadAffinity(const ThreadAffinity &affinity, const AffinityConfig &config)
      : affinity_(affinity), thread_id_(pthread_self()) {
    // Save current affinity
    if (pthread_getaffinity_np(thread_id_, sizeof(old_mask_), &old_mask_) == 0) {
      saved_affinity_ = true;
      // Apply new affinity
      affinity_.set_current_thread_affinity(config);
    }
  }

  ~ScopedThreadAffinity() {
    if (saved_affinity_) {
      // Restore original affinity
      pthread_setaffinity_np(thread_id_, sizeof(old_mask_), &old_mask_);
    }
  }

  // Non-copyable, non-movable
  ScopedThreadAffinity(const ScopedThreadAffinity &) = delete;
  ScopedThreadAffinity &operator=(const ScopedThreadAffinity &) = delete;
  ScopedThreadAffinity(ScopedThreadAffinity &&) = delete;
  ScopedThreadAffinity &operator=(ScopedThreadAffinity &&) = delete;

private:
  const ThreadAffinity &affinity_;
  pthread_t thread_id_;
  cpu_set_t old_mask_;
  bool saved_affinity_ = false;
};

} // namespace tnn