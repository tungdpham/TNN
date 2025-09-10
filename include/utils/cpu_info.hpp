#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace utils {

/**
 * Represents information about a single CPU core
 */
struct CoreInfo {
  int physical_id = -1;
  int core_id = -1;
  int processor_id = -1;
  double current_freq_mhz = 0.0;
  double max_freq_mhz = 0.0;
  double min_freq_mhz = 0.0;
  bool is_performance_core = true;
  std::string governor = "unknown";
  double utilization_percent = 0.0;
  int cache_level1_kb = 0;
  int cache_level2_kb = 0;
  int cache_level3_kb = 0;
};

/**
 * Thermal information for CPU monitoring
 */
struct ThermalInfo {
  double current_temp_celsius = 0.0;
  double max_temp_celsius = 0.0;
  double critical_temp_celsius = 0.0;
  bool thermal_throttling = false;
  std::vector<double> per_core_temps;
};

/**
 * Memory hierarchy information relevant for NN/CNN workloads
 */
struct MemoryHierarchy {
  struct CacheLevel {
    int size_kb = 0;
    int line_size_bytes = 64;
    int associativity = 0;
    bool shared = false;
    std::string type;
  };

  std::vector<CacheLevel> l1_caches;
  std::vector<CacheLevel> l2_caches;
  std::vector<CacheLevel> l3_caches;

  int numa_nodes = 1;
  std::map<int, std::vector<int>> numa_cpu_map;
};

/**
 * Comprehensive CPU information class for distributed computing
 * Optimized for neural network and CNN workload scheduling
 */
class CpuInfo {
public:
  CpuInfo();
  ~CpuInfo();

  /**
   * Initialize CPU information gathering
   * @return true if successful, false otherwise
   */
  bool initialize();

  /**
   * Check if CPU info was successfully initialized
   */
  bool is_initialized() const { return initialized_; }

  std::string get_vendor() const { return vendor_; }
  std::string get_model_name() const { return model_name_; }
  std::string get_architecture() const { return architecture_; }
  int get_family() const { return family_; }
  int get_model() const { return model_; }
  int get_stepping() const { return stepping_; }

  int get_physical_cores() const { return physical_cores_; }
  int get_logical_cores() const { return logical_cores_; }
  int get_performance_cores() const { return performance_cores_; }
  int get_efficiency_cores() const { return efficiency_cores_; }
  int get_sockets() const { return sockets_; }

  double get_base_frequency_mhz() const { return base_frequency_mhz_; }
  double get_max_frequency_mhz() const { return max_frequency_mhz_; }
  double get_min_frequency_mhz() const { return min_frequency_mhz_; }

  bool supports_avx() const { return supports_avx_; }
  bool supports_avx2() const { return supports_avx2_; }
  bool supports_avx512() const { return supports_avx512_; }
  bool supports_fma() const { return supports_fma_; }
  bool supports_sse4_2() const { return supports_sse4_2_; }
  bool supports_hyperthreading() const { return supports_hyperthreading_; }

  const MemoryHierarchy &get_memory_hierarchy() const {
    return memory_hierarchy_;
  }

  bool is_containerized() const { return is_containerized_; }
  bool is_virtualized() const { return is_virtualized_; }
  int get_container_cpu_limit() const { return container_cpu_limit_; }

  /**
   * Update dynamic CPU metrics (utilization, temperature, frequency)
   * @param sampling_interval_ms Sampling interval in milliseconds (default:
   * 100ms)
   * @return true if successful
   */
  bool update_dynamic_info(int sampling_interval_ms = 100);

  double get_overall_utilization() const { return overall_utilization_; }
  double get_user_utilization() const { return user_utilization_; }
  double get_system_utilization() const { return system_utilization_; }
  double get_iowait_utilization() const { return iowait_utilization_; }

  const std::vector<CoreInfo> &get_cores() const { return cores_; }
  CoreInfo get_core_info(int logical_core_id) const;

  const ThermalInfo &get_thermal_info() const { return thermal_info_; }

  std::vector<double> get_load_averages() const { return load_averages_; }

  /**
   * Get optimal thread count for NN/CNN workloads
   * Considers P/E cores, thermal throttling, and current load
   */
  int get_optimal_thread_count() const;

  /**
   * Get recommended CPU affinity for high-performance computing
   * @param thread_count Number of threads to assign
   * @return Vector of logical CPU IDs to pin threads to
   */
  std::vector<int> get_recommended_cpu_affinity(int thread_count) const;

  /**
   * Check if CPU is suitable for heavy NN/CNN workloads
   * Considers thermal state, current load, and available features
   */
  bool is_suitable_for_heavy_workload() const;

  /**
   * Estimate relative performance compared to a baseline
   * Useful for load balancing in distributed systems
   */
  double get_performance_score() const;

  /**
   * Get NUMA-aware core distribution
   * @return Map of NUMA node -> recommended core IDs
   */
  std::map<int, std::vector<int>> get_numa_aware_cores() const;

  /**
   * Print comprehensive CPU information
   */
  void print_info() const;

  /**
   * Get JSON representation of CPU info
   */
  std::string to_json() const;

  /**
   * Get last update timestamp
   */
  std::chrono::system_clock::time_point get_last_update() const {
    return last_update_;
  }

private:
  bool initialized_ = false;
  std::chrono::system_clock::time_point last_update_;

  std::string vendor_;
  std::string model_name_;
  std::string architecture_;
  int family_ = 0;
  int model_ = 0;
  int stepping_ = 0;

  int physical_cores_ = 0;
  int logical_cores_ = 0;
  int performance_cores_ = 0;
  int efficiency_cores_ = 0;
  int sockets_ = 1;

  double base_frequency_mhz_ = 0.0;
  double max_frequency_mhz_ = 0.0;
  double min_frequency_mhz_ = 0.0;

  bool supports_avx_ = false;
  bool supports_avx2_ = false;
  bool supports_avx512_ = false;
  bool supports_fma_ = false;
  bool supports_sse4_2_ = false;
  bool supports_hyperthreading_ = false;

  MemoryHierarchy memory_hierarchy_;

  bool is_containerized_ = false;
  bool is_virtualized_ = false;
  int container_cpu_limit_ = 0;

  double overall_utilization_ = 0.0;
  double user_utilization_ = 0.0;
  double system_utilization_ = 0.0;
  double iowait_utilization_ = 0.0;

  std::vector<CoreInfo> cores_;
  ThermalInfo thermal_info_;
  std::vector<double> load_averages_;

  bool init_cpu_identification();
  bool init_core_topology();
  bool init_frequency_info();
  bool init_feature_detection();
  bool init_memory_hierarchy();
  bool init_container_detection();

  bool update_utilization(int sampling_interval_ms);
  bool update_thermal_info();
  bool update_frequency_info();
  bool update_load_averages();

  bool detect_pcore_ecore_topology();
  bool read_cpuinfo_linux();
  bool read_proc_stat_linux();
  bool read_thermal_linux();
  bool read_frequencies_linux();

#ifdef _WIN32
  bool init_windows_wmi();
  bool init_windows_frequency_wmi();
  bool init_windows_memory_hierarchy();
  bool init_windows_container_detection();
  bool update_windows_perfcounters();
  bool read_thermal_windows();
  bool read_frequencies_windows();
#endif

#ifdef __APPLE__
  bool init_macos_sysctl();
  bool update_macos_host_statistics();
#endif

  struct CpuTimes {
    unsigned long long user = 0, nice = 0, system = 0, idle = 0;
    unsigned long long iowait = 0, irq = 0, softirq = 0, steal = 0;
  };
  CpuTimes prev_cpu_times_;
  std::vector<CpuTimes> prev_core_times_;
};

} // namespace utils