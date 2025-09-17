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
 * Information about a single RAM module
 */
struct RamModule {
  std::string manufacturer = "unknown";
  std::string part_number = "unknown";
  std::string serial_number = "unknown";
  long long size_bytes = 0;
  int speed_mhz = 0;
  std::string type = "unknown"; // DDR4, DDR5, etc.
  int rank = 0;
  int data_width_bits = 64;
  std::string form_factor = "unknown"; // DIMM, SO-DIMM, etc.
  std::string voltage = "unknown";
  int cas_latency = 0;
  std::string bank_locator = "unknown";
  std::string device_locator = "unknown";
  bool ecc_capable = false;
};

/**
 * Comprehensive RAM information for system analysis
 */
struct RamInfo {
  long long total_physical_memory_bytes = 0;
  long long available_memory_bytes = 0;
  long long used_memory_bytes = 0;
  long long free_memory_bytes = 0;
  long long cached_memory_bytes = 0;
  long long buffer_memory_bytes = 0;
  long long swap_total_bytes = 0;
  long long swap_used_bytes = 0;
  long long swap_free_bytes = 0;

  int memory_channels = 0;
  int populated_slots = 0;
  int total_slots = 0;
  double memory_bandwidth_gbps = 0.0;
  std::string memory_type = "unknown"; // DDR4, DDR5, LPDDR4, etc.
  int jedec_speed_mhz = 0;
  int effective_speed_mhz = 0;

  std::vector<RamModule> modules;

  // Memory timings
  struct MemoryTimings {
    int cas_latency = 0;
    int tras = 0;
    int trp = 0;
    int trcd = 0;
    int trc = 0;
    int command_rate = 0;
  } timings;

  // NUMA memory information
  std::map<int, long long> numa_memory_per_node;

  // ECC information
  bool ecc_enabled = false;
  long long correctable_errors = 0;
  long long uncorrectable_errors = 0;
};

/**
 * Comprehensive hardware information class for distributed computing
 * Optimized for neural network and CNN workload scheduling
 */
class HardwareInfo {
public:
  HardwareInfo();
  ~HardwareInfo();

  /**
   * Initialize hardware information gathering
   * @return true if successful, false otherwise
   */
  bool initialize();

  /**
   * Check if hardware info was successfully initialized
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

  // Additional CPU specifications for performance analysis
  int get_cache_line_size() const { return cache_line_size_; }
  double get_tdp_watts() const { return tdp_watts_; }
  double get_memory_bandwidth_gbps() const { return memory_bandwidth_gbps_; }
  bool supports_aes() const { return supports_aes_; }
  bool supports_sha() const { return supports_sha_; }
  bool supports_bmi1() const { return supports_bmi1_; }
  bool supports_bmi2() const { return supports_bmi2_; }
  std::string get_microarchitecture() const { return microarchitecture_; }
  int get_process_node_nm() const { return process_node_nm_; }

  const MemoryHierarchy &get_memory_hierarchy() const { return memory_hierarchy_; }

  bool is_containerized() const { return is_containerized_; }
  bool is_virtualized() const { return is_virtualized_; }
  int get_container_cpu_limit() const { return container_cpu_limit_; }

  bool update_dynamic_info();

  double get_overall_utilization() const { return overall_utilization_; }
  double get_user_utilization() const { return user_utilization_; }
  double get_system_utilization() const { return system_utilization_; }
  double get_iowait_utilization() const { return iowait_utilization_; }

  const std::vector<CoreInfo> &get_cores() const { return cores_; }
  CoreInfo get_core_info(int logical_core_id) const;

  const ThermalInfo &get_thermal_info() const { return thermal_info_; }

  // RAM information getters
  const RamInfo &get_ram_info() const { return ram_info_; }
  long long get_total_memory_bytes() const { return ram_info_.total_physical_memory_bytes; }
  long long get_available_memory_bytes() const { return ram_info_.available_memory_bytes; }
  long long get_used_memory_bytes() const { return ram_info_.used_memory_bytes; }
  double get_memory_utilization_percent() const;
  int get_memory_channels() const { return ram_info_.memory_channels; }
  std::string get_memory_type() const { return ram_info_.memory_type; }
  int get_effective_memory_speed_mhz() const { return ram_info_.effective_speed_mhz; }
  const std::vector<RamModule> &get_ram_modules() const { return ram_info_.modules; }
  bool is_ecc_enabled() const { return ram_info_.ecc_enabled; }

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
   * Print comprehensive hardware information
   */
  void print_info() const;

  /**
   * Get JSON representation of hardware info
   */
  std::string to_json() const;

  /**
   * Get last update timestamp
   */
  std::chrono::system_clock::time_point get_last_update() const { return last_update_; }

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

  // Additional CPU specifications
  int cache_line_size_ = 64;
  double tdp_watts_ = 0.0;
  double memory_bandwidth_gbps_ = 0.0;
  int memory_channels_ = 0;
  bool supports_aes_ = false;
  bool supports_sha_ = false;
  bool supports_bmi1_ = false;
  bool supports_bmi2_ = false;
  std::string microarchitecture_ = "unknown";
  int process_node_nm_ = 0;

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
  RamInfo ram_info_;
  std::vector<double> load_averages_;

  bool init_cpu_identification();
  bool init_core_topology();
  bool init_frequency_info();
  bool init_feature_detection();
  bool init_memory_hierarchy();
  bool init_ram_info();
  bool init_container_detection();

  bool update_utilization();
  bool update_thermal_info();
  bool update_frequency_info();
  bool update_load_averages();
  bool update_ram_usage();

  bool detect_pcore_ecore_topology();
  bool populate_core_cache_info();
  bool read_cpuinfo_linux();
  bool read_proc_stat_linux();
  bool read_thermal_linux();
  bool read_frequencies_linux();
  bool read_meminfo_linux();
  bool read_ram_modules_linux();

#ifdef _WIN32
  bool init_windows_wmi();
  bool init_windows_frequency_wmi();
  bool init_windows_memory_hierarchy();
  bool init_windows_ram_info();
  bool init_windows_container_detection();
  bool update_windows_perfcounters();
  bool read_thermal_windows();
  bool read_frequencies_windows();
  bool read_ram_modules_windows();
  bool update_ram_usage_windows();
#endif

#ifdef __APPLE__
  bool init_macos_sysctl();
  bool init_macos_ram_info();
  bool update_macos_host_statistics();
  bool update_macos_ram_usage();
#endif

  struct CpuTimes {
    unsigned long long user = 0, nice = 0, system = 0, idle = 0;
    unsigned long long iowait = 0, irq = 0, softirq = 0, steal = 0;
  };
  CpuTimes prev_cpu_times_;
  std::vector<CpuTimes> prev_core_times_;
};

} // namespace utils