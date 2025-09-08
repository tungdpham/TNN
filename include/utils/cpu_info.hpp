#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>

namespace utils {

/**
 * Represents information about a single CPU core
 */
struct CoreInfo {
    int physical_id = -1;        // Physical CPU package ID
    int core_id = -1;            // Core ID within the package
    int processor_id = -1;       // Logical processor ID (thread)
    double current_freq_mhz = 0.0; // Current frequency in MHz
    double max_freq_mhz = 0.0;   // Maximum frequency in MHz
    double min_freq_mhz = 0.0;   // Minimum frequency in MHz
    bool is_performance_core = true; // P-core vs E-core (Intel 12th gen+)
    std::string governor = "unknown"; // CPU frequency governor
    double utilization_percent = 0.0; // Per-core utilization
    int cache_level1_kb = 0;     // L1 cache size
    int cache_level2_kb = 0;     // L2 cache size
    int cache_level3_kb = 0;     // L3 cache size (shared)
};

/**
 * Thermal information for CPU monitoring
 */
struct ThermalInfo {
    double current_temp_celsius = 0.0;
    double max_temp_celsius = 0.0;
    double critical_temp_celsius = 0.0;
    bool thermal_throttling = false;
    std::vector<double> per_core_temps; // If available
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
        std::string type; // "data", "instruction", "unified"
    };
    
    std::vector<CacheLevel> l1_caches;
    std::vector<CacheLevel> l2_caches;
    std::vector<CacheLevel> l3_caches;
    
    // NUMA information
    int numa_nodes = 1;
    std::map<int, std::vector<int>> numa_cpu_map; // NUMA node -> CPU list
};

/**
 * Comprehensive CPU information class for distributed computing
 * Optimized for neural network and CNN workload scheduling
 */
class CpuInfo {
public:
    // ==== Constructor/Destructor ====
    CpuInfo();
    ~CpuInfo();
    
    // ==== Initialization ====
    /**
     * Initialize CPU information gathering
     * @return true if successful, false otherwise
     */
    bool initialize();
    
    /**
     * Check if CPU info was successfully initialized
     */
    bool is_initialized() const { return initialized_; }
    
    // ==== Static Information (read once) ====
    
    // Basic CPU identification
    std::string get_vendor() const { return vendor_; }
    std::string get_model_name() const { return model_name_; }
    std::string get_architecture() const { return architecture_; }
    int get_family() const { return family_; }
    int get_model() const { return model_; }
    int get_stepping() const { return stepping_; }
    
    // Core and thread counts
    int get_physical_cores() const { return physical_cores_; }
    int get_logical_cores() const { return logical_cores_; }
    int get_performance_cores() const { return performance_cores_; }
    int get_efficiency_cores() const { return efficiency_cores_; }
    int get_sockets() const { return sockets_; }
    
    // Frequency information
    double get_base_frequency_mhz() const { return base_frequency_mhz_; }
    double get_max_frequency_mhz() const { return max_frequency_mhz_; }
    double get_min_frequency_mhz() const { return min_frequency_mhz_; }
    
    // Feature support (important for NN acceleration)
    bool supports_avx() const { return supports_avx_; }
    bool supports_avx2() const { return supports_avx2_; }
    bool supports_avx512() const { return supports_avx512_; }
    bool supports_fma() const { return supports_fma_; }
    bool supports_sse4_2() const { return supports_sse4_2_; }
    bool supports_hyperthreading() const { return supports_hyperthreading_; }
    
    // Memory hierarchy
    const MemoryHierarchy& get_memory_hierarchy() const { return memory_hierarchy_; }
    
    // Container/virtualization detection
    bool is_containerized() const { return is_containerized_; }
    bool is_virtualized() const { return is_virtualized_; }
    int get_container_cpu_limit() const { return container_cpu_limit_; }
    
    // ==== Dynamic Information (needs periodic updates) ====
    
    /**
     * Update dynamic CPU metrics (utilization, temperature, frequency)
     * @param sampling_interval_ms Sampling interval in milliseconds (default: 100ms)
     * @return true if successful
     */
    bool update_dynamic_info(int sampling_interval_ms = 100);
    
    // Overall CPU metrics
    double get_overall_utilization() const { return overall_utilization_; }
    double get_user_utilization() const { return user_utilization_; }
    double get_system_utilization() const { return system_utilization_; }
    double get_iowait_utilization() const { return iowait_utilization_; }
    
    // Per-core information
    const std::vector<CoreInfo>& get_cores() const { return cores_; }
    CoreInfo get_core_info(int logical_core_id) const;
    
    // Thermal information
    const ThermalInfo& get_thermal_info() const { return thermal_info_; }
    
    // Load averages (Linux/Unix)
    std::vector<double> get_load_averages() const { return load_averages_; } // 1min, 5min, 15min
    
    // ==== Utility Methods for Distributed Computing ====
    
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
    
    // ==== Debugging and Monitoring ====
    
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
    std::chrono::system_clock::time_point get_last_update() const { return last_update_; }

private:
    // ==== Private Members ====
    bool initialized_ = false;
    std::chrono::system_clock::time_point last_update_;
    
    // Static CPU information
    std::string vendor_;
    std::string model_name_;
    std::string architecture_;
    int family_ = 0;
    int model_ = 0;
    int stepping_ = 0;
    
    int physical_cores_ = 0;
    int logical_cores_ = 0;
    int performance_cores_ = 0;  // P-cores (Intel 12th gen+)
    int efficiency_cores_ = 0;   // E-cores (Intel 12th gen+)
    int sockets_ = 1;
    
    double base_frequency_mhz_ = 0.0;
    double max_frequency_mhz_ = 0.0;
    double min_frequency_mhz_ = 0.0;
    
    // Feature flags
    bool supports_avx_ = false;
    bool supports_avx2_ = false;
    bool supports_avx512_ = false;
    bool supports_fma_ = false;
    bool supports_sse4_2_ = false;
    bool supports_hyperthreading_ = false;
    
    // Memory hierarchy
    MemoryHierarchy memory_hierarchy_;
    
    // Container/VM detection
    bool is_containerized_ = false;
    bool is_virtualized_ = false;
    int container_cpu_limit_ = 0;
    
    // Dynamic information
    double overall_utilization_ = 0.0;
    double user_utilization_ = 0.0;
    double system_utilization_ = 0.0;
    double iowait_utilization_ = 0.0;
    
    std::vector<CoreInfo> cores_;
    ThermalInfo thermal_info_;
    std::vector<double> load_averages_;
    
    // ==== Private Implementation Methods ====
    
    // Platform-specific initialization
    bool init_cpu_identification();
    bool init_core_topology();
    bool init_frequency_info();
    bool init_feature_detection();
    bool init_memory_hierarchy();
    bool init_container_detection();
    
    // Platform-specific dynamic updates
    bool update_utilization(int sampling_interval_ms);
    bool update_thermal_info();
    bool update_frequency_info();
    bool update_load_averages();
    
    // Helper methods
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
    
    // Internal state for calculations
    struct CpuTimes {
        unsigned long long user = 0, nice = 0, system = 0, idle = 0;
        unsigned long long iowait = 0, irq = 0, softirq = 0, steal = 0;
    };
    CpuTimes prev_cpu_times_;
    std::vector<CpuTimes> prev_core_times_;
};

} // namespace utils