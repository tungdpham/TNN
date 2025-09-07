#include "utils/cpu_info.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <thread>
#include <cstring>
#include <set>

#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __linux__
#include <cpuid.h>
#include <sys/sysinfo.h>
#ifdef NUMA_VERSION1_COMPATIBILITY
#include <numa.h>
#endif
#endif

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#include <pdh.h>
#include <wbemidl.h>
#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "wbemuuid.lib")
#endif

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <IOKit/IOKitLib.h>
#endif

namespace utils {

CpuInfo::CpuInfo() : initialized_(false) {
    last_update_ = std::chrono::system_clock::now();
}

CpuInfo::~CpuInfo() = default;

bool CpuInfo::initialize() {
    if (initialized_) {
        return true;
    }
    
    bool success = true;
    success &= init_cpu_identification();
    success &= init_core_topology();
    success &= init_frequency_info();
    success &= init_feature_detection();
    success &= init_memory_hierarchy();
    success &= init_container_detection();
    
    if (success) {
        // Initialize cores vector
        cores_.resize(logical_cores_);
        for (int i = 0; i < logical_cores_; ++i) {
            cores_[i].processor_id = i;
            cores_[i].physical_id = i / (logical_cores_ / physical_cores_);
            cores_[i].core_id = i % physical_cores_;
        }
        
        // Detect P/E core topology for modern Intel CPUs
        detect_pcore_ecore_topology();
        
        initialized_ = true;
    }
    
    return initialized_;
}

bool CpuInfo::init_cpu_identification() {
#ifdef __linux__
    return read_cpuinfo_linux();
#elif defined(_WIN32)
    return init_windows_wmi();
#elif defined(__APPLE__)
    return init_macos_sysctl();
#else
    return false;
#endif
}

bool CpuInfo::read_cpuinfo_linux() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        return false;
    }
    
    std::string line;
    int processor_count = 0;
    std::set<int> physical_ids;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find("processor") == 0) {
            processor_count++;
        } else if (line.find("vendor_id") == 0) {
            vendor_ = line.substr(line.find(":") + 2);
        } else if (line.find("model name") == 0) {
            model_name_ = line.substr(line.find(":") + 2);
        } else if (line.find("cpu family") == 0) {
            family_ = std::stoi(line.substr(line.find(":") + 2));
        } else if (line.find("model") == 0 && line.find("model name") == std::string::npos) {
            model_ = std::stoi(line.substr(line.find(":") + 2));
        } else if (line.find("stepping") == 0) {
            stepping_ = std::stoi(line.substr(line.find(":") + 2));
        } else if (line.find("physical id") == 0) {
            physical_ids.insert(std::stoi(line.substr(line.find(":") + 2)));
        }
    }
    
    logical_cores_ = processor_count;
    sockets_ = static_cast<int>(physical_ids.size());
    if (sockets_ == 0) sockets_ = 1; // Single socket system
    
    // Determine architecture
    if (vendor_.find("Intel") != std::string::npos) {
        architecture_ = "x86_64";
    } else if (vendor_.find("AMD") != std::string::npos) {
        architecture_ = "x86_64";
    } else if (vendor_.find("ARM") != std::string::npos) {
        architecture_ = "ARM";
    }
    
    return true;
}

bool CpuInfo::init_core_topology() {
#ifdef __linux__
    // Try to get physical core count from topology
    std::set<int> unique_core_ids;
    std::map<int, std::vector<int>> core_to_cpus;
    
    for (int cpu = 0; cpu < logical_cores_; ++cpu) {
        std::string core_id_path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/core_id";
        std::ifstream core_id_file(core_id_path);
        
        if (core_id_file.is_open()) {
            int core_id;
            core_id_file >> core_id;
            unique_core_ids.insert(core_id);
            core_to_cpus[core_id].push_back(cpu);
        }
    }
    
    if (!unique_core_ids.empty()) {
        physical_cores_ = unique_core_ids.size();
        
        // Check if we have hyperthreading by seeing if any core has multiple CPUs
        supports_hyperthreading_ = false;
        for (const auto& [core_id, cpus] : core_to_cpus) {
            if (cpus.size() > 1) {
                supports_hyperthreading_ = true;
                break;
            }
        }
        
        std::cout << "DEBUG: Detected " << physical_cores_ << " physical cores from topology" << std::endl;
        std::cout << "DEBUG: Hyperthreading: " << (supports_hyperthreading_ ? "Yes" : "No") << std::endl;
    } else {
        // Fallback: try thread siblings
        std::ifstream core_siblings("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
        if (core_siblings.is_open()) {
            std::string siblings;
            std::getline(core_siblings, siblings);
            // Count commas + 1 for hyperthreading
            int threads_per_core = std::count(siblings.begin(), siblings.end(), ',') + 1;
            physical_cores_ = logical_cores_ / threads_per_core;
            supports_hyperthreading_ = (threads_per_core > 1);
        } else {
            // Final fallback: assume no hyperthreading
            physical_cores_ = logical_cores_;
            supports_hyperthreading_ = false;
        }
    }
    
    return true;
#elif defined(_WIN32)
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    logical_cores_ = sysInfo.dwNumberOfProcessors;
    
    // Use WMI to get physical core count
    // TODO: Implement WMI query for physical cores
    physical_cores_ = logical_cores_; // Fallback
    
    return true;
#elif defined(__APPLE__)
    size_t size = sizeof(logical_cores_);
    sysctlbyname("hw.logicalcpu", &logical_cores_, &size, NULL, 0);
    
    size = sizeof(physical_cores_);
    sysctlbyname("hw.physicalcpu", &physical_cores_, &size, NULL, 0);
    
    supports_hyperthreading_ = (logical_cores_ > physical_cores_);
    
    return true;
#endif
    
    return false;
}

bool CpuInfo::init_frequency_info() {
#ifdef __linux__
    // Try to read base frequency from cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("cpu MHz") == 0) {
            base_frequency_mhz_ = std::stod(line.substr(line.find(":") + 2));
            break;
        }
    }
    
    // Try to read max frequency from scaling_max_freq
    std::ifstream max_freq("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq");
    if (max_freq.is_open()) {
        int freq_khz;
        max_freq >> freq_khz;
        max_frequency_mhz_ = freq_khz / 1000.0;
    }
    
    // Try to read min frequency
    std::ifstream min_freq("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq");
    if (min_freq.is_open()) {
        int freq_khz;
        min_freq >> freq_khz;
        min_frequency_mhz_ = freq_khz / 1000.0;
    }
    
    return true;
#elif defined(_WIN32)
    // Use WMI to get CPU frequency information
    // TODO: Implement WMI frequency queries
    return true;
#elif defined(__APPLE__)
    size_t size = sizeof(uint64_t);
    uint64_t freq;
    if (sysctlbyname("hw.cpufrequency", &freq, &size, NULL, 0) == 0) {
        base_frequency_mhz_ = freq / 1000000.0;
    }
    return true;
#endif
    
    return false;
}

bool CpuInfo::init_feature_detection() {
#if defined(__x86_64__) || defined(_M_X64)
    // Use CPUID to detect CPU features
    unsigned int eax, ebx, ecx, edx;
    
#ifdef _WIN32
    // MSVC intrinsics
    int cpuInfo[4];
    
    // Get feature flags from CPUID (leaf 1)
    __cpuid(cpuInfo, 1);
    eax = cpuInfo[0]; ebx = cpuInfo[1]; ecx = cpuInfo[2]; edx = cpuInfo[3];
    supports_sse4_2_ = (ecx & (1 << 20)) != 0;
    supports_fma_ = (ecx & (1 << 12)) != 0;
    
    // Extended features (leaf 7, subleaf 0)
    __cpuidex(cpuInfo, 7, 0);
    eax = cpuInfo[0]; ebx = cpuInfo[1]; ecx = cpuInfo[2]; edx = cpuInfo[3];
    supports_avx2_ = (ebx & (1 << 5)) != 0;
    supports_avx512_ = (ebx & (1 << 16)) != 0;
    
    // AVX support (requires OS support too)
    __cpuid(cpuInfo, 1);
    ecx = cpuInfo[2];
    bool osxsave = (ecx & (1 << 27)) != 0;
    bool avx_cpu = (ecx & (1 << 28)) != 0;
    if (osxsave && avx_cpu) {
        // Check if OS supports AVX using _xgetbv
        #if defined(_MSC_VER) && (_MSC_VER >= 1600)
        unsigned long long xcr0 = _xgetbv(0);
        supports_avx_ = (xcr0 & 0x6) == 0x6;
        #else
        // Fallback: assume AVX is supported if CPU supports it
        supports_avx_ = true;
        #endif
    }
#else
    // GCC intrinsics
    // Get feature flags from CPUID
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        supports_sse4_2_ = (ecx & (1 << 20)) != 0;
        supports_fma_ = (ecx & (1 << 12)) != 0;
    }
    
    // Extended features
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        supports_avx2_ = (ebx & (1 << 5)) != 0;
        supports_avx512_ = (ebx & (1 << 16)) != 0;
    }
    
    // AVX support (requires OS support too)
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        bool osxsave = (ecx & (1 << 27)) != 0;
        bool avx_cpu = (ecx & (1 << 28)) != 0;
        if (osxsave && avx_cpu) {
            // Check if OS supports AVX
            unsigned long long xcr0;
            __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "edx");
            supports_avx_ = (xcr0 & 0x6) == 0x6;
        }
    }
#endif
    
    return true;
#else
    // Non-x86 architectures
    return true;
#endif
}

bool CpuInfo::init_memory_hierarchy() {
#ifdef __linux__
    // Read cache information from sysfs
    for (int level = 1; level <= 3; ++level) {
        std::string cache_path = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(level - 1) + "/size";
        std::ifstream cache_file(cache_path);
        if (cache_file.is_open()) {
            std::string size_str;
            cache_file >> size_str;
            
            MemoryHierarchy::CacheLevel cache;
            cache.size_kb = std::stoi(size_str.substr(0, size_str.length() - 1)); // Remove 'K'
            
            if (level == 1) {
                memory_hierarchy_.l1_caches.push_back(cache);
            } else if (level == 2) {
                memory_hierarchy_.l2_caches.push_back(cache);
            } else if (level == 3) {
                memory_hierarchy_.l3_caches.push_back(cache);
            }
        }
    }
    
    // NUMA information
#ifdef NUMA_VERSION1_COMPATIBILITY
    if (numa_available() != -1) {
        memory_hierarchy_.numa_nodes = numa_num_configured_nodes();
        
        for (int node = 0; node < memory_hierarchy_.numa_nodes; ++node) {
            struct bitmask* cpus = numa_allocate_cpumask();
            int ret = numa_node_to_cpus(node, (unsigned long*)cpus->maskp, cpus->size);
            
            if (ret == 0) {
                std::vector<int> cpu_list;
                for (int cpu = 0; cpu < logical_cores_; ++cpu) {
                    if (numa_bitmask_isbitset(cpus, cpu)) {
                        cpu_list.push_back(cpu);
                    }
                }
                memory_hierarchy_.numa_cpu_map[node] = cpu_list;
            }
            
            numa_free_cpumask(cpus);
        }
    }
#endif
    
    return true;
#else
    return true; // Not implemented for other platforms yet
#endif
}

bool CpuInfo::init_container_detection() {
#ifdef __linux__
    // Check for container indicators
    std::ifstream cgroup("/proc/1/cgroup");
    if (cgroup.is_open()) {
        std::string line;
        while (std::getline(cgroup, line)) {
            if (line.find("docker") != std::string::npos ||
                line.find("containerd") != std::string::npos ||
                line.find("kubepods") != std::string::npos) {
                is_containerized_ = true;
                break;
            }
        }
    }
    
    // Check for virtualization
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("hypervisor") != std::string::npos) {
                is_virtualized_ = true;
                break;
            }
        }
    }
    
    // Try to get container CPU limit
    std::ifstream cpu_quota("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
    std::ifstream cpu_period("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
    
    if (cpu_quota.is_open() && cpu_period.is_open()) {
        int quota, period;
        cpu_quota >> quota;
        cpu_period >> period;
        
        if (quota > 0 && period > 0) {
            container_cpu_limit_ = static_cast<int>((double)quota / period * logical_cores_);
        }
    }
    
    return true;
#else
    return true; // Not implemented for other platforms yet
#endif
}

bool CpuInfo::detect_pcore_ecore_topology() {
#ifdef __linux__
    // For Intel 12th gen+, read from CPU topology more carefully
    std::map<int, std::vector<int>> core_to_threads;
    std::map<int, double> core_max_freq;
    
    // Parse topology to understand core relationships
    for (int cpu = 0; cpu < logical_cores_; ++cpu) {
        // Read core ID
        std::string core_id_path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/core_id";
        std::ifstream core_id_file(core_id_path);
        int core_id = -1;
        if (core_id_file.is_open()) {
            core_id_file >> core_id;
            core_to_threads[core_id].push_back(cpu);
        }
        
        // Read max frequency for this logical CPU
        std::string freq_path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/cpuinfo_max_freq";
        std::ifstream freq_file(freq_path);
        if (freq_file.is_open()) {
            int freq_khz;
            freq_file >> freq_khz;
            double freq_mhz = freq_khz / 1000.0;
            
            if (core_id >= 0) {
                if (core_max_freq.find(core_id) == core_max_freq.end() || freq_mhz > core_max_freq[core_id]) {
                    core_max_freq[core_id] = freq_mhz;
                }
            }
        }
    }
    
    // Determine P vs E cores based on frequency and thread count
    if (!core_max_freq.empty()) {
        // Find the maximum frequency among all cores
        double max_freq = 0;
        for (const auto& [core_id, freq] : core_max_freq) {
            max_freq = std::max(max_freq, freq);
        }
        
        // P-cores typically have higher max frequency and support hyperthreading
        // E-cores have lower frequency and no hyperthreading
        double p_core_threshold = max_freq * 0.85; // P-cores should be within 15% of max
        
        performance_cores_ = 0;
        efficiency_cores_ = 0;
        
        for (const auto& [core_id, threads] : core_to_threads) {
            double freq = core_max_freq[core_id];
            std::cout << "core_id: " << core_id << ", freq: " << freq << " MHz, threads: " << threads.size() << std::endl;
            bool is_p_core = (freq >= p_core_threshold) && (threads.size() > 1); // P-cores have hyperthreading
            
            if (is_p_core) {
                performance_cores_++;
                // Mark all threads of this core as P-core
                for (int thread : threads) {
                    if (thread < static_cast<int>(cores_.size())) {
                        cores_[thread].is_performance_core = true;
                        cores_[thread].max_freq_mhz = freq;
                        cores_[thread].core_id = core_id;
                    }
                }
            } else {
                efficiency_cores_++;
                // Mark threads as E-core
                for (int thread : threads) {
                    if (thread < static_cast<int>(cores_.size())) {
                        cores_[thread].is_performance_core = false;
                        cores_[thread].max_freq_mhz = freq;
                        cores_[thread].core_id = core_id;
                    }
                }
            }
        }
        
        std::cout << "DEBUG: Detected " << performance_cores_ << " P-cores and " 
                  << efficiency_cores_ << " E-cores based on topology and frequency" << std::endl;
    }else {
        std::cout << "DEBUG: Unable to determine P/E core topology, defaulting all to P-cores" << std::endl;
    }
#else
    // For non-Linux platforms, assume all cores are P-cores for now
    performance_cores_ = physical_cores_;
    efficiency_cores_ = 0;
#endif
    
    return true;
}

bool CpuInfo::update_dynamic_info(int sampling_interval_ms) {
    bool success = true;
    
    success &= update_utilization(sampling_interval_ms);
    success &= update_thermal_info();
    success &= update_frequency_info();
    success &= update_load_averages();
    
    if (success) {
        last_update_ = std::chrono::system_clock::now();
    }
    
    return success;
}

bool CpuInfo::update_utilization(int sampling_interval_ms) {
#ifdef __linux__
    return read_proc_stat_linux();
#else
    return false; // Not implemented for other platforms yet
#endif
}

bool CpuInfo::read_proc_stat_linux() {
#ifdef __linux__
    std::ifstream stat_file("/proc/stat");
    if (!stat_file.is_open()) {
        return false;
    }
    
    std::string line;
    std::getline(stat_file, line); // First line is overall CPU stats
    
    std::istringstream iss(line);
    std::string cpu_label;
    CpuTimes current_times;
    
    iss >> cpu_label >> current_times.user >> current_times.nice >> current_times.system 
        >> current_times.idle >> current_times.iowait >> current_times.irq 
        >> current_times.softirq >> current_times.steal;
    
    // Calculate utilization if we have previous data
    if (prev_cpu_times_.user != 0) {
        unsigned long long prev_idle = prev_cpu_times_.idle + prev_cpu_times_.iowait;
        unsigned long long idle = current_times.idle + current_times.iowait;
        
        unsigned long long prev_non_idle = prev_cpu_times_.user + prev_cpu_times_.nice + 
                                          prev_cpu_times_.system + prev_cpu_times_.irq + 
                                          prev_cpu_times_.softirq + prev_cpu_times_.steal;
        unsigned long long non_idle = current_times.user + current_times.nice + 
                                     current_times.system + current_times.irq + 
                                     current_times.softirq + current_times.steal;
        
        unsigned long long prev_total = prev_idle + prev_non_idle;
        unsigned long long total = idle + non_idle;
        
        unsigned long long total_diff = total - prev_total;
        unsigned long long idle_diff = idle - prev_idle;
        
        if (total_diff > 0) {
            overall_utilization_ = 100.0 * (total_diff - idle_diff) / total_diff;
            user_utilization_ = 100.0 * (current_times.user - prev_cpu_times_.user) / total_diff;
            system_utilization_ = 100.0 * (current_times.system - prev_cpu_times_.system) / total_diff;
            iowait_utilization_ = 100.0 * (current_times.iowait - prev_cpu_times_.iowait) / total_diff;
        }
    }
    
    prev_cpu_times_ = current_times;
    
    // Read per-core statistics
    std::vector<CpuTimes> current_core_times(logical_cores_);
    
    for (int core = 0; core < logical_cores_; ++core) {
        if (std::getline(stat_file, line)) {
            std::istringstream core_iss(line);
            std::string core_label;
            
            core_iss >> core_label >> current_core_times[core].user >> current_core_times[core].nice >> current_core_times[core].system 
                     >> current_core_times[core].idle >> current_core_times[core].iowait >> current_core_times[core].irq 
                     >> current_core_times[core].softirq >> current_core_times[core].steal;
        }
    }
    
    // Calculate per-core utilization based on difference from previous sample
    if (!prev_core_times_.empty() && prev_core_times_.size() == static_cast<size_t>(logical_cores_)) {
        for (int core = 0; core < logical_cores_; ++core) {
            unsigned long long prev_idle = prev_core_times_[core].idle + prev_core_times_[core].iowait;
            unsigned long long idle = current_core_times[core].idle + current_core_times[core].iowait;
            
            unsigned long long prev_non_idle = prev_core_times_[core].user + prev_core_times_[core].nice +
                                               prev_core_times_[core].system + prev_core_times_[core].irq +
                                               prev_core_times_[core].softirq + prev_core_times_[core].steal;
            unsigned long long non_idle = current_core_times[core].user + current_core_times[core].nice +
                                          current_core_times[core].system + current_core_times[core].irq +
                                          current_core_times[core].softirq + current_core_times[core].steal;

            unsigned long long total_diff = (idle + non_idle) - (prev_idle + prev_non_idle);
            unsigned long long idle_diff = idle - prev_idle;

            if (total_diff > 0) {
                cores_[core].utilization_percent = 100.0 * (total_diff - idle_diff) / total_diff;
            }
        }
    }
    
    // Update previous values for the next call
    prev_core_times_ = current_core_times;
    
    return true;
#else
    return false;
#endif
}

bool CpuInfo::update_thermal_info() {
#ifdef __linux__
    return read_thermal_linux();
#else
    return false; // Not implemented for other platforms yet
#endif
}

bool CpuInfo::read_thermal_linux() {
#ifdef __linux__
    bool found_temp = false;
    
    // Try to read from thermal zones - look for CPU/package temperatures
    for (int zone = 0; zone < 20; ++zone) {
        std::string temp_path = "/sys/class/thermal/thermal_zone" + std::to_string(zone) + "/temp";
        std::string type_path = "/sys/class/thermal/thermal_zone" + std::to_string(zone) + "/type";
        
        std::ifstream temp_file(temp_path);
        std::ifstream type_file(type_path);
        
        if (temp_file.is_open() && type_file.is_open()) {
            int temp_millidegrees;
            std::string zone_type;
            
            temp_file >> temp_millidegrees;
            type_file >> zone_type;
            
            double temp_celsius = temp_millidegrees / 1000.0;
            
            // Look for CPU-related thermal zones
            if (zone_type.find("cpu") != std::string::npos || 
                zone_type.find("x86_pkg_temp") != std::string::npos ||
                zone_type.find("coretemp") != std::string::npos ||
                zone_type.find("Package") != std::string::npos) {
                
                thermal_info_.current_temp_celsius = temp_celsius;
                found_temp = true;
                
                // Check for thermal throttling
                if (temp_celsius > 85.0) {
                    thermal_info_.thermal_throttling = true;
                }
                
                std::cout << "DEBUG: Found thermal zone " << zone << " (" << zone_type 
                          << "): " << temp_celsius << "°C" << std::endl;
                break; // Use first CPU thermal zone found
            }
        }
    }
    
    // Alternative: try hwmon interfaces (more reliable for some systems)
    if (!found_temp) {
        for (int hwmon = 0; hwmon < 10; ++hwmon) {
            std::string hwmon_path = "/sys/class/hwmon/hwmon" + std::to_string(hwmon);
            std::string name_path = hwmon_path + "/name";
            
            std::ifstream name_file(name_path);
            if (name_file.is_open()) {
                std::string hwmon_name;
                name_file >> hwmon_name;
                
                // Look for coretemp or similar CPU temperature monitors
                if (hwmon_name.find("coretemp") != std::string::npos ||
                    hwmon_name.find("k10temp") != std::string::npos) {
                    
                    // Try temp1_input (package temperature)
                    std::string temp_path = hwmon_path + "/temp1_input";
                    std::ifstream temp_file(temp_path);
                    
                    if (temp_file.is_open()) {
                        int temp_millidegrees;
                        temp_file >> temp_millidegrees;
                        
                        thermal_info_.current_temp_celsius = temp_millidegrees / 1000.0;
                        found_temp = true;
                        
                        std::cout << "DEBUG: Found hwmon temperature (" << hwmon_name 
                                  << "): " << thermal_info_.current_temp_celsius << "°C" << std::endl;
                        break;
                    }
                }
            }
        }
    }
    
    // Fallback for virtualized environments
    if (!found_temp) {
        if (is_virtualized_ || is_containerized_) {
            // In VMs/containers, thermal sensors are often not available
            // Set a reasonable default that indicates "thermal monitoring unavailable"
            thermal_info_.current_temp_celsius = -1.0; // Special value indicating unavailable
            std::cout << "DEBUG: Thermal monitoring unavailable in virtualized environment" << std::endl;
        }
    }
    
    return true; // Always return true since missing thermal sensors isn't an error
#else
    return false;
#endif
}

bool CpuInfo::update_frequency_info() {
#ifdef __linux__
    return read_frequencies_linux();
#else
    return false; // Not implemented for other platforms yet
#endif
}

bool CpuInfo::read_frequencies_linux() {
#ifdef __linux__
    // Update per-core frequencies
    for (int core = 0; core < logical_cores_; ++core) {
        std::string freq_path = "/sys/devices/system/cpu/cpu" + std::to_string(core) + "/cpufreq/scaling_cur_freq";
        std::ifstream freq_file(freq_path);
        
        if (freq_file.is_open()) {
            int freq_khz;
            freq_file >> freq_khz;
            cores_[core].current_freq_mhz = freq_khz / 1000.0;
        }
        
        // Read governor
        std::string gov_path = "/sys/devices/system/cpu/cpu" + std::to_string(core) + "/cpufreq/scaling_governor";
        std::ifstream gov_file(gov_path);
        
        if (gov_file.is_open()) {
            std::getline(gov_file, cores_[core].governor);
        }
    }
    
    return true;
#else
    return false;
#endif
}

bool CpuInfo::update_load_averages() {
#ifdef __linux__
    std::ifstream loadavg("/proc/loadavg");
    if (loadavg.is_open()) {
        double load1, load5, load15;
        loadavg >> load1 >> load5 >> load15;
        
        load_averages_ = {load1, load5, load15};
        return true;
    }
#endif
    return false;
}

CoreInfo CpuInfo::get_core_info(int logical_core_id) const {
    if (logical_core_id >= 0 && logical_core_id < static_cast<int>(cores_.size())) {
        return cores_[logical_core_id];
    }
    return CoreInfo{}; // Return default-constructed CoreInfo for invalid ID
}

int CpuInfo::get_optimal_thread_count() const {
    throw new std::runtime_error("Not implemented yet");
}

std::vector<int> CpuInfo::get_recommended_cpu_affinity(int thread_count) const {
    throw new std::runtime_error("Not implemented yet");
}

bool CpuInfo::is_suitable_for_heavy_workload() const {
    throw new std::runtime_error("Not implemented yet");
}

double CpuInfo::get_performance_score() const {
    throw new std::runtime_error("Not implemented yet");
}

std::map<int, std::vector<int>> CpuInfo::get_numa_aware_cores() const {
    std::map<int, std::vector<int>> numa_cores;
    
    if (memory_hierarchy_.numa_nodes <= 1) {
        // Single NUMA node - return all cores
        std::vector<int> all_cores;
        for (int i = 0; i < logical_cores_; ++i) {
            all_cores.push_back(i);
        }
        numa_cores[0] = all_cores;
    } else {
        // Use detected NUMA topology
        numa_cores = memory_hierarchy_.numa_cpu_map;
    }
    
    return numa_cores;
}

void CpuInfo::print_info() const {
    std::cout << "=== CPU Information ===" << std::endl;
    std::cout << "Vendor: " << vendor_ << std::endl;
    std::cout << "Model: " << model_name_ << std::endl;
    std::cout << "Architecture: " << architecture_ << std::endl;
    std::cout << "Physical Cores: " << physical_cores_ << std::endl;
    std::cout << "Logical Cores: " << logical_cores_ << std::endl;
    std::cout << "Performance Cores: " << performance_cores_ << std::endl;
    std::cout << "Efficiency Cores: " << efficiency_cores_ << std::endl;
    std::cout << "Base Frequency: " << base_frequency_mhz_ << " MHz" << std::endl;
    std::cout << "Max Frequency: " << max_frequency_mhz_ << " MHz" << std::endl;
    
    std::cout << "\n=== Features ===" << std::endl;
    std::cout << "AVX: " << (supports_avx_ ? "Yes" : "No") << std::endl;
    std::cout << "AVX2: " << (supports_avx2_ ? "Yes" : "No") << std::endl;
    std::cout << "AVX-512: " << (supports_avx512_ ? "Yes" : "No") << std::endl;
    std::cout << "FMA: " << (supports_fma_ ? "Yes" : "No") << std::endl;
    std::cout << "Hyperthreading: " << (supports_hyperthreading_ ? "Yes" : "No") << std::endl;
    
    std::cout << "\n=== Dynamic Info ===" << std::endl;
    std::cout << "Overall Utilization: " << overall_utilization_ << "%" << std::endl;
    
    if (thermal_info_.current_temp_celsius >= 0) {
        std::cout << "Temperature: " << thermal_info_.current_temp_celsius << "°C" << std::endl;
    } else {
        std::cout << "Temperature: Not available (virtualized environment)" << std::endl;
    }
    
    std::cout << "Thermal Throttling: " << (thermal_info_.thermal_throttling ? "Yes" : "No") << std::endl;
    
    std::cout << "\n=== Environment ===" << std::endl;
    std::cout << "Containerized: " << (is_containerized_ ? "Yes" : "No") << std::endl;
    std::cout << "Virtualized: " << (is_virtualized_ ? "Yes" : "No") << std::endl;
    
    if (!load_averages_.empty()) {
        std::cout << "Load Averages: " << load_averages_[0] << " " 
                  << load_averages_[1] << " " << load_averages_[2] << std::endl;
    }
}

std::string CpuInfo::to_json() const {
    std::ostringstream json;
    json << "{\n";
    json << "  \"vendor\": \"" << vendor_ << "\",\n";
    json << "  \"model_name\": \"" << model_name_ << "\",\n";
    json << "  \"architecture\": \"" << architecture_ << "\",\n";
    json << "  \"physical_cores\": " << physical_cores_ << ",\n";
    json << "  \"logical_cores\": " << logical_cores_ << ",\n";
    json << "  \"performance_cores\": " << performance_cores_ << ",\n";
    json << "  \"efficiency_cores\": " << efficiency_cores_ << ",\n";
    json << "  \"base_frequency_mhz\": " << base_frequency_mhz_ << ",\n";
    json << "  \"max_frequency_mhz\": " << max_frequency_mhz_ << ",\n";
    json << "  \"supports_avx\": " << (supports_avx_ ? "true" : "false") << ",\n";
    json << "  \"supports_avx2\": " << (supports_avx2_ ? "true" : "false") << ",\n";
    json << "  \"supports_avx512\": " << (supports_avx512_ ? "true" : "false") << ",\n";
    json << "  \"supports_fma\": " << (supports_fma_ ? "true" : "false") << ",\n";
    json << "  \"overall_utilization\": " << overall_utilization_ << ",\n";
    json << "  \"temperature_celsius\": " << thermal_info_.current_temp_celsius << ",\n";
    json << "  \"thermal_throttling\": " << (thermal_info_.thermal_throttling ? "true" : "false") << ",\n";
    json << "  \"is_containerized\": " << (is_containerized_ ? "true" : "false") << ",\n";
    json << "  \"is_virtualized\": " << (is_virtualized_ ? "true" : "false") << ",\n";
    json << "}";
    return json.str();
}

#ifdef _WIN32
bool CpuInfo::init_windows_wmi() {
    // Basic Windows CPU info detection without WMI
    // Use Win32 API functions as a fallback
    
    // Get processor name from registry
    HKEY hKey;
    if (RegOpenKeyEx(HKEY_LOCAL_MACHINE, 
                     "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 
                     0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char buffer[256];
        DWORD bufferSize = sizeof(buffer);
        if (RegQueryValueEx(hKey, "ProcessorNameString", nullptr, nullptr, 
                           (LPBYTE)buffer, &bufferSize) == ERROR_SUCCESS) {
            model_name_ = std::string(buffer);
        }
        RegCloseKey(hKey);
    }
    
    // Set vendor (simple detection)
    if (model_name_.find("Intel") != std::string::npos) {
        vendor_ = "Intel";
    } else if (model_name_.find("AMD") != std::string::npos) {
        vendor_ = "AMD";
    } else {
        vendor_ = "Unknown";
    }
    
    // Set architecture based on compilation target
#ifdef _M_X64
    architecture_ = "x86_64";
#elif defined(_M_IX86)
    architecture_ = "x86";
#elif defined(_M_ARM64)
    architecture_ = "arm64";
#else
    architecture_ = "unknown";
#endif
    
    return true;
}

bool CpuInfo::update_windows_perfcounters() {
    // Basic implementation that returns true
    // In a full implementation, this would query Windows performance counters
    return true;
}
#endif

} // namespace utils
