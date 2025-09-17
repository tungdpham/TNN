/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "utils/hardware_info.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <thread>

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
#include <intrin.h>
#include <pdh.h>
#include <wbemidl.h>
#include <windows.h>
#ifdef _MSC_VER
#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "wbemuuid.lib")
#endif
#endif

#ifdef __APPLE__
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace utils {

HardwareInfo::HardwareInfo() : initialized_(false) {
  last_update_ = std::chrono::system_clock::now();
}

HardwareInfo::~HardwareInfo() = default;

bool HardwareInfo::initialize() {
  if (initialized_) {
    return true;
  }

  std::cout << "Initializing hardware information..." << std::endl;

  bool success = true;
  success &= init_cpu_identification();
  success &= init_core_topology();
  success &= init_frequency_info();
  success &= init_feature_detection();
  success &= init_memory_hierarchy();
  success &= init_ram_info();
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

    // Populate cache information for each core
    populate_core_cache_info();

    initialized_ = true;
  }

  return initialized_;
}

bool HardwareInfo::init_cpu_identification() {
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

bool HardwareInfo::read_cpuinfo_linux() {
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
  if (sockets_ == 0)
    sockets_ = 1; // Single socket system

  // Determine architecture
  if (vendor_.find("Intel") != std::string::npos) {
    architecture_ = "x86_64";
    // Detect Intel microarchitecture based on family/model
    if (family_ == 6) {
      if (model_ >= 0x8C && model_ <= 0x8F) {
        microarchitecture_ = "Alder Lake"; // 12th gen
        process_node_nm_ = 10;
      } else if (model_ >= 0xB7 && model_ <= 0xBA) {
        microarchitecture_ = "Raptor Lake"; // 13th/14th gen
        process_node_nm_ = 10;
      } else if (model_ >= 0xA7 && model_ <= 0xA8) {
        microarchitecture_ = "Rocket Lake"; // 11th gen
        process_node_nm_ = 14;
      } else if (model_ >= 0x7D && model_ <= 0x7E) {
        microarchitecture_ = "Ice Lake"; // 10th gen mobile
        process_node_nm_ = 10;
      } else {
        microarchitecture_ = "Unknown Intel";
      }
    }
  } else if (vendor_.find("AMD") != std::string::npos) {
    architecture_ = "x86_64";
    // Detect AMD microarchitecture based on family/model
    if (family_ == 23) {
      microarchitecture_ = "Zen/Zen+";
      process_node_nm_ = 14;
    } else if (family_ == 25) {
      microarchitecture_ = "Zen 3";
      process_node_nm_ = 7;
    } else if (family_ == 26) {
      microarchitecture_ = "Zen 4";
      process_node_nm_ = 5;
    } else {
      microarchitecture_ = "Unknown AMD";
    }
  } else if (vendor_.find("ARM") != std::string::npos) {
    architecture_ = "ARM";
  }

  return true;
}

bool HardwareInfo::init_core_topology() {
#ifdef __linux__
  // Try to get physical core count from topology
  std::set<int> unique_core_ids;
  std::map<int, std::vector<int>> core_to_cpus;

  for (int cpu = 0; cpu < logical_cores_; ++cpu) {
    std::string core_id_path =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/core_id";
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
    for (const auto &[core_id, cpus] : core_to_cpus) {
      if (cpus.size() > 1) {
        supports_hyperthreading_ = true;
        break;
      }
    }

    std::cout << "DEBUG: Detected " << physical_cores_ << " physical cores from topology"
              << std::endl;
    std::cout << "DEBUG: Hyperthreading: " << (supports_hyperthreading_ ? "Yes" : "No")
              << std::endl;
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

  // Get logical processor information to determine physical cores
  DWORD bufferSize = 0;
  GetLogicalProcessorInformation(nullptr, &bufferSize);

  if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
        bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));

    if (GetLogicalProcessorInformation(buffer.data(), &bufferSize)) {
      int physical_core_count = 0;
      int hyperthreading_count = 0;

      for (const auto &info : buffer) {
        if (info.Relationship == RelationProcessorCore) {
          physical_core_count++;
          // Count set bits in ProcessorMask to see threads per core
          ULONG_PTR mask = info.ProcessorMask;
          int threads_per_core = 0;
          while (mask) {
            if (mask & 1)
              threads_per_core++;
            mask >>= 1;
          }
          if (threads_per_core > 1) {
            hyperthreading_count++;
          }
        }
      }

      physical_cores_ = physical_core_count;
      supports_hyperthreading_ = (hyperthreading_count > 0);

      std::cout << "DEBUG: Detected " << physical_cores_ << " physical cores from Windows API"
                << std::endl;
      std::cout << "DEBUG: Hyperthreading: " << (supports_hyperthreading_ ? "Yes" : "No")
                << std::endl;
    } else {
      // Fallback: assume no hyperthreading
      physical_cores_ = logical_cores_;
      supports_hyperthreading_ = false;
    }
  } else {
    // Fallback
    physical_cores_ = logical_cores_;
    supports_hyperthreading_ = false;
  }

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

bool HardwareInfo::init_frequency_info() {
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
  // Query processor information using CPUID to get base frequency
  int cpuInfo[4];
  __cpuid(cpuInfo, 0x16); // Processor Frequency Information Leaf

  if (cpuInfo[0] != 0) {              // Check if leaf is supported
    base_frequency_mhz_ = cpuInfo[0]; // EAX = Base frequency in MHz
    max_frequency_mhz_ = cpuInfo[1];  // EBX = Maximum frequency in MHz
    // ECX = Bus frequency in MHz (not used here)
  } else {
    // Fallback: try to get frequency from registry
    HKEY hKey;
    if (RegOpenKeyEx(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0,
                     KEY_READ, &hKey) == ERROR_SUCCESS) {
      DWORD frequency = 0;
      DWORD bufferSize = sizeof(DWORD);
      if (RegQueryValueEx(hKey, "~MHz", nullptr, nullptr, (LPBYTE)&frequency, &bufferSize) ==
          ERROR_SUCCESS) {
        base_frequency_mhz_ = frequency;
        max_frequency_mhz_ = frequency; // Assume base = max without better info
      }
      RegCloseKey(hKey);
    }
  }

  // Try to get frequency info from WMI if available
  return init_windows_frequency_wmi();
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

bool HardwareInfo::init_feature_detection() {
#if defined(__x86_64__) || defined(_M_X64)
  // Use CPUID to detect CPU features
  [[maybe_unused]] unsigned int eax, ebx, ecx, edx;

#ifdef _WIN32
  // MSVC intrinsics
  int cpuInfo[4];

  // Get feature flags from CPUID (leaf 1)
  __cpuid(cpuInfo, 1);
  eax = cpuInfo[0];
  ebx = cpuInfo[1];
  ecx = cpuInfo[2];
  edx = cpuInfo[3];
  supports_sse4_2_ = (ecx & (1 << 20)) != 0;
  supports_fma_ = (ecx & (1 << 12)) != 0;
  supports_aes_ = (ecx & (1 << 25)) != 0;

  // Extended features (leaf 7, subleaf 0)
  __cpuidex(cpuInfo, 7, 0);
  eax = cpuInfo[0];
  ebx = cpuInfo[1];
  ecx = cpuInfo[2];
  edx = cpuInfo[3];
  supports_avx2_ = (ebx & (1 << 5)) != 0;
  supports_avx512_ = (ebx & (1 << 16)) != 0;
  supports_bmi1_ = (ebx & (1 << 3)) != 0;
  supports_bmi2_ = (ebx & (1 << 8)) != 0;
  supports_sha_ = (ebx & (1 << 29)) != 0;

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
    supports_aes_ = (ecx & (1 << 25)) != 0;
  }

  // Extended features
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    supports_avx2_ = (ebx & (1 << 5)) != 0;
    supports_avx512_ = (ebx & (1 << 16)) != 0;
    supports_bmi1_ = (ebx & (1 << 3)) != 0;
    supports_bmi2_ = (ebx & (1 << 8)) != 0;
    supports_sha_ = (ebx & (1 << 29)) != 0;
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

bool HardwareInfo::init_memory_hierarchy() {
#ifdef __linux__
  // Read cache information from sysfs - need to check all indices, not just
  // levels Linux sysfs uses indices that don't directly map to cache levels
  for (int index = 0; index < 10; ++index) {
    std::string base_path = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index);

    // Check if this cache index exists
    std::ifstream level_file(base_path + "/level");
    if (!level_file.is_open()) {
      break; // No more cache levels
    }

    MemoryHierarchy::CacheLevel cache;

    // Read cache level
    int level;
    level_file >> level;

    // Read cache size
    std::ifstream size_file(base_path + "/size");
    if (size_file.is_open()) {
      std::string size_str;
      size_file >> size_str;
      cache.size_kb = std::stoi(size_str.substr(0, size_str.length() - 1)); // Remove 'K'
    }

    // Read cache type
    std::ifstream type_file(base_path + "/type");
    if (type_file.is_open()) {
      type_file >> cache.type;
    }

    // Read line size
    std::ifstream line_size_file(base_path + "/coherency_line_size");
    if (line_size_file.is_open()) {
      line_size_file >> cache.line_size_bytes;
    }

    // Read associativity
    std::ifstream assoc_file(base_path + "/ways_of_associativity");
    if (assoc_file.is_open()) {
      assoc_file >> cache.associativity;
    }

    // Read shared CPU list to determine if cache is shared
    std::ifstream shared_file(base_path + "/shared_cpu_list");
    if (shared_file.is_open()) {
      std::string shared_cpus;
      std::getline(shared_file, shared_cpus);
      // If shared_cpus contains more than one CPU, it's shared
      cache.shared = (shared_cpus.find('-') != std::string::npos ||
                      shared_cpus.find(',') != std::string::npos);
    }

    // Add to appropriate cache level vector
    if (level == 1) {
      memory_hierarchy_.l1_caches.push_back(cache);
      // Set cache line size from first L1 cache found
      if (cache_line_size_ == 64 && cache.line_size_bytes > 0) {
        cache_line_size_ = cache.line_size_bytes;
      }
    } else if (level == 2) {
      memory_hierarchy_.l2_caches.push_back(cache);
    } else if (level == 3) {
      memory_hierarchy_.l3_caches.push_back(cache);
    }
  }

  // NUMA information
#ifdef NUMA_VERSION1_COMPATIBILITY
  if (numa_available() != -1) {
    memory_hierarchy_.numa_nodes = numa_num_configured_nodes();

    for (int node = 0; node < memory_hierarchy_.numa_nodes; ++node) {
      struct bitmask *cpus = numa_allocate_cpumask();
      int ret = numa_node_to_cpus(node, (unsigned long *)cpus->maskp, cpus->size);

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
#elif defined(_WIN32)
  return init_windows_memory_hierarchy();
#else
  return true; // Not implemented for other platforms yet
#endif
}

bool HardwareInfo::init_ram_info() {
#ifdef __linux__
  bool success = true;
  success &= read_meminfo_linux();
  success &= read_ram_modules_linux();
  return success;
#elif defined(_WIN32)
  return init_windows_ram_info();
#elif defined(__APPLE__)
  return init_macos_ram_info();
#else
  return true; // Not implemented for other platforms yet
#endif
}

bool HardwareInfo::init_container_detection() {
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
#elif defined(_WIN32)
  return init_windows_container_detection();
#else
  return true; // Not implemented for other platforms yet
#endif
}

bool HardwareInfo::detect_pcore_ecore_topology() {
#ifdef __linux__
  // For Intel 12th gen+, read from CPU topology more carefully
  std::map<int, std::vector<int>> core_to_threads;
  std::map<int, double> core_max_freq;

  // Parse topology to understand core relationships
  for (int cpu = 0; cpu < logical_cores_; ++cpu) {
    // Read core ID
    std::string core_id_path =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/core_id";
    std::ifstream core_id_file(core_id_path);
    int core_id = -1;
    if (core_id_file.is_open()) {
      core_id_file >> core_id;
      core_to_threads[core_id].push_back(cpu);
    }

    // Read max frequency for this logical CPU
    std::string freq_path =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/cpuinfo_max_freq";
    std::ifstream freq_file(freq_path);
    if (freq_file.is_open()) {
      int freq_khz;
      freq_file >> freq_khz;
      double freq_mhz = freq_khz / 1000.0;

      if (core_id >= 0) {
        if (core_max_freq.find(core_id) == core_max_freq.end() ||
            freq_mhz > core_max_freq[core_id]) {
          core_max_freq[core_id] = freq_mhz;
        }
      }
    }
  }

  // Determine P vs E cores based on frequency and thread count
  if (!core_max_freq.empty()) {
    // Find the maximum frequency among all cores
    double max_freq = 0;
    for (const auto &[core_id, freq] : core_max_freq) {
      max_freq = std::max(max_freq, freq);
    }

    // P-cores typically have higher max frequency and support hyperthreading
    // E-cores have lower frequency and no hyperthreading
    double p_core_threshold = max_freq * 0.85; // P-cores should be within 15% of max

    performance_cores_ = 0;
    efficiency_cores_ = 0;

    for (const auto &[core_id, threads] : core_to_threads) {
      double freq = core_max_freq[core_id];
      std::cout << "core_id: " << core_id << ", freq: " << freq
                << " MHz, threads: " << threads.size() << std::endl;
      bool is_p_core =
          (freq >= p_core_threshold) && (threads.size() > 1); // P-cores have hyperthreading

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
  } else {
    std::cerr << "WARNING: Unable to determine P/E core topology, defaulting all "
                 "to P-cores"
              << std::endl;
  }
#else
  // For non-Linux platforms, assume all cores are P-cores for now
  performance_cores_ = physical_cores_;
  efficiency_cores_ = 0;
#endif

  return true;
}

bool HardwareInfo::populate_core_cache_info() {
#ifdef __linux__
  // Populate cache information for each logical core
  for (int cpu = 0; cpu < logical_cores_; ++cpu) {
    if (cpu >= static_cast<int>(cores_.size())) {
      continue;
    }

    // For each cache level, find the total cache size accessible to this CPU
    int l1_total_kb = 0, l2_total_kb = 0, l3_total_kb = 0;

    // Check cache indices for this specific CPU
    for (int index = 0; index < 10; ++index) {
      std::string base_path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache/index" +
                              std::to_string(index);

      // Check if this cache index exists
      std::ifstream level_file(base_path + "/level");
      if (!level_file.is_open()) {
        break; // No more cache levels
      }

      int level;
      level_file >> level;

      // Read cache size
      std::ifstream size_file(base_path + "/size");
      if (size_file.is_open()) {
        std::string size_str;
        size_file >> size_str;
        int size_kb = std::stoi(size_str.substr(0, size_str.length() - 1)); // Remove 'K'

        // Accumulate cache sizes by level (handles separate I/D caches)
        if (level == 1) {
          l1_total_kb += size_kb;
        } else if (level == 2) {
          l2_total_kb += size_kb;
        } else if (level == 3) {
          l3_total_kb += size_kb;
        }
      }
    }

    // Update core cache information
    cores_[cpu].cache_level1_kb = l1_total_kb;
    cores_[cpu].cache_level2_kb = l2_total_kb;
    cores_[cpu].cache_level3_kb = l3_total_kb;
  }

  return true;
#else
  // For non-Linux platforms, set cache info from memory hierarchy if available
  for (int cpu = 0; cpu < logical_cores_; ++cpu) {
    if (cpu >= static_cast<int>(cores_.size())) {
      continue;
    }

    // Set cache sizes based on memory hierarchy information
    int l1_total = 0, l2_total = 0, l3_total = 0;

    for (const auto &cache : memory_hierarchy_.l1_caches) {
      l1_total += cache.size_kb;
    }
    for (const auto &cache : memory_hierarchy_.l2_caches) {
      l2_total += cache.size_kb;
    }
    for (const auto &cache : memory_hierarchy_.l3_caches) {
      l3_total += cache.size_kb;
    }

    cores_[cpu].cache_level1_kb = l1_total;
    cores_[cpu].cache_level2_kb = l2_total;
    cores_[cpu].cache_level3_kb = l3_total;
  }

  return true;
#endif
}

bool HardwareInfo::update_dynamic_info() {
  bool success = true;

  success &= update_utilization();
  success &= update_thermal_info();
  success &= update_frequency_info();
  success &= update_load_averages();
  success &= update_ram_usage();

  if (success) {
    last_update_ = std::chrono::system_clock::now();
  }

  return success;
}

bool HardwareInfo::update_utilization() {
#ifdef __linux__
  return read_proc_stat_linux();
#elif defined(_WIN32)
  return update_windows_perfcounters();
#else
  return false; // Not implemented for other platforms yet
#endif
}

bool HardwareInfo::read_proc_stat_linux() {
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

  iss >> cpu_label >> current_times.user >> current_times.nice >> current_times.system >>
      current_times.idle >> current_times.iowait >> current_times.irq >> current_times.softirq >>
      current_times.steal;

  // Calculate utilization if we have previous data
  if (prev_cpu_times_.user != 0) {
    unsigned long long prev_idle = prev_cpu_times_.idle + prev_cpu_times_.iowait;
    unsigned long long idle = current_times.idle + current_times.iowait;

    unsigned long long prev_non_idle = prev_cpu_times_.user + prev_cpu_times_.nice +
                                       prev_cpu_times_.system + prev_cpu_times_.irq +
                                       prev_cpu_times_.softirq + prev_cpu_times_.steal;
    unsigned long long non_idle = current_times.user + current_times.nice + current_times.system +
                                  current_times.irq + current_times.softirq + current_times.steal;

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

      core_iss >> core_label >> current_core_times[core].user >> current_core_times[core].nice >>
          current_core_times[core].system >> current_core_times[core].idle >>
          current_core_times[core].iowait >> current_core_times[core].irq >>
          current_core_times[core].softirq >> current_core_times[core].steal;
    }
  }

  // Calculate per-core utilization based on difference from previous sample
  if (!prev_core_times_.empty() && prev_core_times_.size() == static_cast<size_t>(logical_cores_)) {
    for (int core = 0; core < logical_cores_; ++core) {
      unsigned long long prev_idle = prev_core_times_[core].idle + prev_core_times_[core].iowait;
      unsigned long long idle = current_core_times[core].idle + current_core_times[core].iowait;

      unsigned long long prev_non_idle =
          prev_core_times_[core].user + prev_core_times_[core].nice +
          prev_core_times_[core].system + prev_core_times_[core].irq +
          prev_core_times_[core].softirq + prev_core_times_[core].steal;
      unsigned long long non_idle = current_core_times[core].user + current_core_times[core].nice +
                                    current_core_times[core].system + current_core_times[core].irq +
                                    current_core_times[core].softirq +
                                    current_core_times[core].steal;

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

bool HardwareInfo::update_thermal_info() {
#ifdef __linux__
  return read_thermal_linux();
#elif defined(_WIN32)
  return read_thermal_windows();
#else
  return false; // Not implemented for other platforms yet
#endif
}

bool HardwareInfo::read_thermal_linux() {
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
      // Set a reasonable default that indicates "thermal monitoring
      // unavailable"
      thermal_info_.current_temp_celsius = -1.0; // Special value indicating unavailable
      std::cout << "DEBUG: Thermal monitoring unavailable in virtualized environment" << std::endl;
    }
  }

  return true; // Always return true since missing thermal sensors isn't an
               // error
#else
  return false;
#endif
}

bool HardwareInfo::update_frequency_info() {
#ifdef __linux__
  return read_frequencies_linux();
#elif defined(_WIN32)
  return read_frequencies_windows();
#else
  return false; // Not implemented for other platforms yet
#endif
}

bool HardwareInfo::read_frequencies_linux() {
#ifdef __linux__
  // Update per-core frequencies
  for (int core = 0; core < logical_cores_; ++core) {
    std::string freq_path =
        "/sys/devices/system/cpu/cpu" + std::to_string(core) + "/cpufreq/scaling_cur_freq";
    std::ifstream freq_file(freq_path);

    if (freq_file.is_open()) {
      int freq_khz;
      freq_file >> freq_khz;
      cores_[core].current_freq_mhz = freq_khz / 1000.0;
    }

    // Read governor
    std::string gov_path =
        "/sys/devices/system/cpu/cpu" + std::to_string(core) + "/cpufreq/scaling_governor";
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

bool HardwareInfo::update_load_averages() {
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

CoreInfo HardwareInfo::get_core_info(int logical_core_id) const {
  if (logical_core_id >= 0 && logical_core_id < static_cast<int>(cores_.size())) {
    return cores_[logical_core_id];
  }
  return CoreInfo{}; // Return default-constructed CoreInfo for invalid ID
}

std::map<int, std::vector<int>> HardwareInfo::get_numa_aware_cores() const {
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

void HardwareInfo::print_info() const {
  std::cout << "=== Hardware Information ===" << std::endl;
  std::cout << "=== CPU Information ===" << std::endl;
  std::cout << "Vendor: " << vendor_ << std::endl;
  std::cout << "Model: " << model_name_ << std::endl;
  std::cout << "Architecture: " << architecture_ << std::endl;
  std::cout << "Microarchitecture: " << microarchitecture_ << std::endl;
  if (process_node_nm_ > 0) {
    std::cout << "Process Node: " << process_node_nm_ << "nm" << std::endl;
  }
  std::cout << "Family/Model/Stepping: " << family_ << "/" << model_ << "/" << stepping_
            << std::endl;
  std::cout << "Physical Cores: " << physical_cores_ << std::endl;
  std::cout << "Logical Cores: " << logical_cores_ << std::endl;
  std::cout << "Performance Cores: " << performance_cores_ << std::endl;
  std::cout << "Efficiency Cores: " << efficiency_cores_ << std::endl;
  std::cout << "Sockets: " << sockets_ << std::endl;
  std::cout << "Base Frequency: " << base_frequency_mhz_ << " MHz" << std::endl;
  std::cout << "Max Frequency: " << max_frequency_mhz_ << " MHz" << std::endl;

  std::cout << "\n=== Cache Hierarchy ===" << std::endl;
  std::cout << "Cache Line Size: " << cache_line_size_ << " bytes" << std::endl;

  // L1 Caches
  if (!memory_hierarchy_.l1_caches.empty()) {
    std::cout << "L1 Caches:" << std::endl;
    for (size_t i = 0; i < memory_hierarchy_.l1_caches.size(); ++i) {
      const auto &cache = memory_hierarchy_.l1_caches[i];
      std::cout << "  " << cache.type << ": " << cache.size_kb << "KB, " << cache.associativity
                << "-way, " << cache.line_size_bytes << "B line, "
                << (cache.shared ? "shared" : "private") << std::endl;
    }
  }

  // L2 Caches
  if (!memory_hierarchy_.l2_caches.empty()) {
    std::cout << "L2 Caches:" << std::endl;
    for (size_t i = 0; i < memory_hierarchy_.l2_caches.size(); ++i) {
      const auto &cache = memory_hierarchy_.l2_caches[i];
      std::cout << "  " << cache.type << ": " << cache.size_kb << "KB, " << cache.associativity
                << "-way, " << cache.line_size_bytes << "B line, "
                << (cache.shared ? "shared" : "private") << std::endl;
    }
  }

  // L3 Caches
  if (!memory_hierarchy_.l3_caches.empty()) {
    std::cout << "L3 Caches:" << std::endl;
    for (size_t i = 0; i < memory_hierarchy_.l3_caches.size(); ++i) {
      const auto &cache = memory_hierarchy_.l3_caches[i];
      std::cout << "  " << cache.type << ": " << cache.size_kb << "KB, " << cache.associativity
                << "-way, " << cache.line_size_bytes << "B line, "
                << (cache.shared ? "shared" : "private") << std::endl;
    }
  }

  std::cout << "\n=== Features ===" << std::endl;
  std::cout << "AVX: " << (supports_avx_ ? "Yes" : "No") << std::endl;
  std::cout << "AVX2: " << (supports_avx2_ ? "Yes" : "No") << std::endl;
  std::cout << "AVX-512: " << (supports_avx512_ ? "Yes" : "No") << std::endl;
  std::cout << "FMA: " << (supports_fma_ ? "Yes" : "No") << std::endl;
  std::cout << "SSE 4.2: " << (supports_sse4_2_ ? "Yes" : "No") << std::endl;
  std::cout << "AES: " << (supports_aes_ ? "Yes" : "No") << std::endl;
  std::cout << "SHA: " << (supports_sha_ ? "Yes" : "No") << std::endl;
  std::cout << "BMI1: " << (supports_bmi1_ ? "Yes" : "No") << std::endl;
  std::cout << "BMI2: " << (supports_bmi2_ ? "Yes" : "No") << std::endl;
  std::cout << "Hyperthreading: " << (supports_hyperthreading_ ? "Yes" : "No") << std::endl;

  std::cout << "\n=== Memory ===" << std::endl;
  std::cout << "NUMA Nodes: " << memory_hierarchy_.numa_nodes << std::endl;
  if (memory_channels_ > 0) {
    std::cout << "Memory Channels: " << memory_channels_ << std::endl;
  }
  if (memory_bandwidth_gbps_ > 0) {
    std::cout << "Memory Bandwidth: " << memory_bandwidth_gbps_ << " GB/s" << std::endl;
  }

  std::cout << "\n=== RAM Information ===" << std::endl;
  std::cout << "Total Memory: "
            << (ram_info_.total_physical_memory_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB"
            << std::endl;
  std::cout << "Available Memory: "
            << (ram_info_.available_memory_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB"
            << std::endl;
  std::cout << "Used Memory: " << (ram_info_.used_memory_bytes / (1024.0 * 1024.0 * 1024.0))
            << " GB" << std::endl;
  std::cout << "Memory Utilization: " << get_memory_utilization_percent() << "%" << std::endl;
  std::cout << "Memory Type: " << ram_info_.memory_type << std::endl;
  std::cout << "Effective Speed: " << ram_info_.effective_speed_mhz << " MHz" << std::endl;
  std::cout << "Populated Slots: " << ram_info_.populated_slots << "/" << ram_info_.total_slots
            << std::endl;
  std::cout << "ECC Enabled: " << (ram_info_.ecc_enabled ? "Yes" : "No") << std::endl;
  if (ram_info_.swap_total_bytes > 0) {
    std::cout << "Swap Total: " << (ram_info_.swap_total_bytes / (1024.0 * 1024.0 * 1024.0))
              << " GB" << std::endl;
    std::cout << "Swap Used: " << (ram_info_.swap_used_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB"
              << std::endl;
  }

  std::cout << "\n=== Performance ===" << std::endl;
  if (tdp_watts_ > 0) {
    std::cout << "TDP: " << tdp_watts_ << " Watts" << std::endl;
  }

  std::cout << "\n=== Dynamic Info ===" << std::endl;
  std::cout << "Overall Utilization: " << overall_utilization_ << "%" << std::endl;

  if (thermal_info_.current_temp_celsius >= 0) {
    std::cout << "Temperature: " << thermal_info_.current_temp_celsius << "°C" << std::endl;
  } else {
    std::cout << "Temperature: Not available (virtualized environment)" << std::endl;
  }

  std::cout << "Thermal Throttling: " << (thermal_info_.thermal_throttling ? "Yes" : "No")
            << std::endl;

  std::cout << "\n=== Environment ===" << std::endl;
  std::cout << "Containerized: " << (is_containerized_ ? "Yes" : "No") << std::endl;
  std::cout << "Virtualized: " << (is_virtualized_ ? "Yes" : "No") << std::endl;

  if (!load_averages_.empty()) {
    std::cout << "Load Averages: " << load_averages_[0] << " " << load_averages_[1] << " "
              << load_averages_[2] << std::endl;
  }
}

std::string HardwareInfo::to_json() const {
  std::ostringstream json;
  json << "{\n";
  json << "  \"vendor\": \"" << vendor_ << "\",\n";
  json << "  \"model_name\": \"" << model_name_ << "\",\n";
  json << "  \"architecture\": \"" << architecture_ << "\",\n";
  json << "  \"microarchitecture\": \"" << microarchitecture_ << "\",\n";
  json << "  \"process_node_nm\": " << process_node_nm_ << ",\n";
  json << "  \"family\": " << family_ << ",\n";
  json << "  \"model\": " << model_ << ",\n";
  json << "  \"stepping\": " << stepping_ << ",\n";
  json << "  \"physical_cores\": " << physical_cores_ << ",\n";
  json << "  \"logical_cores\": " << logical_cores_ << ",\n";
  json << "  \"performance_cores\": " << performance_cores_ << ",\n";
  json << "  \"efficiency_cores\": " << efficiency_cores_ << ",\n";
  json << "  \"sockets\": " << sockets_ << ",\n";
  json << "  \"base_frequency_mhz\": " << base_frequency_mhz_ << ",\n";
  json << "  \"max_frequency_mhz\": " << max_frequency_mhz_ << ",\n";
  json << "  \"min_frequency_mhz\": " << min_frequency_mhz_ << ",\n";

  // Cache information
  json << "  \"cache_line_size\": " << cache_line_size_ << ",\n";
  json << "  \"cache_hierarchy\": {\n";

  // L1 caches
  json << "    \"l1_caches\": [\n";
  for (size_t i = 0; i < memory_hierarchy_.l1_caches.size(); ++i) {
    const auto &cache = memory_hierarchy_.l1_caches[i];
    json << "      {\n";
    json << "        \"size_kb\": " << cache.size_kb << ",\n";
    json << "        \"type\": \"" << cache.type << "\",\n";
    json << "        \"line_size_bytes\": " << cache.line_size_bytes << ",\n";
    json << "        \"associativity\": " << cache.associativity << ",\n";
    json << "        \"shared\": " << (cache.shared ? "true" : "false") << "\n";
    json << "      }" << (i < memory_hierarchy_.l1_caches.size() - 1 ? "," : "") << "\n";
  }
  json << "    ],\n";

  // L2 caches
  json << "    \"l2_caches\": [\n";
  for (size_t i = 0; i < memory_hierarchy_.l2_caches.size(); ++i) {
    const auto &cache = memory_hierarchy_.l2_caches[i];
    json << "      {\n";
    json << "        \"size_kb\": " << cache.size_kb << ",\n";
    json << "        \"type\": \"" << cache.type << "\",\n";
    json << "        \"line_size_bytes\": " << cache.line_size_bytes << ",\n";
    json << "        \"associativity\": " << cache.associativity << ",\n";
    json << "        \"shared\": " << (cache.shared ? "true" : "false") << "\n";
    json << "      }" << (i < memory_hierarchy_.l2_caches.size() - 1 ? "," : "") << "\n";
  }
  json << "    ],\n";

  // L3 caches
  json << "    \"l3_caches\": [\n";
  for (size_t i = 0; i < memory_hierarchy_.l3_caches.size(); ++i) {
    const auto &cache = memory_hierarchy_.l3_caches[i];
    json << "      {\n";
    json << "        \"size_kb\": " << cache.size_kb << ",\n";
    json << "        \"type\": \"" << cache.type << "\",\n";
    json << "        \"line_size_bytes\": " << cache.line_size_bytes << ",\n";
    json << "        \"associativity\": " << cache.associativity << ",\n";
    json << "        \"shared\": " << (cache.shared ? "true" : "false") << "\n";
    json << "      }" << (i < memory_hierarchy_.l3_caches.size() - 1 ? "," : "") << "\n";
  }
  json << "    ]\n";
  json << "  },\n";

  // Feature flags
  json << "  \"features\": {\n";
  json << "    \"supports_avx\": " << (supports_avx_ ? "true" : "false") << ",\n";
  json << "    \"supports_avx2\": " << (supports_avx2_ ? "true" : "false") << ",\n";
  json << "    \"supports_avx512\": " << (supports_avx512_ ? "true" : "false") << ",\n";
  json << "    \"supports_fma\": " << (supports_fma_ ? "true" : "false") << ",\n";
  json << "    \"supports_sse4_2\": " << (supports_sse4_2_ ? "true" : "false") << ",\n";
  json << "    \"supports_aes\": " << (supports_aes_ ? "true" : "false") << ",\n";
  json << "    \"supports_sha\": " << (supports_sha_ ? "true" : "false") << ",\n";
  json << "    \"supports_bmi1\": " << (supports_bmi1_ ? "true" : "false") << ",\n";
  json << "    \"supports_bmi2\": " << (supports_bmi2_ ? "true" : "false") << ",\n";
  json << "    \"supports_hyperthreading\": " << (supports_hyperthreading_ ? "true" : "false")
       << "\n";
  json << "  },\n";

  // Memory information
  json << "  \"memory\": {\n";
  json << "    \"numa_nodes\": " << memory_hierarchy_.numa_nodes << ",\n";
  json << "    \"memory_channels\": " << memory_channels_ << ",\n";
  json << "    \"memory_bandwidth_gbps\": " << memory_bandwidth_gbps_ << "\n";
  json << "  },\n";

  // Dynamic information
  json << "  \"dynamic\": {\n";
  json << "    \"overall_utilization\": " << overall_utilization_ << ",\n";
  json << "    \"temperature_celsius\": " << thermal_info_.current_temp_celsius << ",\n";
  json << "    \"thermal_throttling\": " << (thermal_info_.thermal_throttling ? "true" : "false")
       << "\n";
  json << "  },\n";

  // RAM information
  json << "  \"ram\": {\n";
  json << "    \"total_physical_memory_bytes\": " << ram_info_.total_physical_memory_bytes << ",\n";
  json << "    \"available_memory_bytes\": " << ram_info_.available_memory_bytes << ",\n";
  json << "    \"used_memory_bytes\": " << ram_info_.used_memory_bytes << ",\n";
  json << "    \"memory_utilization_percent\": " << get_memory_utilization_percent() << ",\n";
  json << "    \"memory_type\": \"" << ram_info_.memory_type << "\",\n";
  json << "    \"effective_speed_mhz\": " << ram_info_.effective_speed_mhz << ",\n";
  json << "    \"memory_channels\": " << ram_info_.memory_channels << ",\n";
  json << "    \"populated_slots\": " << ram_info_.populated_slots << ",\n";
  json << "    \"total_slots\": " << ram_info_.total_slots << ",\n";
  json << "    \"ecc_enabled\": " << (ram_info_.ecc_enabled ? "true" : "false") << ",\n";
  json << "    \"swap_total_bytes\": " << ram_info_.swap_total_bytes << ",\n";
  json << "    \"swap_used_bytes\": " << ram_info_.swap_used_bytes << "\n";
  json << "  },\n";

  // Environment
  json << "  \"environment\": {\n";
  json << "    \"is_containerized\": " << (is_containerized_ ? "true" : "false") << ",\n";
  json << "    \"is_virtualized\": " << (is_virtualized_ ? "true" : "false") << ",\n";
  json << "    \"container_cpu_limit\": " << container_cpu_limit_ << "\n";
  json << "  },\n";

  // Performance specs
  json << "  \"performance\": {\n";
  json << "    \"tdp_watts\": " << tdp_watts_ << "\n";
  json << "  }\n";
  json << "}";
  return json.str();
}

#ifdef _WIN32
bool HardwareInfo::init_windows_wmi() {
  // Basic Windows CPU info detection without WMI
  // Use Win32 API functions as a fallback

  // Get processor name from registry
  HKEY hKey;
  if (RegOpenKeyEx(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0,
                   KEY_READ, &hKey) == ERROR_SUCCESS) {
    char buffer[256];
    DWORD bufferSize = sizeof(buffer);
    if (RegQueryValueEx(hKey, "ProcessorNameString", nullptr, nullptr, (LPBYTE)buffer,
                        &bufferSize) == ERROR_SUCCESS) {
      model_name_ = std::string(buffer);
    }

    // Get additional CPU info
    bufferSize = sizeof(buffer);
    if (RegQueryValueEx(hKey, "VendorIdentifier", nullptr, nullptr, (LPBYTE)buffer, &bufferSize) ==
        ERROR_SUCCESS) {
      vendor_ = std::string(buffer);
    }

    // Get family, model, stepping from registry
    bufferSize = sizeof(buffer);
    if (RegQueryValueEx(hKey, "Identifier", nullptr, nullptr, (LPBYTE)buffer, &bufferSize) ==
        ERROR_SUCCESS) {
      std::string identifier(buffer);
      // Parse "x86 Family X Model Y Stepping Z" format
      size_t family_pos = identifier.find("Family ");
      size_t model_pos = identifier.find("Model ");
      size_t stepping_pos = identifier.find("Stepping ");

      if (family_pos != std::string::npos) {
        family_ = std::stoi(identifier.substr(family_pos + 7));
      }
      if (model_pos != std::string::npos) {
        model_ = std::stoi(identifier.substr(model_pos + 6));
      }
      if (stepping_pos != std::string::npos) {
        stepping_ = std::stoi(identifier.substr(stepping_pos + 9));
      }
    }

    RegCloseKey(hKey);
  }

  // Set vendor (simple detection)
  if (model_name_.find("Intel") != std::string::npos) {
    vendor_ = "Intel";
  } else if (model_name_.find("AMD") != std::string::npos) {
    vendor_ = "AMD";
  } else if (vendor_.empty()) {
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

bool HardwareInfo::init_windows_frequency_wmi() {
  // Try to get more detailed frequency information using performance counters
  PDH_HQUERY hQuery;
  PDH_HCOUNTER hCounter;

  if (PdhOpenQuery(nullptr, 0, &hQuery) == ERROR_SUCCESS) {
    // Query processor frequency counter (using ANSI strings for MinGW
    // compatibility)
    if (PdhAddCounterA(hQuery, "\\Processor Information(_Total)\\Processor Frequency", 0,
                       &hCounter) == ERROR_SUCCESS) {
      if (PdhCollectQueryData(hQuery) == ERROR_SUCCESS) {
        PDH_FMT_COUNTERVALUE counterValue;
        if (PdhGetFormattedCounterValue(hCounter, PDH_FMT_DOUBLE, nullptr, &counterValue) ==
            ERROR_SUCCESS) {
          base_frequency_mhz_ = counterValue.doubleValue;
        }
      }
    }
    PdhCloseQuery(hQuery);
  }

  return true;
}

bool HardwareInfo::init_windows_memory_hierarchy() {
  // Use CPUID to get cache information
  int cpuInfo[4];

  // Intel cache information (leaf 4)
  for (int cache_id = 0; cache_id < 8; ++cache_id) {
    __cpuidex(cpuInfo, 4, cache_id);
    int cache_type = cpuInfo[0] & 0x1F;

    if (cache_type == 0)
      break; // No more caches

    if (cache_type == 1 || cache_type == 3) { // Data or Unified cache
      int level = (cpuInfo[0] >> 5) & 0x7;
      int ways = ((cpuInfo[1] >> 22) & 0x3FF) + 1;
      int partitions = ((cpuInfo[1] >> 12) & 0x3FF) + 1;
      int line_size = (cpuInfo[1] & 0xFFF) + 1;
      int sets = cpuInfo[2] + 1;

      MemoryHierarchy::CacheLevel cache;
      cache.size_kb = (ways * partitions * line_size * sets) / 1024;
      cache.line_size_bytes = line_size;
      cache.associativity = ways;
      cache.type = (cache_type == 1) ? "data" : "unified";

      if (level == 1) {
        memory_hierarchy_.l1_caches.push_back(cache);
      } else if (level == 2) {
        memory_hierarchy_.l2_caches.push_back(cache);
      } else if (level == 3) {
        memory_hierarchy_.l3_caches.push_back(cache);
      }
    }
  }

  // Get NUMA information
  ULONG highestNodeNumber = 0;
  if (GetNumaHighestNodeNumber(&highestNodeNumber)) {
    memory_hierarchy_.numa_nodes = highestNodeNumber + 1;

    for (ULONG node = 0; node <= highestNodeNumber; ++node) {
      ULONGLONG nodeAffinityMask = 0;
      if (GetNumaNodeProcessorMask(node, &nodeAffinityMask)) {
        std::vector<int> cpu_list;
        for (int cpu = 0; cpu < 64; ++cpu) {
          if (nodeAffinityMask & (1ULL << cpu)) {
            cpu_list.push_back(cpu);
          }
        }
        if (!cpu_list.empty()) {
          memory_hierarchy_.numa_cpu_map[node] = cpu_list;
        }
      }
    }
  }

  return true;
}

bool HardwareInfo::init_windows_container_detection() {
  // Check for Hyper-V VM (common for containers)
  int cpuInfo[4];
  __cpuid(cpuInfo, 1);

  // Check hypervisor present bit
  if (cpuInfo[2] & (1 << 31)) {
    is_virtualized_ = true;

    // Get hypervisor vendor
    __cpuid(cpuInfo, 0x40000000);
    char hypervisor_vendor[13] = {0};
    memcpy(hypervisor_vendor, &cpuInfo[1], 4);
    memcpy(hypervisor_vendor + 4, &cpuInfo[2], 4);
    memcpy(hypervisor_vendor + 8, &cpuInfo[3], 4);

    std::string vendor_str(hypervisor_vendor);
    if (vendor_str.find("Microsoft Hv") != std::string::npos) {
      // Likely Hyper-V, check for container indicators
      HKEY hKey;
      if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                       "SYSTEM\\CurrentControlSet\\Control\\Session "
                       "Manager\\Environment",
                       0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char buffer[256];
        DWORD bufferSize = sizeof(buffer);
        if (RegQueryValueEx(hKey, "CONTAINER_NAME", nullptr, nullptr, (LPBYTE)buffer,
                            &bufferSize) == ERROR_SUCCESS) {
          is_containerized_ = true;
        }
        RegCloseKey(hKey);
      }
    }
  }

  // Check for Windows containers by looking for specific registry keys
  HKEY hKey;
  if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                   "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Containers", 0, KEY_READ,
                   &hKey) == ERROR_SUCCESS) {
    is_containerized_ = true;
    RegCloseKey(hKey);
  }

  return true;
}

bool HardwareInfo::update_windows_perfcounters() {
  static PDH_HQUERY hQuery = nullptr;
  static PDH_HCOUNTER hTotalCounter = nullptr;
  static std::vector<PDH_HCOUNTER> hCoreCounters;
  static bool initialized = false;

  if (!initialized) {
    if (PdhOpenQuery(nullptr, 0, &hQuery) != ERROR_SUCCESS) {
      return false;
    }

    // Add total processor counter (using ANSI strings for MinGW compatibility)
    if (PdhAddCounterA(hQuery, "\\Processor(_Total)\\% Processor Time", 0, &hTotalCounter) !=
        ERROR_SUCCESS) {
      PdhCloseQuery(hQuery);
      return false;
    }

    // Add per-core counters
    hCoreCounters.resize(logical_cores_);
    for (int i = 0; i < logical_cores_; ++i) {
      std::string counterPath = "\\Processor(" + std::to_string(i) + ")\\% Processor Time";
      if (PdhAddCounterA(hQuery, counterPath.c_str(), 0, &hCoreCounters[i]) != ERROR_SUCCESS) {
        // Continue even if some counters fail
        hCoreCounters[i] = nullptr;
      }
    }

    // First collection to establish baseline
    PdhCollectQueryData(hQuery);
    Sleep(100); // Wait a bit before second collection
    initialized = true;
  }

  if (PdhCollectQueryData(hQuery) != ERROR_SUCCESS) {
    return false;
  }

  // Get total CPU utilization
  PDH_FMT_COUNTERVALUE counterValue;
  if (PdhGetFormattedCounterValue(hTotalCounter, PDH_FMT_DOUBLE, nullptr, &counterValue) ==
      ERROR_SUCCESS) {
    overall_utilization_ = 100.0 - counterValue.doubleValue; // PDH returns % idle time
  }

  // Get per-core utilization
  for (int i = 0; i < logical_cores_; ++i) {
    if (hCoreCounters[i] && PdhGetFormattedCounterValue(hCoreCounters[i], PDH_FMT_DOUBLE, nullptr,
                                                        &counterValue) == ERROR_SUCCESS) {
      if (i < static_cast<int>(cores_.size())) {
        cores_[i].utilization_percent = 100.0 - counterValue.doubleValue; // PDH returns % idle time
      }
    }
  }

  return true;
}

bool HardwareInfo::read_thermal_windows() {
  // Windows thermal information is typically not exposed through standard APIs
  // This would require WMI queries to MSAcpi_ThermalZoneTemperature class
  // For now, set thermal information as unavailable
  thermal_info_.current_temp_celsius = -1.0; // Indicates unavailable
  thermal_info_.thermal_throttling = false;

  // Try to detect thermal throttling by monitoring frequency drops
  static double prev_freq = 0.0;
  static int throttle_count = 0;

  if (base_frequency_mhz_ > 0 && prev_freq > 0) {
    double freq_drop = (prev_freq - base_frequency_mhz_) / prev_freq;
    if (freq_drop > 0.1) { // More than 10% frequency drop
      throttle_count++;
      if (throttle_count > 3) {
        thermal_info_.thermal_throttling = true;
      }
    } else {
      throttle_count = 0;
      thermal_info_.thermal_throttling = false;
    }
  }
  prev_freq = base_frequency_mhz_;

  return true;
}

bool HardwareInfo::read_frequencies_windows() {
  // Query current processor frequency using performance counters
  PDH_HQUERY hQuery;
  PDH_HCOUNTER hCounter;

  if (PdhOpenQuery(nullptr, 0, &hQuery) == ERROR_SUCCESS) {
    if (PdhAddCounterA(hQuery, "\\Processor Information(_Total)\\Processor Frequency", 0,
                       &hCounter) == ERROR_SUCCESS) {
      if (PdhCollectQueryData(hQuery) == ERROR_SUCCESS) {
        PDH_FMT_COUNTERVALUE counterValue;
        if (PdhGetFormattedCounterValue(hCounter, PDH_FMT_DOUBLE, nullptr, &counterValue) ==
            ERROR_SUCCESS) {
          // Update all cores with the same frequency (Windows doesn't easily
          // expose per-core frequencies)
          double current_freq = counterValue.doubleValue;
          for (auto &core : cores_) {
            core.current_freq_mhz = current_freq;
            core.governor = "windows"; // Windows manages frequency automatically
          }
        }
      }
    }
    PdhCloseQuery(hQuery);
  }

  return true;
}
#endif

// RAM-related methods implementation

double HardwareInfo::get_memory_utilization_percent() const {
  if (ram_info_.total_physical_memory_bytes > 0) {
    return (static_cast<double>(ram_info_.used_memory_bytes) /
            ram_info_.total_physical_memory_bytes) *
           100.0;
  }
  return 0.0;
}

bool HardwareInfo::update_ram_usage() {
#ifdef __linux__
  return read_meminfo_linux();
#elif defined(_WIN32)
  return update_ram_usage_windows();
#elif defined(__APPLE__)
  return update_macos_ram_usage();
#else
  return false;
#endif
}

bool HardwareInfo::read_meminfo_linux() {
#ifdef __linux__
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo.is_open()) {
    return false;
  }

  std::string line;
  while (std::getline(meminfo, line)) {
    std::istringstream iss(line);
    std::string key, value_str;
    long long value;

    if (iss >> key >> value) {
      value *= 1024; // Convert from KB to bytes

      if (key == "MemTotal:") {
        ram_info_.total_physical_memory_bytes = value;
      } else if (key == "MemAvailable:") {
        ram_info_.available_memory_bytes = value;
      } else if (key == "MemFree:") {
        ram_info_.free_memory_bytes = value;
      } else if (key == "Cached:") {
        ram_info_.cached_memory_bytes = value;
      } else if (key == "Buffers:") {
        ram_info_.buffer_memory_bytes = value;
      } else if (key == "SwapTotal:") {
        ram_info_.swap_total_bytes = value;
      } else if (key == "SwapFree:") {
        ram_info_.swap_free_bytes = value;
      }
    }
  }

  // Calculate used memory
  ram_info_.used_memory_bytes =
      ram_info_.total_physical_memory_bytes - ram_info_.available_memory_bytes;
  ram_info_.swap_used_bytes = ram_info_.swap_total_bytes - ram_info_.swap_free_bytes;

  return true;
#else
  return false;
#endif
}

bool HardwareInfo::read_ram_modules_linux() {
#ifdef __linux__
  // Try to read from dmidecode if available (requires root)
  // For now, set basic info from /proc/meminfo

  // Try to determine memory type and speed from /sys/bus/memory_technology_device
  // This is platform-specific and may not always be available

  // Read memory channels from CPU topology
  ram_info_.memory_channels = memory_channels_;

  // For basic systems, estimate one module if we have total memory
  if (ram_info_.total_physical_memory_bytes > 0 && ram_info_.modules.empty()) {
    RamModule module;
    module.size_bytes = ram_info_.total_physical_memory_bytes;
    module.type = "DDR4";    // Default assumption
    module.speed_mhz = 2400; // Default assumption
    ram_info_.modules.push_back(module);
    ram_info_.populated_slots = 1;
    ram_info_.total_slots = 4; // Common default
  }

  return true;
#else
  return false;
#endif
}

#ifdef _WIN32
bool HardwareInfo::init_windows_ram_info() {
  // Get basic memory info using GlobalMemoryStatusEx
  MEMORYSTATUSEX memStatus;
  memStatus.dwLength = sizeof(memStatus);

  if (GlobalMemoryStatusEx(&memStatus)) {
    ram_info_.total_physical_memory_bytes = memStatus.ullTotalPhys;
    ram_info_.available_memory_bytes = memStatus.ullAvailPhys;
    ram_info_.used_memory_bytes =
        ram_info_.total_physical_memory_bytes - ram_info_.available_memory_bytes;
  }

  // Try to get more detailed info using WMI (Win32_PhysicalMemory)
  // This would require WMI initialization which is complex
  // For now, use basic info

  return true;
}

bool HardwareInfo::update_ram_usage_windows() {
  MEMORYSTATUSEX memStatus;
  memStatus.dwLength = sizeof(memStatus);

  if (GlobalMemoryStatusEx(&memStatus)) {
    ram_info_.available_memory_bytes = memStatus.ullAvailPhys;
    ram_info_.used_memory_bytes =
        ram_info_.total_physical_memory_bytes - ram_info_.available_memory_bytes;
    return true;
  }

  return false;
}

bool HardwareInfo::read_ram_modules_windows() {
  // Would require WMI queries to Win32_PhysicalMemory
  // Complex implementation - for now return basic success
  return true;
}
#endif

#ifdef __APPLE__
bool HardwareInfo::init_macos_ram_info() {
  // Get total memory
  size_t size = sizeof(ram_info_.total_physical_memory_bytes);
  sysctlbyname("hw.memsize", &ram_info_.total_physical_memory_bytes, &size, NULL, 0);

  return update_macos_ram_usage();
}

bool HardwareInfo::update_macos_ram_usage() {
  vm_statistics64_data_t vm_stat;
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stat, &count) ==
      KERN_SUCCESS) {
    vm_size_t page_size;
    size_t size = sizeof(page_size);
    sysctlbyname("hw.pagesize", &page_size, &size, NULL, 0);

    ram_info_.free_memory_bytes = vm_stat.free_count * page_size;
    ram_info_.used_memory_bytes =
        ram_info_.total_physical_memory_bytes - ram_info_.free_memory_bytes;
    ram_info_.available_memory_bytes =
        ram_info_.free_memory_bytes + (vm_stat.inactive_count * page_size);

    return true;
  }

  return false;
}
#endif

} // namespace utils
