#include "device/cpu/cpu_context.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace tdevice {
CPUContext::CPUContext() : Context() {}

size_t CPUContext::getTotalMemory() const {
#ifdef __linux__
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo.is_open()) {
    return 0;
  }

  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.substr(0, 9) == "MemTotal:") {
      std::istringstream iss(line);
      std::string key;
      size_t value;
      std::string unit;

      if (iss >> key >> value >> unit) {
        // Value is in kB, convert to bytes
        return value * 1024;
      }
    }
  }
  return 0;
#elif defined(_WIN32)
  // Windows implementation
  MEMORYSTATUSEX memStatus;
  memStatus.dwLength = sizeof(memStatus);
  if (GlobalMemoryStatusEx(&memStatus)) {
    return static_cast<size_t>(memStatus.ullTotalPhys);
  }
  return 0;
#elif defined(__APPLE__)
  // macOS implementation
  int64_t physical_memory;
  size_t length = sizeof(physical_memory);
  if (sysctlbyname("hw.memsize", &physical_memory, &length, nullptr, 0) == 0) {
    return static_cast<size_t>(physical_memory);
  }
  return 0;
#else
  // Fallback for other platforms
  return 0;
#endif
}

size_t CPUContext::getAvailableMemory() const {
#ifdef __linux__
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo.is_open()) {
    return 0;
  }

  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.substr(0, 13) == "MemAvailable:") {
      std::istringstream iss(line);
      std::string key;
      size_t value;
      std::string unit;

      if (iss >> key >> value >> unit) {
        // Value is in kB, convert to bytes
        return value * 1024;
      }
    }
  }
  return 0;
#elif defined(_WIN32)
  // Windows implementation
  MEMORYSTATUSEX memStatus;
  memStatus.dwLength = sizeof(memStatus);
  if (GlobalMemoryStatusEx(&memStatus)) {
    return static_cast<size_t>(memStatus.ullAvailPhys);
  }
  return 0;
#elif defined(__APPLE__)
  // macOS implementation
  vm_size_t page_size;
  vm_statistics64_data_t vm_stat;
  mach_msg_type_number_t host_size = sizeof(vm_stat) / sizeof(natural_t);

  host_page_size(mach_host_self(), &page_size);
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size) ==
      KERN_SUCCESS) {
    return static_cast<size_t>(vm_stat.free_count * page_size);
  }
  return 0;
#else
  // Fallback for other platforms
  return 0;
#endif
}

void *CPUContext::allocateMemory(size_t size) { return std::malloc(size); }

void CPUContext::deallocateMemory(void *ptr) { std::free(ptr); }

void CPUContext::copyToDevice(void *dest, const void *src, size_t size) {
  std::memcpy(dest, src, size);
}

void CPUContext::copyToHost(void *dest, const void *src, size_t size) {
  std::memcpy(dest, src, size);
}

} // namespace tdevice