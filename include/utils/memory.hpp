#pragma once

#ifdef __unix__
#include <unistd.h>
#endif
#include <fstream>
#include <string>

namespace tnn {
inline long get_memory_usage_kb() {
  std::ifstream file("/proc/" + std::to_string(getpid()) + "/status");
  std::string line;
  long vmrss = 0;
  while (std::getline(file, line)) {
    if (line.rfind("VmRSS:", 0) == 0) { // Check if the line starts with "VmRSS:"
      std::string value_str = line.substr(7, line.size() - 7 - 3); // Extract the value
      vmrss = std::stoi(value_str);                                // Convert to integer
      break;
    }
  }
  return vmrss;
}

} // namespace tnn