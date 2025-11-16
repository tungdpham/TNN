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
  long vmsize = 0;
  while (std::getline(file, line)) {
    if (line.rfind("VmSize:", 0) == 0) { // Check if the line starts with "VmSize:"
      std::string value_str = line.substr(8, line.size() - 8 - 3); // Extract the value
      vmsize = std::stoi(value_str);                               // Convert to integer
      break;
    }
  }
  return vmsize;
}

} // namespace tnn