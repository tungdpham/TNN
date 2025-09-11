#include "utils/cpu_info.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

int main() {
  try {
    std::cout << "=== DCNN CPU Information Test ===" << std::endl;

    utils::CpuInfo cpu_info;

    // Create directories if they don't exist
    std::filesystem::create_directories("./logs");
    std::filesystem::create_directories("./temp");

    // Initialize CPU information
    if (!cpu_info.initialize()) {
      std::cerr << "Failed to initialize CPU information!" << std::endl;
      return 1;
    }

    std::cout << "CPU information initialized successfully!" << std::endl;

    // Print static information
    cpu_info.print_info();

    // Write JSON output to file
    std::cout << "\n=== Writing CPU specs to JSON file ===" << std::endl;
    std::ofstream json_file("./temp/cpu_info.json");
    if (json_file.is_open()) {
      std::string json_output = cpu_info.to_json();
      // Remove trailing comma if present to make valid JSON
      size_t last_comma = json_output.find_last_of(',');
      if (last_comma != std::string::npos) {
        size_t last_brace = json_output.find_last_of('}');
        if (last_brace != std::string::npos && last_comma > last_brace - 10) {
          json_output.erase(last_comma, 1);
        }
      }
      json_file << json_output << std::endl;
      json_file.close();
      std::cout << "CPU specifications written to ./temp/cpu_info.json"
                << std::endl;
    } else {
      std::cerr << "Failed to open ./temp/cpu_info.json for writing!"
                << std::endl;
    }

    // Create CSV file with headers
    std::ofstream csv_file("./logs/cpu_status.csv");
    if (csv_file.is_open()) {
      csv_file << "timestamp,cpu_utilization_percent,temperature_celsius,"
                  "update_number"
               << std::endl;
      std::cout << "Created CSV file: ./logs/cpu_status.csv" << std::endl;
    } else {
      std::cerr << "Failed to create ./logs/cpu_status.csv!" << std::endl;
      return 1;
    }

    // Update dynamic information and print again
    std::cout << "\n=== Updating dynamic information ===" << std::endl;
    if (cpu_info.update_dynamic_info()) {
      std::cout << "Dynamic info updated successfully!" << std::endl;
    }

    // Test distributed computing utilities
    std::cout << "\n=== Distributed Computing Recommendations ===" << std::endl;

    // NUMA information
    auto numa_cores = cpu_info.get_numa_aware_cores();
    std::cout << "NUMA topology:" << std::endl;
    for (const auto &[node, cores] : numa_cores) {
      std::cout << "  NUMA Node " << node << ": ";
      for (size_t i = 0; i < cores.size(); ++i) {
        std::cout << cores[i];
        if (i < cores.size() - 1)
          std::cout << ", ";
      }
      std::cout << std::endl;
    }

    // JSON output (display only)
    std::cout << "\n=== JSON Output (displayed) ===" << std::endl;
    std::cout << cpu_info.to_json() << std::endl;

    // Demonstrate dynamic monitoring and write to CSV
    std::cout << "\n=== Dynamic Monitoring===" << std::endl;
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(1));

      if (cpu_info.update_dynamic_info()) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;

        std::stringstream timestamp;
        timestamp << std::put_time(std::localtime(&time_t),
                                   "%Y-%m-%d %H:%M:%S");
        timestamp << "." << std::setfill('0') << std::setw(3) << ms.count();

        double cpu_util = cpu_info.get_overall_utilization();
        double temp = cpu_info.get_thermal_info().current_temp_celsius;

        // Write to CSV
        csv_file << timestamp.str() << "," << cpu_util << "," << temp
                 << std::endl;

        std::cout << "Update: " << "CPU: " << cpu_util << "%, "
                  << "Temp: " << temp << "°C, "
                  << "Timestamp: " << timestamp.str() << std::endl;
      }
    }

    csv_file.close();
    std::cout << "\nCPU status data written to ./logs/cpu_status.csv"
              << std::endl;

    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "Exception occurred: " << ex.what() << std::endl;
    return 1;
  }
}
