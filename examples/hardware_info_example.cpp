#include "utils/hardware_info.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

using namespace tnn;
using namespace std;

int main() {
  try {
    cout << "=== TNN Hardware Information Test ===" << endl;

    HardwareInfo hardware_info;

    filesystem::create_directories("./logs");
    filesystem::create_directories("./temp");

    if (!hardware_info.initialize()) {
      cerr << "Failed to initialize CPU information!" << endl;
      return 1;
    }

    cout << "CPU information initialized successfully!" << endl;

    hardware_info.print_info();

    cout << "\n=== Writing CPU specs to JSON file ===" << endl;
    ofstream json_file("./temp/hardware_info.json");
    if (json_file.is_open()) {
      string json_output = hardware_info.to_json();

      size_t last_comma = json_output.find_last_of(',');
      if (last_comma != string::npos) {
        size_t last_brace = json_output.find_last_of('}');
        if (last_brace != string::npos && last_comma > last_brace - 10) {
          json_output.erase(last_comma, 1);
        }
      }
      json_file << json_output << endl;
      json_file.close();
      cout << "CPU specifications written to ./temp/hardware_info.json" << endl;
    } else {
      cerr << "Failed to open ./temp/hardware_info.json for writing!" << endl;
    }

    ofstream csv_file("./logs/cpu_status.csv");
    if (csv_file.is_open()) {
      csv_file << "timestamp,cpu_utilization_percent,temperature_celsius,"
                  "update_number"
               << endl;
      cout << "Created CSV file: ./logs/cpu_status.csv" << endl;
    } else {
      cerr << "Failed to create ./logs/cpu_status.csv!" << endl;
      return 1;
    }

    cout << "\n=== Updating dynamic information ===" << endl;
    if (hardware_info.update_dynamic_info()) {
      cout << "Dynamic info updated successfully!" << endl;
    }

    cout << "\n=== Distributed Computing Recommendations ===" << endl;

    auto numa_cores = hardware_info.get_numa_aware_cores();
    cout << "NUMA topology:" << endl;
    for (const auto &[node, cores] : numa_cores) {
      cout << "  NUMA Node " << node << ": ";
      for (size_t i = 0; i < cores.size(); ++i) {
        cout << cores[i];
        if (i < cores.size() - 1)
          cout << ", ";
      }
      cout << endl;
    }

    cout << "\n=== JSON Output (displayed) ===" << endl;
    cout << hardware_info.to_json() << endl;

    cout << "\n=== Dynamic Monitoring===" << endl;
    while (true) {
      this_thread::sleep_for(chrono::seconds(1));

      if (hardware_info.update_dynamic_info()) {
        auto now = chrono::system_clock::now();
        auto time_t = chrono::system_clock::to_time_t(now);
        auto ms = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()) % 1000;

        stringstream timestamp;
        timestamp << put_time(localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        timestamp << "." << setfill('0') << setw(3) << ms.count();

        double cpu_util = hardware_info.get_overall_utilization();
        double temp = hardware_info.get_thermal_info().current_temp_celsius;

        csv_file << timestamp.str() << "," << cpu_util << "," << temp << endl;

        cout << "Update: " << "CPU: " << cpu_util << "%, " << "Temp: " << temp << "°C, "
             << "Timestamp: " << timestamp.str() << endl;
      }
    }

    csv_file.close();
    cout << "\nCPU status data written to ./logs/cpu_status.csv" << endl;

    return 0;
  } catch (const exception &ex) {
    cerr << "Exception occurred: " << ex.what() << endl;
    return 1;
  }
}
