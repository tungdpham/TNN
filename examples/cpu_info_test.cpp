#include "utils/cpu_info.hpp"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "=== DCNN CPU Information Test ===" << std::endl;
    
    utils::CpuInfo cpu_info;
    
    // Initialize CPU information
    if (!cpu_info.initialize()) {
        std::cerr << "Failed to initialize CPU information!" << std::endl;
        return 1;
    }
    
    std::cout << "CPU information initialized successfully!" << std::endl;
    
    // Print static information
    cpu_info.print_info();
    
    // Update dynamic information and print again
    std::cout << "\n=== Updating dynamic information ===" << std::endl;
    if (cpu_info.update_dynamic_info()) {
        std::cout << "Dynamic info updated successfully!" << std::endl;
    }
    
    // Test distributed computing utilities
    std::cout << "\n=== Distributed Computing Recommendations ===" << std::endl;
    
    int optimal_threads = cpu_info.get_optimal_thread_count();
    std::cout << "Optimal thread count for NN/CNN workloads: " << optimal_threads << std::endl;
    
    auto affinity = cpu_info.get_recommended_cpu_affinity(optimal_threads);
    std::cout << "Recommended CPU affinity for " << optimal_threads << " threads: ";
    for (size_t i = 0; i < affinity.size(); ++i) {
        std::cout << affinity[i];
        if (i < affinity.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    
    std::cout << "Suitable for heavy workload: " << 
        (cpu_info.is_suitable_for_heavy_workload() ? "Yes" : "No") << std::endl;
    
    std::cout << "Performance score: " << cpu_info.get_performance_score() << std::endl;
    
    // NUMA information
    auto numa_cores = cpu_info.get_numa_aware_cores();
    std::cout << "NUMA topology:" << std::endl;
    for (const auto& [node, cores] : numa_cores) {
        std::cout << "  NUMA Node " << node << ": ";
        for (size_t i = 0; i < cores.size(); ++i) {
            std::cout << cores[i];
            if (i < cores.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    
    // JSON output
    std::cout << "\n=== JSON Output ===" << std::endl;
    std::cout << cpu_info.to_json() << std::endl;
    
    // Demonstrate dynamic monitoring
    std::cout << "\n=== Dynamic Monitoring (5 seconds) ===" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        if (cpu_info.update_dynamic_info()) {
            std::cout << "Update " << (i + 1) << ": "
                      << "CPU: " << cpu_info.get_overall_utilization() << "%, "
                      << "Temp: " << cpu_info.get_thermal_info().current_temp_celsius << "°C, "
                      << "Suitable: " << (cpu_info.is_suitable_for_heavy_workload() ? "Yes" : "No")
                      << std::endl;
        }
    }
    
    return 0;
}
