#include "pipeline/network_stage_worker.hpp"
#include <cstdlib>
#include <iostream>
#include <string>
#ifdef USE_MKL
#include <mkl.h>
#endif
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>

constexpr int MAX_THREADS = 16;

void print_usage(const char *program_name) {
  std::cout << "Usage: " << program_name << " <listen_port> [options]" << std::endl;
  std::cout << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --ecore           Enable E-core affinity for energy efficiency" << std::endl;
  std::cout << "  --max-ecores <N>  Maximum number of E-cores to use (default: all)" << std::endl;
  std::cout << "  --show-cores      Display CPU core topology and exit" << std::endl;
  std::cout << "  --help           Show this help message" << std::endl;
  std::cout << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout << "  " << program_name << " 8001                    # Default mode" << std::endl;
  std::cout << "  " << program_name << " 8001 --ecore            # Use E-cores for efficiency"
            << std::endl;
  std::cout << "  " << program_name << " 8001 --ecore --max-ecores 2  # Use maximum 2 E-cores"
            << std::endl;
  std::cout << "  " << program_name << " 8001 --show-cores       # Show CPU topology" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  // Parse command line arguments
  int listen_port = 0;
  bool use_ecore_affinity = false;
  int max_ecore_threads = -1;
  bool show_cores_only = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--ecore") {
      use_ecore_affinity = true;
    } else if (arg == "--max-ecores") {
      if (i + 1 < argc) {
        max_ecore_threads = std::atoi(argv[++i]);
        if (max_ecore_threads <= 0) {
          std::cerr << "Invalid max-ecores value: " << argv[i] << std::endl;
          return 1;
        }
      } else {
        std::cerr << "--max-ecores requires a number argument" << std::endl;
        return 1;
      }
    } else if (arg == "--show-cores") {
      show_cores_only = true;
    } else if (listen_port == 0) {
      // First non-flag argument is the port
      listen_port = std::atoi(arg.c_str());
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      print_usage(argv[0]);
      return 1;
    }
  }

  if (listen_port <= 0 || listen_port > 65535) {
    std::cerr << "Invalid port number: " << listen_port << std::endl;
    return 1;
  }

  // Show core topology if requested
  if (show_cores_only) {
    utils::HardwareInfo hw_info;
    if (hw_info.initialize()) {
      utils::ThreadAffinity affinity(hw_info);
      affinity.print_affinity_info();
    } else {
      std::cerr << "Failed to initialize hardware info" << std::endl;
      return 1;
    }
    return 0;
  }

  std::cout << "Network Stage Worker Configuration" << std::endl;
  std::cout << "Listen port: " << listen_port << std::endl;
  std::cout << "E-core affinity: " << (use_ecore_affinity ? "Enabled" : "Disabled") << std::endl;
  if (use_ecore_affinity) {
    std::cout << "Max E-cores: "
              << (max_ecore_threads == -1 ? "All available" : std::to_string(max_ecore_threads))
              << std::endl;
  }
#ifdef USE_TBB
  // get system max threads
  int max_threads = std::thread::hardware_concurrency();
  max_threads = std::min(max_threads, MAX_THREADS);
  tbb::task_arena arena(tbb::task_arena::constraints{}.set_max_concurrency(max_threads));

  std::cout << "TBB max threads limited to: " << arena.max_concurrency() << std::endl;
#ifdef USE_MKL
  // set MKL to use TBB threading layer
  std::cout << "Setting MKL number of threads to: " << arena.max_concurrency() << std::endl;
  mkl_set_threading_layer(MKL_THREADING_TBB);
#endif
  arena.execute([&] {
#endif
    tpipeline::StandaloneNetworkWorker<float>::run_worker(listen_port, use_ecore_affinity,
                                                          max_ecore_threads);
#ifdef USE_TBB
  });
#endif
  return 0;
}