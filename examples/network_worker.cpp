#include "pipeline/network_stage_worker.hpp"
#include "threading/thread_wrapper.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

using namespace tnn;

constexpr int MAX_THREADS = 4;
using namespace std;

void print_usage(const char *program_name) {
  cout << "Usage: " << program_name << " <listen_port> [options]" << endl;
  cout << endl;
  cout << "Options:" << endl;
  cout << "  --ecore           Enable E-core affinity for energy efficiency" << endl;
  cout << "  --max-ecores <N>  Maximum number of E-cores to use (default: all)" << endl;
  cout << "  --show-cores      Display CPU core topology and exit" << endl;
  cout << "  --help           Show this help message" << endl;
  cout << endl;
  cout << "Examples:" << endl;
  cout << "  " << program_name << " 8001                    # Default mode" << endl;
  cout << "  " << program_name << " 8001 --ecore            # Use E-cores for efficiency" << endl;
  cout << "  " << program_name << " 8001 --ecore --max-ecores 2  # Use maximum 2 E-cores" << endl;
  cout << "  " << program_name << " 8001 --show-cores       # Show CPU topology" << endl;
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
    string arg = argv[i];

    if (arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--ecore") {
      use_ecore_affinity = true;
    } else if (arg == "--max-ecores") {
      if (i + 1 < argc) {
        max_ecore_threads = atoi(argv[++i]);
        if (max_ecore_threads <= 0) {
          cerr << "Invalid max-ecores value: " << argv[i] << endl;
          return 1;
        }
      } else {
        cerr << "--max-ecores requires a number argument" << endl;
        return 1;
      }
    } else if (arg == "--show-cores") {
      show_cores_only = true;
    } else if (listen_port == 0) {
      // First non-flag argument is the port
      listen_port = atoi(arg.c_str());
    } else {
      cerr << "Unknown argument: " << arg << endl;
      print_usage(argv[0]);
      return 1;
    }
  }

  if (listen_port <= 0 || listen_port > 65535) {
    cerr << "Invalid port number: " << listen_port << endl;
    return 1;
  }

  // Show core topology if requested
  if (show_cores_only) {
    HardwareInfo hw_info;
    if (hw_info.initialize()) {
      ThreadAffinity affinity(hw_info);
      affinity.print_affinity_info();
    } else {
      cerr << "Failed to initialize hardware info" << endl;
      return 1;
    }
    return 0;
  }

  cout << "Network Stage Worker Configuration" << endl;
  cout << "Listen port: " << listen_port << endl;
  cout << "E-core affinity: " << (use_ecore_affinity ? "Enabled" : "Disabled") << endl;
  if (use_ecore_affinity) {
    cout << "Max E-cores: "
         << (max_ecore_threads == -1 ? "All available" : to_string(max_ecore_threads)) << endl;
  }
  ThreadWrapper thread_wrapper(
      {MAX_THREADS}); // wrapper to cleanly manage branching with different backends

  thread_wrapper.execute([&]() {
    NetworkStageWorker worker(listen_port, use_ecore_affinity, max_ecore_threads);
    worker.start();
  });
  return 0;
}