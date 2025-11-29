#include "pipeline/network_stage_worker.hpp"
#include "threading/thread_wrapper.hpp"
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <string>
#include <unistd.h>

using namespace tnn;

using namespace std;

struct Config {
  int listen_port = 0;
  bool use_ecore_affinity = false;
  int max_ecore_threads = -1;
  bool show_cores_only = false;
  bool use_gpu = false;
  size_t io_threads = 4;
  size_t num_threads = 8;
};

void print_usage(const char *program_name) {
  cout << "Usage: " << program_name << " [options] <listen_port>" << endl;
  cout << endl;
  cout << "Options:" << endl;
  cout << "  --ecore            Enable E-core affinity for energy efficiency" << endl;
  cout << "  --max-ecores <N>   Maximum number of E-cores to use (default: all)" << endl;
  cout << "  --show-cores       Display CPU core topology and exit" << endl;
  cout << "  --gpu              Enable GPU offloading for processing" << endl;
  cout << "  --io-threads <N>   Number of IO threads for networking (default: 1)" << endl;
  cout << "  --num-threads <N>  Number of worker threads for processing (default: 8)" << endl;
  cout << "  -h, --help         Show this help message" << endl;
  cout << endl;
  cout << "Examples:" << endl;
  cout << "  " << program_name << " 8001                      # Default mode" << endl;
  cout << "  " << program_name << " --ecore 8001              # Use E-cores for efficiency" << endl;
  cout << "  " << program_name << " --max-ecores 2 --ecore 8001  # Use max 2 E-cores" << endl;
  cout << "  " << program_name << " --gpu 8001                # Enable GPU processing" << endl;
  cout << "  " << program_name << " --io-threads 4 8001       # Use 4 IO threads" << endl;
  cout << "  " << program_name << " --show-cores              # Show CPU topology" << endl;
}

bool parse_arguments(int argc, char *argv[], Config &cfg) {
  int c;

  static struct option long_options[] = {{"ecore", no_argument, 0, 'e'},
                                         {"max-ecores", required_argument, 0, 'm'},
                                         {"show-cores", no_argument, 0, 's'},
                                         {"gpu", no_argument, 0, 'g'},
                                         {"io-threads", required_argument, 0, 'i'},
                                         {"num-threads", required_argument, 0, 'n'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  optind = 1;

  while ((c = getopt_long(argc, argv, "h", long_options, nullptr)) != -1) {
    switch (c) {
    case 'e':
      cfg.use_ecore_affinity = true;
      break;
    case 'm':
      try {
        cfg.max_ecore_threads = stoi(optarg);
        if (cfg.max_ecore_threads <= 0) {
          cerr << "Invalid max-ecores value: " << optarg << endl;
          return false;
        }
      } catch (...) {
        cerr << "--max-ecores requires a valid number argument" << endl;
        return false;
      }
      break;
    case 's':
      cfg.show_cores_only = true;
      break;
    case 'g':
      cfg.use_gpu = true;
      break;
    case 'i':
      try {
        int threads = stoi(optarg);
        if (threads <= 0) {
          cerr << "Invalid io-threads value: " << optarg << endl;
          return false;
        }
        cfg.io_threads = static_cast<size_t>(threads);
      } catch (...) {
        cerr << "--io-threads requires a valid number argument" << endl;
        return false;
      }
      break;
    case 'n':
      try {
        int threads = stoi(optarg);
        if (threads <= 0) {
          cerr << "Invalid num-threads value: " << optarg << endl;
          return false;
        }
        cfg.num_threads = static_cast<size_t>(threads);
      } catch (...) {
        cerr << "--num-threads requires a valid number argument" << endl;
        return false;
      }
      break;
    case 'h':
      print_usage(argv[0]);
      return false;
    case '?':
      return false;
    default:
      return false;
    }
  }

  if (!cfg.show_cores_only) {
    if (optind < argc) {
      try {
        cfg.listen_port = stoi(argv[optind]);
      } catch (...) {
        cerr << "Invalid port argument: " << argv[optind] << endl;
        print_usage(argv[0]);
        return false;
      }

      if (optind + 1 < argc) {
        cerr << "Too many arguments. Only the port number is expected after options." << endl;
        print_usage(argv[0]);
        return false;
      }

    } else {
      cerr << "Missing required argument: <listen_port>" << endl;
      print_usage(argv[0]);
      return false;
    }

    if (cfg.listen_port <= 0 || cfg.listen_port > 65535) {
      cerr << "Invalid port number: " << cfg.listen_port << endl;
      return false;
    }
  } else {
    if (optind < argc) {
      cerr << "Unexpected argument '" << argv[optind] << "' provided with --show-cores." << endl;
      print_usage(argv[0]);
      return false;
    }
  }

  return true;
}

int main(int argc, char *argv[]) {
  Config cfg;

  if (!parse_arguments(argc, argv, cfg)) {
    return 1;
  }

  if (cfg.show_cores_only) {
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
  cout << "Listen port: " << cfg.listen_port << endl;
  cout << "E-core affinity: " << (cfg.use_ecore_affinity ? "Enabled" : "Disabled") << endl;
  if (cfg.use_ecore_affinity) {
    cout << "Max E-cores: "
         << (cfg.max_ecore_threads == -1 ? "All available" : to_string(cfg.max_ecore_threads))
         << endl;
  }
  cout << "GPU offloading: " << (cfg.use_gpu ? "Enabled" : "Disabled") << endl;
  cout << "IO threads: " << cfg.io_threads << endl;
  cout << "Worker threads: " << cfg.num_threads << endl;

  ThreadWrapper thread_wrapper({static_cast<unsigned int>(cfg.num_threads)});

  thread_wrapper.execute([&]() {
    NetworkStageWorker worker(cfg.listen_port, cfg.use_gpu, cfg.use_ecore_affinity,
                              cfg.max_ecore_threads, cfg.io_threads);
    worker.start();
  });

  return 0;
}