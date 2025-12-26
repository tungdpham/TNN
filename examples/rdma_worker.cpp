#include "distributed/rdma_worker.hpp"
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
  std::string host = "0.0.0.0";
  int ib_port = 1;
  int gid_index = 0;
  bool use_ecore_affinity = false;
  int max_ecore_threads = -1;
  bool show_cores_only = false;
  bool use_gpu = false;
  size_t num_threads = 8;
};

void print_usage(const char *program_name) {
  cout << "Usage: " << program_name << " [options] <listen_port>" << endl;
  cout << endl;
  cout << "Options:" << endl;
  cout << "  --host <IP>        Host IP to bind for OOB (default: 0.0.0.0)" << endl;
  cout << "  --ib-port <N>      InfiniBand port number (default: 1)" << endl;
  cout << "  --gid-index <N>    GID index for RoCE (default: 0)" << endl;
  cout << "  --ecore            Enable E-core affinity for energy efficiency" << endl;
  cout << "  --max-ecores <N>   Maximum number of E-cores to use (default: all)" << endl;
  cout << "  --show-cores       Display CPU core topology and exit" << endl;
  cout << "  --gpu              Enable GPU offloading for processing" << endl;
  cout << "  --num-threads <N>  Number of worker threads for processing (default: 8)" << endl;
  cout << "  -h, --help         Show this help message" << endl;
  cout << endl;
  cout << "Examples:" << endl;
  cout << "  " << program_name << " 8001                      # Default mode" << endl;
  cout << "  " << program_name << " --ib-port 1 --gid-index 3 8001 # RDMA config" << endl;
}

bool parse_arguments(int argc, char *argv[], Config &cfg) {
  int c;

  static struct option long_options[] = {{"host", required_argument, 0, 'H'},
                                         {"ib-port", required_argument, 0, 'p'},
                                         {"gid-index", required_argument, 0, 'x'},
                                         {"ecore", no_argument, 0, 'e'},
                                         {"max-ecores", required_argument, 0, 'm'},
                                         {"show-cores", no_argument, 0, 's'},
                                         {"gpu", no_argument, 0, 'g'},
                                         {"num-threads", required_argument, 0, 'n'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  optind = 1;

  while ((c = getopt_long(argc, argv, "H:p:x:em:sgn:h", long_options, nullptr)) != -1) {
    switch (c) {
    case 'H':
      cfg.host = optarg;
      break;
    case 'p':
      try {
        cfg.ib_port = stoi(optarg);
      } catch (...) {
        cerr << "Invalid ib-port" << endl;
        return false;
      }
      break;
    case 'x':
      try {
        cfg.gid_index = stoi(optarg);
      } catch (...) {
        cerr << "Invalid gid-index" << endl;
        return false;
      }
      break;
    case 'e':
      cfg.use_ecore_affinity = true;
      break;
    case 'm':
      try {
        cfg.max_ecore_threads = stoi(optarg);
      } catch (...) {
        cerr << "Invalid max-ecores" << endl;
        return false;
      }
      break;
    case 's':
      cfg.show_cores_only = true;
      break;
    case 'g':
      cfg.use_gpu = true;
      break;
    case 'n':
      try {
        cfg.num_threads = stoi(optarg);
      } catch (...) {
        cerr << "Invalid num-threads" << endl;
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
        return false;
      }
    } else {
      cerr << "Missing required argument: <listen_port>" << endl;
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
    }
    return 0;
  }

  cout << "RDMA Network Stage Worker Configuration" << endl;
  cout << "Listen port: " << cfg.listen_port << endl;
  cout << "IB Port: " << cfg.ib_port << ", GID Index: " << cfg.gid_index << endl;
  cout << "Worker threads: " << cfg.num_threads << endl;

  Endpoint endpoint = Endpoint::network(cfg.host, cfg.listen_port);
  endpoint.set_parameter("ib_port", std::to_string(cfg.ib_port));
  endpoint.set_parameter("gid_index", std::to_string(cfg.gid_index));

  ThreadWrapper thread_wrapper({static_cast<unsigned int>(cfg.num_threads)});

  thread_wrapper.execute([&]() {
    RdmaNetworkStageWorker<float> worker(endpoint, cfg.use_gpu, cfg.use_ecore_affinity,
                                         cfg.max_ecore_threads);
    worker.start();
  });

  return 0;
}
