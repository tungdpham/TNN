#include "distributed/roce_worker.hpp"

#include <getopt.h>

#include <iostream>
#include <string>

#include "distributed/endpoint.hpp"

using namespace tnn;
using namespace std;

struct Config {
  std::string host = "localhost";
  int port = 0;
  std::string device_name;
  int gid_index = -1;
  bool use_gpu = false;
};

void print_usage(const char *program_name) {
  cout << "Usage: " << program_name << " [options]" << endl;
  cout << endl;
  cout << "Options:" << endl;
  cout << "  --host <address>       Hostname or IP to bind to (required)" << endl;
  cout << "  --port <number>        TCP port for initial connection (required)" << endl;
  cout << "  --device <name>        IB device name (e.g., mlx5_0) (required)" << endl;
  cout << "  --gid-index <index>    GID index for RoCE (required)" << endl;
  cout << "  --gpu                  Enable GPU offloading" << endl;
  cout << "  -h, --help             Show this help message" << endl;
}

bool parse_arguments(int argc, char *argv[], Config &cfg) {
  int c;

  static struct option long_options[] = {{"host", required_argument, 0, 'H'},
                                         {"port", required_argument, 0, 'p'},
                                         {"device", required_argument, 0, 'd'},
                                         {"gid-index", required_argument, 0, 'g'},
                                         {"gpu", no_argument, 0, 'G'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  while ((c = getopt_long(argc, argv, "H:p:d:g:Gh", long_options, nullptr)) != -1) {
    switch (c) {
      case 'H':
        cfg.host = optarg;
        break;
      case 'p':
        try {
          cfg.port = stoi(optarg);
        } catch (...) {
          cerr << "Invalid port value: " << optarg << endl;
          return false;
        }
        break;
      case 'd':
        cfg.device_name = optarg;
        break;
      case 'g':
        try {
          cfg.gid_index = stoi(optarg);
        } catch (...) {
          cerr << "Invalid gid-index value: " << optarg << endl;
          return false;
        }
        break;
      case 'G':
        cfg.use_gpu = true;
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

  if (cfg.host.empty()) {
    cerr << "Missing required argument: --host" << endl;
    print_usage(argv[0]);
    return false;
  }
  if (cfg.port <= 0 || cfg.port > 65535) {
    cerr << "Invalid or missing port: " << cfg.port << endl;
    print_usage(argv[0]);
    return false;
  }
  if (cfg.device_name.empty()) {
    cerr << "Missing required argument: --device" << endl;
    print_usage(argv[0]);
    return false;
  }
  if (cfg.gid_index < 0) {
    cout << "Since gid-index is not specified, auto-selecting GID index." << endl;
  }

  return true;
}

int main(int argc, char *argv[]) {
  Config cfg;

  if (!parse_arguments(argc, argv, cfg)) {
    return 1;
  }

  try {
    Endpoint worker_endpoint = Endpoint::roce(cfg.host, cfg.port, cfg.device_name, cfg.gid_index);
    RoceWorker worker(worker_endpoint, cfg.use_gpu);
    worker.start();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
