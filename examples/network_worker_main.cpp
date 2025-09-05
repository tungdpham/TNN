#include "pipeline/network_stage_worker.hpp"
#include <cstdlib>
#include <iostream>
#include "utils/misc.hpp"

int main(int argc, char *argv[]) {
  std::cout.tie(nullptr);
  std::cin.tie(nullptr);
  std::ios::sync_with_stdio(false);

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <listen_port>" << std::endl;
    std::cerr << "Example: " << argv[0] << " 8001" << std::endl;
    return 1;
  }

  utils::set_num_threads(8);

  int listen_port = std::atoi(argv[1]);

  if (listen_port <= 0 || listen_port > 65535) {
    std::cerr << "Invalid port number: " << listen_port << std::endl;
    return 1;
  }

  tpipeline::StandaloneNetworkWorker<float>::run_worker(listen_port);
  return 0;
}
