#include "pipeline/network_stage_worker.hpp"
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <listen_port>" << std::endl;
    std::cerr << "Example: " << argv[0] << " 8001" << std::endl;
    return 1;
  }

#ifdef USE_TBB
  tbb::global_control c(tbb::global_control::max_allowed_parallelism, 8);
  std::cout << "tbb::global_control::active_value(max_allowed_parallelism): "
            << tbb::global_control::active_value(
                   tbb::global_control::max_allowed_parallelism)
            << "\n";
#endif

#ifdef _OPENMP
  const int num_threads = omp_get_max_threads();
  omp_set_num_threads(std::min(num_threads, 4));
  std::cout << "Number of OpenMP threads set to: " << omp_get_max_threads()
            << std::endl;
#endif

  int listen_port = std::atoi(argv[1]);

  if (listen_port <= 0 || listen_port > 65535) {
    std::cerr << "Invalid port number: " << listen_port << std::endl;
    return 1;
  }

  std::cout << "Using listen port: " << listen_port << std::endl;

  tpipeline::StandaloneNetworkWorker<float>::run_worker(listen_port);
  return 0;
}
