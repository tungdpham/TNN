#include "pipeline/network_stage_worker.hpp"
#include <cstdlib>
#include <iostream>
#include <mkl.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>
int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <listen_port>" << std::endl;
    std::cerr << "Example: " << argv[0] << " 8001" << std::endl;
    return 1;
  }

  int listen_port = std::atoi(argv[1]);

  if (listen_port <= 0 || listen_port > 65535) {
    std::cerr << "Invalid port number: " << listen_port << std::endl;
    return 1;
  }

#ifdef USE_TBB
  tbb::task_arena arena(tbb::task_arena::constraints{}.set_max_concurrency(8));

  std::cout << "TBB max threads limited to: " << arena.max_concurrency() << std::endl;
#ifdef USE_MKL
  // set MKL to use TBB threading layer
  std::cout << "Setting MKL number of threads to: " << arena.max_concurrency() << std::endl;
  mkl_set_threading_layer(MKL_THREADING_TBB);
#endif
  arena.execute([&] {
#endif
    tpipeline::StandaloneNetworkWorker<float>::run_worker(listen_port);
#ifdef USE_TBB
  });
#endif
  return 0;
}
