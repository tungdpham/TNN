#include "pipeline/network_stage_worker.hpp"
#include <iostream>
#include <cstdlib>
#include <omp.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <listen_port>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 8001" << std::endl;
        return 1;
    }

    omp_set_num_threads(4); // Set OpenMP thread count

    std::cout << "Number of OpenMP threads set to: " << omp_get_max_threads() << std::endl;
    
    int listen_port = std::atoi(argv[1]);

    std::cout << "Parsed listen port: " << listen_port << std::endl;

    if (listen_port <= 0 || listen_port > 65535) {
        std::cerr << "Invalid port number: " << listen_port << std::endl;
        return 1;
    }

    std::cout << "Using listen port: " << listen_port << std::endl;

    std::cout << "Starting network pipeline stage worker..." << std::endl;

    std::cout << "Running worker on port " << listen_port << std::endl;

    tpipeline::StandaloneNetworkWorker<float>::run_worker(listen_port);
    return 0;
}
