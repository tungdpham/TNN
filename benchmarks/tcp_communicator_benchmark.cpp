#include <getopt.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>

#include "device/device_manager.hpp"
#include "distributed/message.hpp"
#include "distributed/tcp_communicator.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_wrapper.hpp"

using namespace tnn;
using namespace std;

struct Config {
  std::string host = "localhost";
  int port = 0;
  std::string peer_host = "localhost";
  int peer_port = 0;
  size_t num_threads = 8;
};

void print_usage(const char *program_name) {
  cout << "Usage: " << program_name << " [options]" << endl;
  cout << endl;
  cout << "Options:" << endl;
  cout << "  --host <hostname>       Host to listen on (default: localhost)" << endl;
  cout << "  --port <port>           Port to listen on (required)" << endl;
  cout << "  --peer-host <hostname>  Peer host to connect to (default: localhost)" << endl;
  cout << "  --peer-port <port>      Peer port to connect to (required)" << endl;
  cout << "  --num-threads <N>       Number of worker threads (default: 8)" << endl;
  cout << "  -h, --help              Show this help message" << endl;
  cout << endl;
  cout << "Examples:" << endl;
  cout << "  " << program_name << " --port 8001 --peer-port 8002" << endl;
  cout << "  " << program_name
       << " --host 0.0.0.0 --port 8001 --peer-host 192.168.1.10 --peer-port 8002" << endl;
  cout << "  " << program_name << " --port 8001 --peer-port 8002 --num-threads 16" << endl;
}

bool parse_arguments(int argc, char *argv[], Config &cfg) {
  int c;

  static struct option long_options[] = {{"host", required_argument, 0, 'H'},
                                         {"port", required_argument, 0, 'p'},
                                         {"peer-host", required_argument, 0, 'P'},
                                         {"peer-port", required_argument, 0, 'r'},
                                         {"num-threads", required_argument, 0, 'n'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  optind = 1;

  while ((c = getopt_long(argc, argv, "h", long_options, nullptr)) != -1) {
    switch (c) {
      case 'H':
        cfg.host = optarg;
        break;
      case 'p':
        try {
          cfg.port = stoi(optarg);
          if (cfg.port <= 0 || cfg.port > 65535) {
            cerr << "Invalid port value: " << optarg << endl;
            return false;
          }
        } catch (...) {
          cerr << "--port requires a valid number argument" << endl;
          return false;
        }
        break;
      case 'P':
        cfg.peer_host = optarg;
        break;
      case 'r':
        try {
          cfg.peer_port = stoi(optarg);
          if (cfg.peer_port <= 0 || cfg.peer_port > 65535) {
            cerr << "Invalid peer-port value: " << optarg << endl;
            return false;
          }
        } catch (...) {
          cerr << "--peer-port requires a valid number argument" << endl;
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

  if (cfg.port == 0) {
    cerr << "Missing required option: --port" << endl;
    print_usage(argv[0]);
    return false;
  }

  if (cfg.peer_port == 0) {
    cerr << "Missing required option: --peer-port" << endl;
    print_usage(argv[0]);
    return false;
  }

  return true;
}

int main(int argc, char *argv[]) {
  Config cfg;

  if (!parse_arguments(argc, argv, cfg)) {
    return 1;
  }

  cout << "Communicator Benchmark Configuration" << endl;
  cout << "Listen host: " << cfg.host << endl;
  cout << "Listen port: " << cfg.port << endl;
  cout << "Peer host: " << cfg.peer_host << endl;
  cout << "Peer port: " << cfg.peer_port << endl;
  cout << "Worker threads: " << cfg.num_threads << endl;

  TcpCommunicator communicator(Endpoint::tcp(cfg.host, cfg.port), cfg.num_threads);

  communicator.start_server();

  Endpoint local_endpoint = Endpoint::tcp(cfg.host, cfg.port);
  Endpoint peer_endpoint = Endpoint::tcp(cfg.peer_host, cfg.peer_port);
  while (!communicator.connect(peer_endpoint)) {
    cerr << "Retrying connection to peer..." << endl;
    sleep(1);
  }

  ThreadWrapper thread_wrapper({static_cast<unsigned int>(cfg.num_threads)});
  Tensor master_tensor = make_tensor<float>({128, 512, 16, 16}, getCPU());
  float *master_data = master_tensor->data_as<float>();
  for (size_t i = 0; i < master_tensor->size(); ++i) {
    master_data[i] = static_cast<float>(i);
  }

  for (int i = 0; i < 4; i++) {
    Tensor tensor = make_tensor<float>(master_tensor->shape(), getGPU());
    master_tensor->copy_to(tensor);
    Job job;
    job.mb_id = 10;
    job.data = std::move(tensor);
    Message message(CommandType::FORWARD_JOB, std::move(job));
    communicator.send_message(std::move(message), peer_endpoint);
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  auto current_time = start_time;
  std::atomic<int> num_messages_received(0);
  condition_variable message_available_cv_;
  mutex message_available_mutex_;
  communicator.set_callback([&]() {
    std::unique_lock<std::mutex> lock(message_available_mutex_);
    message_available_cv_.notify_one();
  });
  thread_wrapper.execute([&]() {
    while (current_time - start_time < std::chrono::seconds(10)) {
      std::unique_lock<std::mutex> lock(message_available_mutex_);
      message_available_cv_.wait_until(lock, start_time + std::chrono::seconds(10),
                                       [&]() { return communicator.has_input_message(); });

      if (std::chrono::high_resolution_clock::now() - start_time >= std::chrono::seconds(10)) {
        break;
      }

      while (communicator.has_input_message()) {
        auto message = communicator.dequeue_input_message();
        if (message.header().command_type != CommandType::FORWARD_JOB) {
          continue;
        }
        // verify integrity of received tensor
        Job &job = message.get<Job>();
        Tensor &tensor = job.data;
        Tensor cpu_tensor = tensor->to_device(getCPU());
        assert(cpu_tensor->shape() == master_tensor->shape());
        assert(cpu_tensor->data_type() == master_tensor->data_type());
        assert(cpu_tensor->size() == master_tensor->size());
        assert(job.mb_id == 10 && "Unexpected mb_id in received job");
        // float *recv_data = cpu_tensor->data_as<float>();
        // bool valid = true;
        // for (size_t i = 0; i < tensor->size(); ++i) {
        //   if (recv_data[i] != master_data[i]) {
        //     valid = false;
        //     break;
        //   }
        // }
        // if (!valid) {
        //   throw std::runtime_error("Data integrity check failed for received tensor");
        // }
        num_messages_received++;
        communicator.send_message(std::move(message), peer_endpoint);
      }

      current_time = std::chrono::high_resolution_clock::now();
    }
  });
  int kb_per_message = 128 * 512 * 16 * 16 * sizeof(float) / 1024;
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_duration = end_time - start_time;
  double total_kb = static_cast<double>(num_messages_received) * kb_per_message;
  double bandwidth_mbps = (total_kb / 1024.0) / total_duration.count();
  std::cout << "Total messages received: " << num_messages_received << std::endl;
  std::cout << "Total bytes received: " << total_kb * 1024 << " bytes" << std::endl;
  std::cout << "Total time taken: " << total_duration.count() << " seconds" << std::endl;
  std::cout << "Bandwidth: " << bandwidth_mbps << " MB/s" << std::endl;
  communicator.stop();
  return 0;
}