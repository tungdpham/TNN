

#include "data_augmentation/augmentation.hpp"
#include "data_loading/cifar10_data_loader.hpp"
#include "data_loading/mnist_data_loader.hpp"
#include "data_loading/tiny_imagenet_data_loader.hpp"
#include "nn/example_models.hpp"
#include "nn/sequential.hpp"
#include "partitioner/naive_partitioner.hpp"
#include "pipeline/distributed_coordinator.hpp"
#include "pipeline/train.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_wrapper.hpp"
#include "utils/env.hpp"

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <vector>

using namespace tnn;
using namespace std;

constexpr float LR_INITIAL = 0.001f; // Careful, too big can cause exploding gradients
constexpr float EPSILON = 1e-7f;

struct Config {
  size_t io_threads = 4;
};

void print_usage(const char *program_name) {
  cout << "Usage: " << program_name << " [options]" << endl;
  cout << endl;
  cout << "Options:" << endl;
  cout << "  --io-threads <N>   Number of IO threads for networking (default: 1)" << endl;
  cout << "  -h, --help         Show this help message" << endl;
  cout << endl;
  cout << "Examples:" << endl;
  cout << "  " << program_name << "                    # Default mode" << endl;
  cout << "  " << program_name << " --io-threads 4    # Use 4 IO threads" << endl;
}

bool parse_arguments(int argc, char *argv[], Config &cfg) {
  int c;

  static struct option long_options[] = {
      {"io-threads", required_argument, 0, 'i'}, {"help", no_argument, 0, 'h'}, {0, 0, 0, 0}};

  optind = 1;

  while ((c = getopt_long(argc, argv, "h", long_options, nullptr)) != -1) {
    switch (c) {
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
    case 'h':
      print_usage(argv[0]);
      return false;
    case '?':
      return false;
    default:
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
  if (!load_env_file("./.env")) {
    std::cout << "No .env file found, using system environment variables only." << std::endl;
  }

  auto model = create_resnet18_tiny_imagenet();

  string device_type_str = get_env<string>("DEVICE_TYPE", "CPU");

  float lr_initial = get_env<float>("LR_INITIAL", LR_INITIAL);

  TrainingConfig train_config;
  train_config.load_from_env();
  train_config.print_config();

  auto optimizer = std::make_unique<Adam<float>>(lr_initial, 0.9f, 0.999f, EPSILON);
  model.set_optimizer(std::move(optimizer));

  Endpoint coordinator_endpoint =
      Endpoint::network(get_env<std::string>("COORDINATOR_HOST", "localhost"),
                        get_env<int>("COORDINATOR_PORT", 8000));

  std::vector<Endpoint> endpoints = {
      Endpoint::network(get_env<std::string>("WORKER1_HOST", "localhost"),
                        get_env<int>("WORKER1_PORT", 8001)),
      Endpoint::network(get_env<std::string>("WORKER2_HOST", "localhost"),
                        get_env<int>("WORKER2_PORT", 8002)),

  };

  std::cout << "Configured " << endpoints.size() << " remote endpoints:" << std::endl;
  for (const auto &ep : endpoints) {
    std::cout << ep.to_json().dump(4) << std::endl;
  }
  std::cout << "IO threads: " << cfg.io_threads << std::endl;

  std::cout << "Creating distributed coordinator." << std::endl;
  DistributedCoordinator coordinator(std::move(model), coordinator_endpoint, endpoints,
                                     cfg.io_threads);

  coordinator.set_partitioner(std::make_unique<NaivePartitioner<float>>());
  coordinator.initialize();

  auto loss_function = LossFactory<float>::create_logsoftmax_crossentropy();

  coordinator.set_loss_function(std::move(loss_function));
  std::cout << "Deploying stages to remote endpoints." << std::endl;
  for (const auto &ep : endpoints) {
    std::cout << "  Worker expected at " << ep.to_json().dump(4) << std::endl;
  }

  if (!coordinator.deploy_stages()) {
    std::cerr << "Failed to deploy stages. Make sure workers are running." << std::endl;
    return 1;
  }

  coordinator.start();

  TinyImageNetDataLoader<float> train_loader, test_loader;
  std::string dataset_path = "data/tiny-imagenet-200";

  create_tiny_image_loader(dataset_path, train_loader, test_loader);

  auto train_aug = AugmentationBuilder<float>()
                       //  .horizontal_flip(0.25f)
                       //  .rotation(0.3f, 10.0f)
                       //  .brightness(0.3f, 0.15f)
                       //  .contrast(0.3f, 0.15f)
                       //  .gaussian_noise(0.3f, 0.05f)
                       //  .random_crop(0.25, 4)
                       //  .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
                       .build();
  std::cout << "Configuring data augmentation and normalization for training." << std::endl;
  train_loader.set_augmentation(std::move(train_aug));

  auto test_aug = AugmentationBuilder<float>()
                      // .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
                      .build();
  std::cout << "Configuring data normalization for testing." << std::endl;
  test_loader.set_augmentation(std::move(test_aug));

  ThreadWrapper thread_wrapper({get_env<unsigned int>("COORDINATOR_NUM_THREADS", 4)});

  thread_wrapper.execute([&coordinator, &train_loader, &test_loader, &train_config]() {
    train_model(coordinator, train_loader, test_loader, train_config);
  });

  coordinator.stop();

  return 0;
}