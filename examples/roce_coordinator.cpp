#include "distributed/roce_coordinator.hpp"

#include <getopt.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "data_loading/data_loader_factory.hpp"
#include "distributed/coordinator.hpp"
#include "distributed/roce_worker.hpp"
#include "distributed/train.hpp"
#include "nn/example_models.hpp"
#include "nn/optimizers.hpp"
#include "partitioner/naive_partitioner.hpp"
#include "utils/env.hpp"

using namespace tnn;
using namespace std;

struct Config {
  std::string device_name = "";
  int gid_index = -1;
};

void print_usage(const char *program_name) {
  cout << "Usage: " << program_name << " [options] <worker_host:port>..." << endl;
  cout << endl;
  cout << "Options:" << endl;
  cout << "  --host <address>       Hostname or IP to bind to (required)" << endl;
  cout << "  --port <number>        TCP port for initial connection (required)" << endl;
  cout << "  --device <name>        IB device name (e.g., mlx5_0) (required)" << endl;
  cout << "  --gid-index <index>    GID index for RoCE (required)" << endl;
  cout << "  -h, --help             Show this help message" << endl;
}

bool parse_arguments(int argc, char *argv[], Config &cfg) {
  int c;

  static struct option long_options[] = {{"device", required_argument, 0, 'd'},
                                         {"gid-index", required_argument, 0, 'g'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  while ((c = getopt_long(argc, argv, "H:p:d:g:h", long_options, nullptr)) != -1) {
    switch (c) {
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
      case 'h':
        print_usage(argv[0]);
        return false;
      case '?':
        return false;
      default:
        return false;
    }
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

  ExampleModels::register_defaults();

  TrainingConfig train_config;
  train_config.load_from_env();
  train_config.print_config();

  // Prioritize loading existing model, else create from available ones
  DeviceType device_type = train_config.device_type;
  const auto &device = DeviceManager::getInstance().getDevice(device_type);
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);

  Graph graph = load_or_create_model(train_config.model_name, train_config.model_path, allocator);

  if (train_config.dataset_name.empty()) {
    throw std::runtime_error("DATASET_NAME environment variable is not set!");
  }
  auto [train_loader, val_loader] =
      DataLoaderFactory::create(train_config.dataset_name, train_config.dataset_path);
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for model: " << train_config.model_name << endl;
    return 1;
  }

  auto criterion = LossFactory::create_logsoftmax_crossentropy();
  int adamw = 1;
  float adam_beta1 = 0.9f;
  float adam_beta2 = 0.95f;
  float adam_eps = 1e-8f;
  float weight_decay = 0.1f;
  Env::get("ADAMW", adamw);
  Env::get("ADAM_BETA1", adam_beta1);
  Env::get("ADAM_BETA2", adam_beta2);
  Env::get("ADAM_EPS", adam_eps);
  Env::get("ADAM_EPSILON", adam_eps);
  Env::get("WEIGHT_DECAY", weight_decay);

  auto optimizer = OptimizerFactory::create_adam(train_config.lr_initial, adam_beta1, adam_beta2,
                                                 adam_eps, weight_decay, adamw != 0);

  std::string lr_scheduler = "warmup_cosine";
  Env::get("LR_SCHEDULER", lr_scheduler);
  Env::get("SCHEDULER_TYPE", lr_scheduler);

  int step_lr_epochs = 5;
  float step_lr_gamma = 0.1f;
  int step_lr_steps = 0;
  Env::get("STEP_LR_EPOCHS", step_lr_epochs);
  Env::get("STEP_LR_GAMMA", step_lr_gamma);
  Env::get("STEP_LR_STEPS", step_lr_steps);

  size_t steps_per_epoch = train_loader->size() / train_config.batch_size;
  if (steps_per_epoch == 0) steps_per_epoch = 1;

  int cosine_total_steps = 0;
  Env::get("COSINE_TOTAL_STEPS", cosine_total_steps);
  size_t total_steps = 0;
  if (cosine_total_steps > 0) {
    total_steps = static_cast<size_t>(cosine_total_steps);
  } else if (train_config.max_steps > 0) {
    total_steps = static_cast<size_t>(train_config.max_steps);
  } else {
    total_steps = steps_per_epoch * static_cast<size_t>(train_config.epochs);
  }
  if (total_steps == 0) total_steps = 1;

  int warmup_steps = 2000;
  float cosine_start_lr = 0.0f;
  float cosine_eta_min = 0.0f;
  Env::get("WARMUP_STEPS", warmup_steps);
  Env::get("COSINE_START_LR", cosine_start_lr);
  Env::get("COSINE_ETA_MIN", cosine_eta_min);
  if (warmup_steps < 0) warmup_steps = 0;
  if (static_cast<size_t>(warmup_steps) >= total_steps) {
    warmup_steps = total_steps > 1 ? static_cast<int>(total_steps / 10) : 0;
  }

  size_t step_size = step_lr_steps > 0
                         ? static_cast<size_t>(step_lr_steps)
                         : static_cast<size_t>(step_lr_epochs) * steps_per_epoch;
  if (step_size == 0) step_size = 1;

  auto scheduler =
      (lr_scheduler == "warmup_cosine" || lr_scheduler == "cosine")
          ? SchedulerFactory::create_warmup_cosine(optimizer.get(), static_cast<size_t>(warmup_steps),
                                                   total_steps, cosine_start_lr, cosine_eta_min)
          : SchedulerFactory::create_step_lr(optimizer.get(), step_size, step_lr_gamma);

  // IMPORTANT distributed fix:
  // WarmupCosineAnnealing sets optimizer lr to COSINE_START_LR, usually 0.
  // The coordinator serializes optimizer_config later in initialize_topology().
  // Restore base LR before serialization so workers build their scheduler with
  // base_lr=LR_INITIAL instead of 0.
  optimizer->set_learning_rate(train_config.lr_initial);

  std::cout << "[Optim] optimizer=" << optimizer->name() << " lr=" << train_config.lr_initial
            << " beta1=" << adam_beta1 << " beta2=" << adam_beta2 << " eps=" << adam_eps
            << " weight_decay=" << weight_decay << " scheduler=" << scheduler->name()
            << " warmup_steps=" << warmup_steps << " total_steps=" << total_steps << std::endl;
  std::string host = "localhost";
  Env::get("COORDINATOR_HOST", host);
  int port = 9000;
  Env::get("COORDINATOR_PORT", port);

  Endpoint coordinator_endpoint = Endpoint::roce(host, port, cfg.device_name, cfg.gid_index);

  std::string local_worker_host = "localhost";
  int local_worker_port = 8000;
  Env::get("LOCAL_WORKER_HOST", local_worker_host);
  Env::get("LOCAL_WORKER_PORT", local_worker_port);
  Endpoint local_worker_endpoint =
      Endpoint::roce(local_worker_host, local_worker_port, cfg.device_name, cfg.gid_index);

  std::string position_str = "last";
  Env::get("LOCAL_WORKER_POSITION", position_str);
  int worker_position = position_str == "first" ? 0 : 1;

  std::string worker1_host = "10.10.0.2";
  int worker1_port = 8001;
  Env::get("WORKER1_HOST", worker1_host);
  Env::get("WORKER1_PORT", worker1_port);
  Vec<Endpoint> endpoints = {
      Endpoint::roce(worker1_host, worker1_port, "rocep131s0f0", -1),
  };

  if (worker_position) {
    endpoints.push_back(local_worker_endpoint);
  } else {
    endpoints.insert(endpoints.begin(), local_worker_endpoint);
  }

  auto local_worker =
      std::make_unique<RoCEWorker>(local_worker_endpoint, device_type == DeviceType::GPU);

  // Parse partition split ratio from environment variable
  std::string split_ratio_str = "2,1";
  Env::get("PARTITION_SPLIT_RATIO", split_ratio_str);
  std::vector<size_t> split_ratios;
  std::stringstream ss(split_ratio_str);
  std::string token;
  while (std::getline(ss, token, ',')) {
    split_ratios.push_back(static_cast<size_t>(std::stoi(token)));
  }

  auto partitioner =
      std::make_unique<NaivePipelinePartitioner>(NaivePartitionerConfig(split_ratios));

  CoordinatorConfig config{
      ParallelMode_t::PIPELINE, std::move(graph),        std::move(optimizer), std::move(scheduler),
      std::move(partitioner),   std::move(local_worker), coordinator_endpoint, endpoints};

  RoCECoordinator coordinator(std::move(config));

  coordinator.initialize();

  std::cout << "Deploying stages to remote endpoints." << std::endl;
  for (const auto &ep : endpoints) {
    std::cout << "  Worker expected at " << ep.to_json().dump(4) << std::endl;
  }

  if (!coordinator.deploy_stages()) {
    std::cerr << "Failed to deploy stages. Make sure workers are running." << std::endl;
    return 1;
  }

  try {
    train_model(coordinator, train_loader, val_loader, criterion, train_config);
    std::cout << "Coordinator initialized successfully." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
