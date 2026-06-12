#include <getopt.h>

#include <memory>

#include "data_loading/data_loader_factory.hpp"
#include "device/device_manager.hpp"
#include "nn/example_graphs.hpp"
#include "nn/example_models.hpp"
#include "nn/graph.hpp"
#include "nn/schedulers.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

using namespace std;
using namespace synet;

signed main(int argc, char *argv[]) {
  ExampleGraphs::register_defaults();
  ExampleModels::register_defaults();

  std::string config_path;
  static struct option long_options[] = {
      {"config", required_argument, 0, 'c'}, {"help", no_argument, 0, 'h'}, {0, 0, 0, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "c:h", long_options, nullptr)) != -1) {
    switch (opt) {
      case 'c':
        config_path = optarg;
        break;
      case 'h':
        cout << "Usage: " << argv[0] << " [options]" << endl;
        cout << "Options:" << endl;
        cout << "  --config <path>    Path to the JSON configuration file" << endl;
        cout << "  -h, --help         Show this help message" << endl;
        return 0;
      default:
        return 1;
    }
  }

  TrainingConfig train_config;

  if (!config_path.empty()) {
    train_config.load_from_json(config_path);
  } else {
    train_config.load_from_env();
  }
  train_config.print_config();

  // Prioritize loading existing model, else create from available ones
  const auto &device = DeviceManager::getInstance().getDevice(train_config.device_type);
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);

  if (train_config.dataset_name.empty()) {
    throw std::runtime_error("DATASET_NAME environment variable is not set!");
  }
  auto [train_loader, val_loader] = DataLoaderFactory::create(
      train_config.dataset_name, train_config.dataset_path, train_config.io_dtype);
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for dataset: " << train_config.dataset_name << endl;
    return 1;
  }
  train_loader->set_seed(123456);

  Graph graph = load_or_create_graph(train_config.model_name, train_config.model_path, allocator);

  auto criterion = LossFactory::create_crossentropy();
  int adamw = 1;
  float adam_beta1 = 0.9f;
  float adam_beta2 = 0.95f;
  float adam_eps = 1e-8f;
  float weight_decay = 0.1f;
  Env::get("ADAMW", adamw);
  Env::get("ADAM_BETA1", adam_beta1);
  Env::get("ADAM_BETA2", adam_beta2);
  Env::get("ADAM_EPS", adam_eps);
  Env::get("WEIGHT_DECAY", weight_decay);

  auto optimizer = OptimizerFactory::create_adam(train_config.lr_initial, adam_beta1, adam_beta2,
                                                 adam_eps, weight_decay, adamw != 0);

  std::string scheduler_type = "warmup_cosine";
  Env::get("SCHEDULER_TYPE", scheduler_type);

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

  size_t step_size = step_lr_steps > 0 ? static_cast<size_t>(step_lr_steps)
                                       : static_cast<size_t>(step_lr_epochs) * steps_per_epoch;
  if (step_size == 0) step_size = 1;

  auto scheduler =
      (scheduler_type == "warmup_cosine" || scheduler_type == "cosine")
          ? SchedulerFactory::create_warmup_cosine(optimizer.get(),
                                                   static_cast<size_t>(warmup_steps), total_steps,
                                                   cosine_start_lr, cosine_eta_min)
          : SchedulerFactory::create_step_lr(optimizer.get(), step_size, step_lr_gamma);

  std::cout << "Optimizer: " << optimizer->name() << ", lr:" << train_config.lr_initial
            << ", beta1:" << adam_beta1 << ", beta2:" << adam_beta2 << ", eps:" << adam_eps
            << ", weight_decay:" << weight_decay << ", scheduler:" << scheduler->name()
            << ", warmup_steps:" << warmup_steps << ", total_steps:" << total_steps << std::endl;

  try {
    train_model(graph, train_loader, val_loader, optimizer, criterion, scheduler, train_config);
  } catch (const std::exception &e) {
    cerr << "Training failed: " << e.what() << endl;
    return 1;
  }

  return 0;
}