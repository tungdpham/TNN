/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>

#include "data_loading/data_loader_factory.hpp"
#include "distributed/endpoint.hpp"
#include "distributed/stage_config.hpp"
#include "distributed/tcp_communicator.hpp"
#include "distributed/tcp_worker.hpp"
#include "nn/example_models.hpp"
#include "nn/layers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

using namespace tnn;
using namespace std;

int main() {
  ExampleModels::register_defaults();

  TrainingConfig train_config;
  train_config.load_from_env();
  train_config.print_config();

  // Prioritize loading existing model, else create from available ones
  std::string model_name = Env::get<std::string>("MODEL_NAME", "cifar10_resnet9");
  std::string model_path = Env::get<std::string>("MODEL_PATH", "");

  std::string device_str = Env::get<std::string>("DEVICE_TYPE", "CPU");
  DeviceType device_type = (device_str == "GPU") ? DeviceType::GPU : DeviceType::CPU;
  const auto &device = DeviceManager::getInstance().getDevice(device_type);

  std::unique_ptr<Sequential> model;
  if (!model_path.empty()) {
    cout << "Loading model from: " << model_path << endl;
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open model file");
    }
    model = load_state<Sequential>(file, device);
    file.close();
  } else {
    cout << "Creating model: " << model_name << endl;
    try {
      Sequential temp_model = ExampleModels::create(model_name);
      model = std::make_unique<Sequential>(std::move(temp_model));
    } catch (const std::exception &e) {
      cerr << "Error creating model: " << e.what() << endl;
      cout << "Available models are: ";
      for (const auto &name : ExampleModels::available_models()) {
        cout << name << "\n";
      }
      cout << endl;
      return 1;
    }
    model->set_device(device);
    model->init();
  }

  string dataset_name = Env::get<std::string>("DATASET_NAME", "");
  if (dataset_name.empty()) {
    throw std::runtime_error("DATASET_NAME environment variable is not set!");
  }
  string dataset_path = Env::get<std::string>("DATASET_PATH", "data");
  auto [train_loader, val_loader] = DataLoaderFactory::create(dataset_name, dataset_path);
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for model: " << model_name << endl;
    return 1;
  }

  cout << "Training model on device: " << (device_type == DeviceType::CPU ? "CPU" : "GPU") << endl;

  auto criterion = LossFactory::create_logsoftmax_crossentropy();
  auto optimizer =
      OptimizerFactory::create_adam(train_config.lr_initial, 0.9f, 0.999f, 1e-5f, 1e-4f, false);
  auto scheduler = SchedulerFactory::create_step_lr(
      optimizer.get(), 5 * train_loader->size() / train_config.batch_size, 0.1f);

  auto comm_endpoint = Endpoint::tcp(Env::get<std::string>("COORDINATOR_HOST", "localhost"),
                                     Env::get<int>("COORDINATOR_PORT", 9000));

  auto communicator = std::make_unique<TcpCommunicator>(comm_endpoint);

  auto in_process_endpoint = Endpoint::in_process(communicator.get());

  StageConfig config{
      model->get_config(), optimizer->get_config(), comm_endpoint, comm_endpoint, comm_endpoint,
  };

  auto endpoint = Endpoint::tcp("localhost", 8000);
  auto worker = std::make_unique<TCPWorker>(endpoint, device_type == DeviceType::GPU, 4);

  std::thread worker_thread([&worker]() { worker->start(); });
  worker_thread.detach();

  return 0;
}