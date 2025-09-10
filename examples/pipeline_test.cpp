#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "pipeline/in_process_coordinator.hpp"
#include "pipeline/message.hpp"
#include "tensor/tensor.hpp"
#include "utils/misc.hpp"
#include "utils/mnist_data_loader.hpp"
#include "utils/ops.hpp"

namespace mnist_constants {
constexpr float EPSILON = 1e-15f;
constexpr int PROGRESS_PRINT_INTERVAL = 100;
constexpr int EPOCHS = 1;
constexpr size_t BATCH_SIZE =
    128; // Good balance between memory and convergence
constexpr int LR_DECAY_INTERVAL = 2;
constexpr float LR_DECAY_FACTOR = 0.8f;
constexpr float LR_INITIAL = 0.01f; // Initial learning rate for training
constexpr int NUM_MICROBATCHES =
    2; // Number of microbatches for pipeline processing
} // namespace mnist_constants

signed main() {
  // Load MNIST dataset
  data_loading::MNISTDataLoader<float> train_loader, test_loader;
  if (!train_loader.load_data("./data/mnist/train.csv")) {
    std::cerr << "Failed to load training data!" << std::endl;
    return -1;
  }

  if (!test_loader.load_data("./data/mnist/test.csv")) {
    std::cerr << "Failed to load test data!" << std::endl;
    return -1;
  }

  utils::set_num_threads(4);

  train_loader.prepare_batches(mnist_constants::BATCH_SIZE);
  test_loader.prepare_batches(mnist_constants::BATCH_SIZE);

  // Create a sequential model using the builder pattern
  auto model =
      tnn::SequentialBuilder<float>("mnist_cnn_classifier")
          .input(
              {1, mnist_constants::IMAGE_HEIGHT, mnist_constants::IMAGE_WIDTH})
          .conv2d(8, 5, 5, 1, 1, 0, 0, "relu", true, "conv1")
          .maxpool2d(3, 3, 3, 3, 0, 0, "pool1")
          .conv2d(16, 1, 1, 1, 1, 0, 0, "relu", true, "conv2_1x1")
          .conv2d(48, 5, 5, 1, 1, 0, 0, "relu", true, "conv3")
          .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
          .flatten("flatten")
          .dense(::mnist_constants::NUM_CLASSES, "linear", true, "output")
          .build();

  auto optimizer = std::make_unique<tnn::Adam<float>>(
      mnist_constants::LR_INITIAL, 0.9f, 0.999f, 1e-8f);
  model.set_optimizer(std::move(optimizer));

  auto loss_functions =
      tnn::LossFactory<float>::create_crossentropy(mnist_constants::EPSILON);

  auto coordinator = tpipeline::InProcessPipelineCoordinator<float>(
      model, 1, mnist_constants::NUM_MICROBATCHES);

  coordinator.set_loss_function(std::move(loss_functions));

  // Get the stages from the coordinator
  auto stages = coordinator.get_stages();

  for (auto &stage : stages) {
    stage->start();
  }

  // Prepare the training data loader
  train_loader.prepare_batches(mnist_constants::BATCH_SIZE);
  test_loader.prepare_batches(mnist_constants::BATCH_SIZE);

  Tensor<float> batch_data;
  Tensor<float> batch_labels;
  int batch_index = 0;
  std::cout << "Starting training..." << std::endl;

  auto loss_function = tnn::LossFactory<float>::create("crossentropy");

#pragma omp parallel sections
  {
#pragma omp section
    {
      // Start the message loop for each stage
      for (auto &stage : stages) {
        stage->message_loop();
      }
    }
#pragma omp section
    {
      // Main training loop
      for (int epoch = 0; epoch < mnist_constants::EPOCHS; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << mnist_constants::EPOCHS
                  << std::endl;
        train_loader.reset();
        batch_index = 0;

        train_loader.shuffle();
        test_loader.shuffle();

        train_loader.reset();
        test_loader.reset();

        auto epoch_start = std::chrono::high_resolution_clock::now();

        while (true) {
          auto get_next_batch_start = std::chrono::high_resolution_clock::now();
          bool is_valid_batch =
              train_loader.get_next_batch(batch_data, batch_labels);
          if (!is_valid_batch) {
            break;
          }
          auto get_next_batch_end = std::chrono::high_resolution_clock::now();
          auto get_next_batch_duration =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  get_next_batch_end - get_next_batch_start);

          auto split_start = std::chrono::high_resolution_clock::now();

          std::vector<Tensor<float>> micro_batches =
              batch_data.split(mnist_constants::NUM_MICROBATCHES);

          std::vector<Tensor<float>> micro_batch_labels =
              batch_labels.split(mnist_constants::NUM_MICROBATCHES);

          auto split_end = std::chrono::high_resolution_clock::now();
          auto split_duration =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  split_end - split_start);

          auto process_start = std::chrono::high_resolution_clock::now();
          coordinator.async_process_batch(micro_batches, micro_batch_labels);
          auto process_end = std::chrono::high_resolution_clock::now();
          auto process_duration =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  process_end - process_start);

          auto update_start = std::chrono::high_resolution_clock::now();
          coordinator.update_parameters();

          auto update_end = std::chrono::high_resolution_clock::now();
          auto update_duration =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  update_end - update_start);

          if (batch_index % mnist_constants::PROGRESS_PRINT_INTERVAL == 0) {
            std::cout << "Get batch completed in "
                      << get_next_batch_duration.count() << " microseconds"
                      << std::endl;
            std::cout << "Split completed in " << split_duration.count()
                      << " microseconds" << std::endl;
            std::cout << "Async process completed in "
                      << process_duration.count() << " microseconds"
                      << std::endl;
            std::cout << "Parameter update completed in "
                      << update_duration.count() << " microseconds"
                      << std::endl;
            std::cout << "Batch " << batch_index << "/"
                      << train_loader.size() / train_loader.get_batch_size()
                      << std::endl;
            coordinator.print_profiling_on_all_stages();
          }
          coordinator.clear_profiling_data();
          ++batch_index;
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end -
                                                                  epoch_start);
        std::cout << "\nEpoch " << (batch_index / train_loader.size()) + 1
                  << " completed in " << epoch_duration.count()
                  << " milliseconds" << std::endl;

        loss_function = coordinator.get_loss_function()->clone();

        double val_loss = 0.0;
        double val_accuracy = 0.0;
        int val_batches = 0;
        while (test_loader.get_batch(mnist_constants::BATCH_SIZE, batch_data,
                                     batch_labels)) {

          std::vector<Tensor<float>> micro_batches =
              batch_data.split(mnist_constants::NUM_MICROBATCHES);

          std::vector<Tensor<float>> micro_batch_labels =
              batch_labels.split(mnist_constants::NUM_MICROBATCHES);

          for (size_t i = 0; i < micro_batches.size(); ++i) {
            coordinator.forward(micro_batches[i], i);
          }

          coordinator.join(1);

          std::vector<tpipeline::Message<float>> all_messages =
              coordinator.dequeue_all_messages(
                  tpipeline::CommandType::FORWARD_TASK);

          if (all_messages.size() != mnist_constants::NUM_MICROBATCHES) {
            throw std::runtime_error(
                "Unexpected number of messages: " +
                std::to_string(all_messages.size()) + ", expected: " +
                std::to_string(mnist_constants::NUM_MICROBATCHES));
          }

          std::vector<tpipeline::Task<float>> forward_tasks;
          for (const auto &message : all_messages) {
            if (message.command_type == tpipeline::CommandType::FORWARD_TASK) {
              forward_tasks.push_back(message.get_task());
            }
          }

          for (auto &task : forward_tasks) {
            utils::apply_softmax<float>(task.data);
            val_loss += loss_function->compute_loss(
                task.data, micro_batch_labels[task.micro_batch_id]);
            val_accuracy += utils::compute_class_accuracy<float>(
                task.data, micro_batch_labels[task.micro_batch_id]);
          }
          ++val_batches;
        }

        std::cout << "\nValidation completed!" << std::endl;
        std::cout << "Average Validation Loss: " << (val_loss / val_batches)
                  << ", Average Validation Accuracy: "
                  << (val_accuracy / val_batches /
                      mnist_constants::NUM_MICROBATCHES) *
                         100.0f
                  << "%" << std::endl;
      }
    }
  }
  std::cout << "Program stopped successfully." << std::endl;
  return 0;
}