#include "nn/sequential.hpp"
#include "utils/mnist_data_loader.hpp"
#include "pipeline_experimental/pipeline_coordinator.hpp"
using namespace tnn;
using namespace data_loading;


signed main() {
  // Load MNIST dataset
  MNISTDataLoader<float> train_loader, test_loader;
  if (!train_loader.load_data("./data/mnist/train.csv")) {
    std::cerr << "Failed to load training data!" << std::endl;
    return -1;
  }

  if (!test_loader.load_data("./data/mnist/test.csv")) {
    std::cerr << "Failed to load test data!" << std::endl;
    return -1;
  }
  // Create a sequential model using the builder pattern
  auto model = SequentialBuilder<float>("mnist_cnn_classifier")
                   // C1: First convolution layer - 5x5 kernel, stride 1, ReLU
                   // activation Input: 1x28x28 → Output: 8x24x24 (28-5+1=24)
                   .conv2d(1, 8, 5, 5, 1, 1, 0, 0, "relu", true, "conv1")
                   // .batchnorm(8, 1e-5, 0.1, true, "batchnorm1")
                   // .activation("relu", "relu1")
                   // P1: Max pooling layer - 3x3 blocks, stride 3
                   // Input: 8x24x24 → Output: 8x8x8 (24/3=8)
                   .maxpool2d(3, 3, 3, 3, 0, 0, "pool1")

                   // C2: Inception-style 1x1 convolution for dimensionality
                   // reduction Input: 8x8x8 → Output: 16x8x8
                   .conv2d(8, 16, 1, 1, 1, 1, 0, 0, "relu", true, "conv2_1x1")

                   // C3: Second convolution layer - 5x5 kernel, stride 1, ReLU
                   // activation Input: 16x8x8 → Output: 48x4x4 (8-5+1=4)
                   .conv2d(16, 48, 5, 5, 1, 1, 0, 0, "relu", true, "conv3")

                   // P2: Second max pooling layer - 2x2 blocks, stride 2
                   // Input: 48x4x4 → Output: 48x2x2 (4/2=2)
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")

                   // FC1: Fully connected output layer
                   // Input: 48x2x2 = 192 features → Output: 10 classes
                   .dense(48 * 2 * 2, mnist_constants::NUM_CLASSES, "linear",
                          true, "output")
                   .build();
  
  auto pipeline_coordinator = tpipeline::InProcessPipelineCoordinator<float>(
      model, 4, 4);

  pipeline_coordinator.start();

  printf("Program stopped successfully.\n");
  return 0;
}