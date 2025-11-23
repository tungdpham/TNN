#include "data_loading/tiny_imagenet_data_loader.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace tnn;

// Shared test fixture that loads data once for all tests
class TinyImageNetLoaderTest : public ::testing::Test {
protected:
  static std::string dataset_path;
  static TinyImageNetDataLoader<float> train_loader;
  static TinyImageNetDataLoader<float> val_loader;
  static bool data_loaded;

  static void SetUpTestSuite() {
    if (!data_loaded) {
      std::cout << "\n=== Loading datasets once for all tests ===\n" << std::endl;

      std::cout << "Loading training data..." << std::endl;
      ASSERT_TRUE(train_loader.load_data(dataset_path, true)) << "Failed to load training data";
      std::cout << "Loaded " << train_loader.size() << " training samples\n" << std::endl;

      std::cout << "Loading validation data..." << std::endl;
      ASSERT_TRUE(val_loader.load_data(dataset_path, false)) << "Failed to load validation data";
      std::cout << "Loaded " << val_loader.size() << " validation samples\n" << std::endl;

      data_loaded = true;
    }
  }

  void SetUp() override {
    // Reset loaders before each test
    train_loader.reset();
    val_loader.reset();
  }
};

// Initialize static members
std::string TinyImageNetLoaderTest::dataset_path = "data/tiny-imagenet-200";
TinyImageNetDataLoader<float> TinyImageNetLoaderTest::train_loader;
TinyImageNetDataLoader<float> TinyImageNetLoaderTest::val_loader;
bool TinyImageNetLoaderTest::data_loaded = false;

TEST_F(TinyImageNetLoaderTest, TrainingDataSize) {
  // Check expected number of training samples (200 classes * 500 images)
  EXPECT_EQ(train_loader.size(), 100000) << "Expected 100,000 training samples";
}

TEST_F(TinyImageNetLoaderTest, ValidationDataSize) {
  // Check expected number of validation samples
  EXPECT_EQ(val_loader.size(), 10000) << "Expected 10,000 validation samples";
}

TEST_F(TinyImageNetLoaderTest, ImageShape) {
  auto shape = train_loader.get_image_shape();
  ASSERT_EQ(shape.size(), 3);
  EXPECT_EQ(shape[0], 3);  // Channels
  EXPECT_EQ(shape[1], 64); // Height
  EXPECT_EQ(shape[2], 64); // Width
}

TEST_F(TinyImageNetLoaderTest, NumClasses) { EXPECT_EQ(train_loader.get_num_classes(), 200); }

TEST_F(TinyImageNetLoaderTest, BatchRetrieval) {
  Tensor<float> batch_data, batch_labels;
  size_t batch_size = 32;

  ASSERT_TRUE(train_loader.get_batch(batch_size, batch_data, batch_labels));

  // Check batch shapes
  EXPECT_EQ(batch_data.shape()[0], batch_size);
  EXPECT_EQ(batch_data.shape()[1], 3);
  EXPECT_EQ(batch_data.shape()[2], 64);
  EXPECT_EQ(batch_data.shape()[3], 64);

  EXPECT_EQ(batch_labels.shape()[0], batch_size);
  EXPECT_EQ(batch_labels.shape()[1], 200);
}

TEST_F(TinyImageNetLoaderTest, PixelNormalization) {
  Tensor<float> batch_data, batch_labels;
  ASSERT_TRUE(train_loader.get_batch(8, batch_data, batch_labels));

  // Check pixel values are normalized to [0, 1]
  float min_val = batch_data.min();
  float max_val = batch_data.max();

  EXPECT_GE(min_val, 0.0f) << "Pixel values should be >= 0";
  EXPECT_LE(max_val, 1.0f) << "Pixel values should be <= 1";
}

TEST_F(TinyImageNetLoaderTest, OneHotLabels) {
  Tensor<float> batch_data, batch_labels;
  ASSERT_TRUE(train_loader.get_batch(16, batch_data, batch_labels));

  // Verify one-hot encoding
  for (size_t i = 0; i < 16; ++i) {
    float sum = 0.0f;
    int num_ones = 0;

    for (size_t j = 0; j < 200; ++j) {
      float val = batch_labels(i, j, 0, 0);
      sum += val;
      if (val > 0.5f)
        num_ones++;
    }

    EXPECT_NEAR(sum, 1.0f, 1e-5) << "Label sum should be 1.0 for sample " << i;
    EXPECT_EQ(num_ones, 1) << "Should have exactly one hot label for sample " << i;
  }
}

TEST_F(TinyImageNetLoaderTest, ShuffleAndReset) {
  // Get first batch
  Tensor<float> batch1_data, batch1_labels;
  train_loader.get_batch(4, batch1_data, batch1_labels);

  // Reset and get first batch again - should be identical
  train_loader.reset();
  Tensor<float> batch2_data, batch2_labels;
  train_loader.get_batch(4, batch2_data, batch2_labels);

  // Compare first samples
  bool identical = true;
  for (size_t i = 0; i < batch1_data.size(); ++i) {
    if (std::abs(batch1_data.data()[i] - batch2_data.data()[i]) > 1e-6) {
      identical = false;
      break;
    }
  }
  EXPECT_TRUE(identical) << "Reset should return to beginning";

  // Shuffle and verify it changes
  train_loader.shuffle();
  train_loader.reset();
  Tensor<float> batch3_data, batch3_labels;
  train_loader.get_batch(4, batch3_data, batch3_labels);

  bool changed = false;
  for (size_t i = 0; i < batch1_data.size(); ++i) {
    if (std::abs(batch1_data.data()[i] - batch3_data.data()[i]) > 1e-6) {
      changed = true;
      break;
    }
  }
  EXPECT_TRUE(changed) << "Shuffle should change data order";
}

TEST_F(TinyImageNetLoaderTest, ClassIdsAndNames) {
  auto class_ids = train_loader.get_class_ids();
  auto class_names = train_loader.get_class_names();

  EXPECT_EQ(class_ids.size(), 200);
  EXPECT_EQ(class_names.size(), 200);

  // Print first few classes for manual verification
  std::cout << "\nFirst 5 classes:" << std::endl;
  for (int i = 0; i < 5; ++i) {
    std::cout << "  " << i << ": " << class_ids[i] << " - " << class_names[i] << std::endl;
  }
}

TEST_F(TinyImageNetLoaderTest, PrepareBatches) {
  // Create a separate loader for this test since it modifies loader state
  TinyImageNetDataLoader<float> loader;
  ASSERT_TRUE(loader.load_data(dataset_path, true));

  size_t batch_size = 64;
  loader.prepare_batches(batch_size);

  EXPECT_TRUE(loader.are_batches_prepared());

  size_t expected_batches = (loader.size() + batch_size - 1) / batch_size;
  EXPECT_EQ(loader.num_batches(), expected_batches);

  // Iterate through all batches
  size_t count = 0;
  Tensor<float> batch_data, batch_labels;
  while (loader.get_next_batch(batch_data, batch_labels)) {
    count++;
  }

  EXPECT_EQ(count, expected_batches) << "Should iterate through all batches";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
