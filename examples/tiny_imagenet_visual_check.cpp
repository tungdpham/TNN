/**
 * Visual inspection tool for Tiny ImageNet data loader
 *
 * This program loads a few sample images from the Tiny ImageNet dataset,
 * saves them as PNG files, and prints detailed information about the data
 * to help verify the loader is working correctly.
 *
 * Requires stb_image_write.h for saving images.
 */

#include "data_loading/tiny_imagenet_data_loader.hpp"
#include "tensor/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// Forward declare stb_image_write function
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace tnn;
using namespace std;

/**
 * Save a single image from a batch to a PNG file
 */
void save_image_from_batch(const Tensor<float> &batch_data, size_t sample_idx,
                           const string &filename) {
  const size_t width = 64;
  const size_t height = 64;
  const size_t channels = 3;

  // Allocate buffer for image data in HWC format (Height, Width, Channels)
  vector<unsigned char> img_data(width * height * channels);

  // Convert from CHW format to HWC and denormalize from [0,1] to [0,255]
  for (size_t h = 0; h < height; ++h) {
    for (size_t w = 0; w < width; ++w) {
      for (size_t c = 0; c < channels; ++c) {
        float pixel = batch_data(sample_idx, c, h, w);

        // Clamp to [0, 1] and convert to [0, 255]
        pixel = max(0.0f, min(1.0f, pixel));
        unsigned char byte_val = static_cast<unsigned char>(pixel * 255.0f);

        size_t idx = (h * width + w) * channels + c;
        img_data[idx] = byte_val;
      }
    }
  }

  // Save as PNG
  if (stbi_write_png(filename.c_str(), width, height, channels, img_data.data(),
                     width * channels)) {
    cout << "  Saved: " << filename << endl;
  } else {
    cerr << "  Failed to save: " << filename << endl;
  }
}

/**
 * Print detailed statistics about a single image
 */
void print_image_stats(const Tensor<float> &batch_data, size_t sample_idx,
                       const string &label_name) {
  const size_t channels = 3;
  const size_t height = 64;
  const size_t width = 64;
  const size_t pixels_per_channel = height * width;

  cout << "\n  Image: " << label_name << endl;

  // Calculate per-channel statistics
  for (size_t c = 0; c < channels; ++c) {
    vector<float> channel_data;
    channel_data.reserve(pixels_per_channel);

    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        channel_data.push_back(batch_data(sample_idx, c, h, w));
      }
    }

    float min_val = *min_element(channel_data.begin(), channel_data.end());
    float max_val = *max_element(channel_data.begin(), channel_data.end());
    float sum = accumulate(channel_data.begin(), channel_data.end(), 0.0f);
    float mean = sum / channel_data.size();

    // Calculate standard deviation
    float sq_sum = 0.0f;
    for (float val : channel_data) {
      sq_sum += (val - mean) * (val - mean);
    }
    float stddev = sqrt(sq_sum / channel_data.size());

    const char *channel_names[] = {"Red", "Green", "Blue"};
    cout << "    " << channel_names[c] << " channel: " << "min=" << fixed << setprecision(4)
         << min_val << ", max=" << max_val << ", mean=" << mean << ", std=" << stddev << endl;
  }
}

/**
 * Print a small ASCII preview of the image (top-left corner)
 */
void print_ascii_preview(const Tensor<float> &batch_data, size_t sample_idx) {
  const size_t preview_size = 8; // Show 8x8 corner

  cout << "\n  ASCII preview (top-left 8x8, grayscale):" << endl;
  cout << "  ";
  for (size_t i = 0; i < preview_size; ++i)
    cout << "--";
  cout << endl;

  for (size_t h = 0; h < preview_size; ++h) {
    cout << "  ";
    for (size_t w = 0; w < preview_size; ++w) {
      // Average RGB to get grayscale
      float gray = 0.0f;
      for (size_t c = 0; c < 3; ++c) {
        gray += batch_data(sample_idx, c, h, w);
      }
      gray /= 3.0f;

      // Convert to ASCII character
      const char *chars = " .:;+=xX$&#";
      int idx = static_cast<int>(gray * 10);
      idx = max(0, min(10, idx));
      cout << chars[idx] << " ";
    }
    cout << endl;
  }

  cout << "  ";
  for (size_t i = 0; i < preview_size; ++i)
    cout << "--";
  cout << endl;
}

int main() {
  try {
    cout << "=== Tiny ImageNet Visual Inspection Tool ===" << endl;
    cout << "\nThis tool will:" << endl;
    cout << "1. Load sample images from the dataset" << endl;
    cout << "2. Save them as PNG files for visual inspection" << endl;
    cout << "3. Print detailed statistics about the images" << endl;
    cout << "4. Show ASCII previews" << endl;

    // Dataset path
    string dataset_path = "data/tiny-imagenet-200";

    // Create loaders
    TinyImageNetDataLoader<float> train_loader;
    TinyImageNetDataLoader<float> val_loader;

    cout << "\n--- Loading Training Data ---" << endl;
    if (!train_loader.load_data(dataset_path, true)) {
      cerr << "Failed to load training data!" << endl;
      return 1;
    }
    cout << "Loaded " << train_loader.size() << " training samples" << endl;

    cout << "\n--- Loading Validation Data ---" << endl;
    if (!val_loader.load_data(dataset_path, false)) {
      cerr << "Failed to load validation data!" << endl;
      return 1;
    }
    cout << "Loaded " << val_loader.size() << " validation samples" << endl;

    // Print overall statistics
    cout << "\n=== Dataset Information ===" << endl;
    train_loader.print_data_stats();

    // Get class names
    auto class_names = train_loader.get_class_names();
    auto class_ids = train_loader.get_class_ids();

    cout << "\n--- Sample Class Names (first 10) ---" << endl;
    for (int i = 0; i < min(10, static_cast<int>(class_names.size())); ++i) {
      cout << "  " << i << ": " << class_ids[i] << " - " << class_names[i] << endl;
    }

    auto aug_strategy = AugmentationBuilder<float>()
                            // .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
                            .build();
    train_loader.set_augmentation(std::move(aug_strategy));

    // Sample and save training images
    cout << "\n=== Sampling Training Images ===" << endl;
    Tensor<float> train_batch_data, train_batch_labels;
    const size_t num_samples = 100;

    if (!train_loader.get_batch(num_samples, train_batch_data, train_batch_labels)) {
      cerr << "Failed to get training batch!" << endl;
      return 1;
    }

    cout << "\nBatch shape: " << train_batch_data.shape()[0] << "x" << train_batch_data.shape()[1]
         << "x" << train_batch_data.shape()[2] << "x" << train_batch_data.shape()[3] << endl;

    // Create output directory if needed
    system("mkdir -p output_images");

    for (size_t i = 0; i < num_samples; ++i) {
      // Find the label
      int label_idx = -1;
      for (size_t j = 0; j < 200; ++j) {
        if (train_batch_labels(i, j, 0, 0) > 0.5f) {
          label_idx = static_cast<int>(j);
          break;
        }
      }

      string label_name = (label_idx >= 0 && label_idx < static_cast<int>(class_names.size()))
                              ? class_names[label_idx]
                              : "unknown";

      cout << "\n--- Sample " << (i + 1) << " ---" << endl;
      cout << "  Class: " << label_idx << " - " << label_name << endl;

      // Save image
      string filename = "output_images/train_sample_" + to_string(i + 1) + "_class_" +
                        to_string(label_idx) + ".png";
      save_image_from_batch(train_batch_data, i, filename);

      // Print statistics
      print_image_stats(train_batch_data, i, label_name);

      // Print ASCII preview
      print_ascii_preview(train_batch_data, i);
    }

    // Sample and save validation images
    cout << "\n=== Sampling Validation Images ===" << endl;
    Tensor<float> val_batch_data, val_batch_labels;

    if (!val_loader.get_batch(num_samples, val_batch_data, val_batch_labels)) {
      cerr << "Failed to get validation batch!" << endl;
      return 1;
    }

    for (size_t i = 0; i < num_samples; ++i) {
      // Find the label
      int label_idx = -1;
      for (size_t j = 0; j < 200; ++j) {
        if (val_batch_labels(i, j, 0, 0) > 0.5f) {
          label_idx = static_cast<int>(j);
          break;
        }
      }

      string label_name = (label_idx >= 0 && label_idx < static_cast<int>(class_names.size()))
                              ? class_names[label_idx]
                              : "unknown";

      cout << "\n--- Validation Sample " << (i + 1) << " ---" << endl;
      cout << "  Class: " << label_idx << " - " << label_name << endl;

      // Save image
      string filename = "output_images/val_sample_" + to_string(i + 1) + "_class_" +
                        to_string(label_idx) + ".png";
      save_image_from_batch(val_batch_data, i, filename);

      // Print statistics
      print_image_stats(val_batch_data, i, label_name);

      // Print ASCII preview
      print_ascii_preview(val_batch_data, i);
    }

    // Verify data properties
    cout << "\n=== Data Validation Checks ===" << endl;

    // Check pixel range
    float train_min = train_batch_data.min();
    float train_max = train_batch_data.max();
    cout << "✓ Training pixel range: [" << train_min << ", " << train_max << "]" << endl;
    if (train_min < 0.0f || train_max > 1.0f) {
      cout << "  WARNING: Pixels outside expected [0, 1] range!" << endl;
    }

    float val_min = val_batch_data.min();
    float val_max = val_batch_data.max();
    cout << "✓ Validation pixel range: [" << val_min << ", " << val_max << "]" << endl;
    if (val_min < 0.0f || val_max > 1.0f) {
      cout << "  WARNING: Pixels outside expected [0, 1] range!" << endl;
    }

    // Check one-hot encoding
    bool all_one_hot = true;
    for (size_t i = 0; i < num_samples; ++i) {
      float sum = 0.0f;
      int num_ones = 0;
      for (size_t j = 0; j < 200; ++j) {
        float val = train_batch_labels(i, j, 0, 0);
        sum += val;
        if (val > 0.5f)
          num_ones++;
      }
      if (abs(sum - 1.0f) > 1e-5 || num_ones != 1) {
        all_one_hot = false;
        break;
      }
    }
    cout << "✓ One-hot encoding: " << (all_one_hot ? "PASS" : "FAIL") << endl;

    cout << "\n=== Inspection Complete ===" << endl;
    cout << "Check the 'output_images/' directory for saved PNG files." << endl;
    cout << "\nThings to verify in the images:" << endl;
    cout << "  1. Images should be recognizable objects/scenes" << endl;
    cout << "  2. Colors should look natural (not inverted or strange)" << endl;
    cout << "  3. Images should be 64x64 pixels" << endl;
    cout << "  4. Class labels should match the image content" << endl;
    cout << "  5. Training and validation images should look similar in quality" << endl;

  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  return 0;
}
