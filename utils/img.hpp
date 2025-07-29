#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../matrix/matrix.hpp"

namespace ImageUtils {

// Structure to hold image data and label
struct ImageData {
  Matrix image;
  int label;

  ImageData(const Matrix &img, int lbl) : image(img), label(lbl) {}
};

// Convert integer label to one-hot encoded vector
inline Matrix label_to_one_hot(int label, int num_classes = 10) {
  Matrix one_hot(1, num_classes, 1);
  one_hot.fill(0.0);
  if (label >= 0 && label < num_classes) {
    one_hot(0, label, 0) = 1.0;
  }
  return one_hot;
}

// Convert one-hot encoded vector to integer label
inline int one_hot_to_label(const Matrix &one_hot) {
  int max_idx = 0;
  double max_val = one_hot(0, 0, 0);

  for (int i = 1; i < one_hot.cols; ++i) {
    if (one_hot(0, i, 0) > max_val) {
      max_val = one_hot(0, i, 0);
      max_idx = i;
    }
  }

  return max_idx;
}

// Load MNIST CSV data
inline std::vector<ImageData> load_mnist_csv(const std::string &filename,
                                             bool normalize = true,
                                             int max_samples = -1) {
  std::vector<ImageData> dataset;
  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  std::string line;
  bool first_line = true;
  int samples_loaded = 0;

  std::cout << "Loading MNIST data from: " << filename << std::endl;

  while (std::getline(file, line) &&
         (max_samples == -1 || samples_loaded < max_samples)) {
    if (first_line) {
      first_line = false;
      continue; // Skip header
    }

    std::stringstream ss(line);
    std::string value;
    std::vector<double> values;

    // Parse CSV line
    while (std::getline(ss, value, ',')) {
      values.push_back(std::stod(value));
    }

    if (values.size() != 785) { // 1 label + 784 pixels (28x28)
      std::cerr << "Warning: Invalid line with " << values.size()
                << " values, skipping." << std::endl;
      continue;
    }

    // Extract label and pixels
    int label = static_cast<int>(values[0]);
    Matrix image(28, 28, 1);

    // Fill image matrix (pixels are in row-major order)
    for (int i = 0; i < 784; ++i) {
      int row = i / 28;
      int col = i % 28;
      double pixel_val = values[i + 1];

      // Normalize to [0, 1] if requested
      if (normalize) {
        pixel_val /= 255.0;
      }

      image(row, col, 0) = pixel_val;
    }

    dataset.emplace_back(image, label);
    samples_loaded++;

    if (samples_loaded % 10000 == 0) {
      std::cout << "Loaded " << samples_loaded << " samples..." << std::endl;
    }
  }

  file.close();
  std::cout << "Successfully loaded " << dataset.size() << " samples from "
            << filename << std::endl;
  return dataset;
}

// Convert dataset to separate images and labels vectors
inline std::pair<std::vector<Matrix>, std::vector<Matrix>>
split_dataset(const std::vector<ImageData> &dataset, int num_classes = 10) {
  std::vector<Matrix> images;
  std::vector<Matrix> labels;

  images.reserve(dataset.size());
  labels.reserve(dataset.size());

  for (const auto &data : dataset) {
    images.push_back(data.image);
    labels.push_back(label_to_one_hot(data.label, num_classes));
  }

  return {images, labels};
}

// Shuffle dataset randomly
inline void shuffle_dataset(std::vector<ImageData> &dataset, int seed = 42) {
  std::mt19937 rng(seed);
  std::shuffle(dataset.begin(), dataset.end(), rng);
}

// Print ASCII art representation of MNIST digit
inline void print_mnist_digit(const Matrix &image, int label = -1) {
  if (image.rows != 28 || image.cols != 28) {
    std::cout << "Error: Expected 28x28 image" << std::endl;
    return;
  }

  if (label >= 0) {
    std::cout << "Label: " << label << std::endl;
  }

  std::cout << "+" << std::string(28, '-') << "+" << std::endl;

  for (int r = 0; r < 28; ++r) {
    std::cout << "|";
    for (int c = 0; c < 28; ++c) {
      double pixel = image(r, c, 0);
      // Convert to ASCII based on intensity
      if (pixel < 0.1)
        std::cout << " ";
      else if (pixel < 0.3)
        std::cout << ".";
      else if (pixel < 0.5)
        std::cout << ":";
      else if (pixel < 0.7)
        std::cout << "o";
      else if (pixel < 0.9)
        std::cout << "O";
      else
        std::cout << "#";
    }
    std::cout << "|" << std::endl;
  }

  std::cout << "+" << std::string(28, '-') << "+" << std::endl;
}

// Data augmentation functions
inline Matrix rotate_image(const Matrix &image, double angle_degrees) {
  // Simple rotation by 90-degree increments for now
  Matrix rotated = image;
  int times = static_cast<int>(angle_degrees / 90.0) % 4;

  for (int t = 0; t < times; ++t) {
    Matrix temp(image.cols, image.rows, image.channels);
    for (int r = 0; r < image.rows; ++r) {
      for (int c = 0; c < image.cols; ++c) {
        temp(c, image.rows - 1 - r, 0) = rotated(r, c, 0);
      }
    }
    rotated = temp;
  }

  return rotated;
}

inline Matrix add_noise(const Matrix &image, double noise_level = 0.1,
                        int seed = 42) {
  Matrix noisy = image;
  std::mt19937 rng(seed);
  std::normal_distribution<double> noise(0.0, noise_level);

  for (int i = 0; i < noisy.size(); ++i) {
    noisy.data[i] = std::max(0.0, std::min(1.0, noisy.data[i] + noise(rng)));
  }

  return noisy;
}

inline Matrix shift_image(const Matrix &image, int dx, int dy) {
  Matrix shifted(image.rows, image.cols, image.channels);
  shifted.fill(0.0);

  for (int r = 0; r < image.rows; ++r) {
    for (int c = 0; c < image.cols; ++c) {
      int new_r = r + dy;
      int new_c = c + dx;

      if (new_r >= 0 && new_r < image.rows && new_c >= 0 &&
          new_c < image.cols) {
        shifted(new_r, new_c, 0) = image(r, c, 0);
      }
    }
  }

  return shifted;
}

// Generate augmented dataset
inline std::vector<ImageData>
augment_dataset(const std::vector<ImageData> &dataset, int augment_factor = 2,
                bool use_rotation = true, bool use_noise = true,
                bool use_shift = true) {
  std::vector<ImageData> augmented = dataset; // Start with original
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> shift_dist(-2, 2);
  std::uniform_real_distribution<double> noise_dist(0.05, 0.15);

  std::cout << "Augmenting dataset from " << dataset.size() << " samples..."
            << std::endl;

  for (int aug = 0; aug < augment_factor; ++aug) {
    for (const auto &data : dataset) {
      Matrix augmented_image = data.image;

      if (use_shift) {
        int dx = shift_dist(rng);
        int dy = shift_dist(rng);
        augmented_image = shift_image(augmented_image, dx, dy);
      }

      if (use_noise) {
        double noise_level = noise_dist(rng);
        augmented_image = add_noise(augmented_image, noise_level);
      }

      if (use_rotation && rng() % 10 == 0) { // 10% chance of rotation
        augmented_image = rotate_image(augmented_image, 90);
      }

      augmented.emplace_back(augmented_image, data.label);
    }
  }

  std::cout << "Augmented dataset to " << augmented.size() << " samples"
            << std::endl;
  return augmented;
}

// Calculate dataset statistics
inline void print_dataset_stats(const std::vector<ImageData> &dataset) {
  if (dataset.empty()) {
    std::cout << "Dataset is empty!" << std::endl;
    return;
  }

  // Count labels
  std::vector<int> label_counts(10, 0);
  double pixel_sum = 0.0;
  double pixel_sum_sq = 0.0;
  int total_pixels = 0;

  for (const auto &data : dataset) {
    label_counts[data.label]++;

    for (int i = 0; i < data.image.size(); ++i) {
      double pixel = data.image.data[i];
      pixel_sum += pixel;
      pixel_sum_sq += pixel * pixel;
      total_pixels++;
    }
  }

  double pixel_mean = pixel_sum / total_pixels;
  double pixel_std =
      sqrt(pixel_sum_sq / total_pixels - pixel_mean * pixel_mean);

  std::cout << "\nDataset Statistics:" << std::endl;
  std::cout << "==================" << std::endl;
  std::cout << "Total samples: " << dataset.size() << std::endl;
  std::cout << "Image size: " << dataset[0].image.rows << "x"
            << dataset[0].image.cols << std::endl;
  std::cout << "Pixel mean: " << pixel_mean << std::endl;
  std::cout << "Pixel std: " << pixel_std << std::endl;
  std::cout << "\nLabel distribution:" << std::endl;

  for (int i = 0; i < 10; ++i) {
    double percentage = 100.0 * label_counts[i] / dataset.size();
    std::cout << "  " << i << ": " << label_counts[i] << " (" << percentage
              << "%)" << std::endl;
  }
  std::cout << std::endl;
}

// Save sample images for visualization
inline void save_sample_images(const std::vector<ImageData> &dataset,
                               const std::string &output_dir = "samples",
                               int num_samples = 10) {
  std::cout << "Saving " << num_samples << " sample images..." << std::endl;

  for (int i = 0; i < std::min(num_samples, (int)dataset.size()); ++i) {
    std::cout << "\nSample " << i << ":" << std::endl;
    print_mnist_digit(dataset[i].image, dataset[i].label);
  }
}

} // namespace ImageUtils