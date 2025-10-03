#pragma once

#include "tensor/tensor.hpp"
#include <algorithm>
#include <memory>
#include <random>
#include <vector>

namespace data_augmentation {

// Forward declarations
template <typename T> class Augmentation;
template <typename T> class AugmentationStrategy;

/**
 * Abstract base class for all augmentation operations
 */
template <typename T = float> class Augmentation {
public:
  Augmentation() = default;
  virtual ~Augmentation() = default;

  virtual void apply(Tensor<T> &data, Tensor<T> &labels) = 0;
  virtual std::unique_ptr<Augmentation<T>> clone() const = 0;

  void set_name(const std::string &name) { name_ = name; }
  std::string get_name() const { return name_; }

protected:
  std::string name_;
  mutable std::mt19937 rng_{std::random_device{}()};
};

} // namespace data_augmentation

// Include concrete augmentation implementations
#include "brightness.hpp"
#include "contrast.hpp"
#include "cutout.hpp"
#include "gaussian_noise.hpp"
#include "horizontal_flip.hpp"
#include "random_crop.hpp"
#include "rotation.hpp"
#include "vertical_flip.hpp"

namespace data_augmentation {

/**
 * Augmentation strategy that manages a pipeline of augmentations
 */
template <typename T = float> class AugmentationStrategy {
public:
  AugmentationStrategy() = default;

  AugmentationStrategy(const AugmentationStrategy &other) {
    for (const auto &aug : other.augmentations_) {
      augmentations_.emplace_back(aug->clone());
    }
  }

  /**
   * Apply augmentations in the order they were added
   */
  void apply(Tensor<T> &data, Tensor<T> &labels) {
    for (auto &aug : augmentations_) {
      aug->apply(data, labels);
    }
  }

  void add_augmentation(const Augmentation<T> &augmentation) {
    augmentations_.emplace_back(augmentation.clone());
  }

  void add_augmentation(std::unique_ptr<Augmentation<T>> augmentation) {
    augmentations_.emplace_back(std::move(augmentation));
  }

  void remove_augmentation(size_t index) {
    if (index >= augmentations_.size())
      return;
    augmentations_.erase(augmentations_.begin() + index);
  }

  void remove_augmentation(const std::string &name) {
    augmentations_.erase(std::remove_if(augmentations_.begin(), augmentations_.end(),
                                        [&name](const std::unique_ptr<Augmentation<T>> &aug) {
                                          return aug->get_name() == name;
                                        }),
                         augmentations_.end());
  }

  void set_augmentations(const std::vector<std::unique_ptr<Augmentation<T>>> &augs) {
    augmentations_.clear();
    for (const auto &aug : augs) {
      augmentations_.emplace_back(aug->clone());
    }
  }

  void clear_augmentations() { augmentations_.clear(); }

  size_t size() const { return augmentations_.size(); }

  const std::vector<std::unique_ptr<Augmentation<T>>> &get_augmentations() const {
    return augmentations_;
  }

protected:
  std::vector<std::unique_ptr<Augmentation<T>>> augmentations_;
};

/**
 * Builder pattern for creating augmentation strategies
 */
template <typename T = float> class AugmentationBuilder {
public:
  AugmentationBuilder() = default;

  AugmentationBuilder &horizontal_flip(float probability = 0.5f) {
    strategy_.add_augmentation(std::make_unique<HorizontalFlipAugmentation<T>>(probability));
    return *this;
  }

  AugmentationBuilder &vertical_flip(float probability = 0.5f) {
    strategy_.add_augmentation(std::make_unique<VerticalFlipAugmentation<T>>(probability));
    return *this;
  }

  AugmentationBuilder &rotation(float probability = 0.5f, float max_angle_degrees = 15.0f) {
    strategy_.add_augmentation(
        std::make_unique<RotationAugmentation<T>>(probability, max_angle_degrees));
    return *this;
  }

  AugmentationBuilder &brightness(float probability = 0.5f, float range = 0.2f) {
    strategy_.add_augmentation(std::make_unique<BrightnessAugmentation<T>>(probability, range));
    return *this;
  }

  AugmentationBuilder &contrast(float probability = 0.5f, float range = 0.2f) {
    strategy_.add_augmentation(std::make_unique<ContrastAugmentation<T>>(probability, range));
    return *this;
  }

  AugmentationBuilder &gaussian_noise(float probability = 0.3f, float std_dev = 0.05f) {
    strategy_.add_augmentation(
        std::make_unique<GaussianNoiseAugmentation<T>>(probability, std_dev));
    return *this;
  }

  AugmentationBuilder &random_crop(float probability = 0.5f, int padding = 4) {
    strategy_.add_augmentation(std::make_unique<RandomCropAugmentation<T>>(probability, padding));
    return *this;
  }

  AugmentationBuilder &cutout(float probability = 0.5f, int cutout_size = 8) {
    strategy_.add_augmentation(std::make_unique<CutoutAugmentation<T>>(probability, cutout_size));
    return *this;
  }

  AugmentationBuilder &custom_augmentation(std::unique_ptr<Augmentation<T>> augmentation) {
    strategy_.add_augmentation(std::move(augmentation));
    return *this;
  }

  std::unique_ptr<AugmentationStrategy<T>> build() {
    return std::make_unique<AugmentationStrategy<T>>(strategy_);
  }

private:
  AugmentationStrategy<T> strategy_;
};

} // namespace data_augmentation