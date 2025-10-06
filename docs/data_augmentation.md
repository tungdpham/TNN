# Data Augmentation Usage Guide

This guide covers how to use the data augmentation system for training data preprocessing and how to implement custom augmentation techniques.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Available Augmentations](#available-augmentations)
- [Advanced Usage](#advanced-usage)
- [Custom Augmentation Implementation](#custom-augmentation-implementation)
- [Performance Considerations](#performance-considerations)

## Basic Usage

### Simple Augmentation Pipeline
```cpp
#include "data_augmentation/augmentation.hpp"

using namespace data_augmentation;

// Create augmentation strategy using builder pattern
auto aug_strategy = AugmentationBuilder<float>()
    .horizontal_flip(0.5f)
    .rotation(0.3f, 15.0f)
    .brightness(0.4f, 0.2f)
    .build();

// Apply to data loader
train_loader.set_augmentation(std::move(aug_strategy));
```

### Manual Strategy Construction
```cpp
// Create strategy manually
AugmentationStrategy<float> strategy;

// Add individual augmentations
strategy.add_augmentation(std::make_unique<HorizontalFlipAugmentation<float>>(0.5f));
strategy.add_augmentation(std::make_unique<BrightnessAugmentation<float>>(0.4f, 0.2f));

// Apply directly to tensors
strategy.apply(data_tensor, labels_tensor);
```

## Available Augmentations

### Geometric Transformations

#### Horizontal Flip
```cpp
.horizontal_flip(probability)
```
**Parameters:**
- `probability`: Probability of applying the transformation (default: 0.5)

#### Vertical Flip
```cpp
.vertical_flip(probability)
```
**Parameters:**
- `probability`: Probability of applying the transformation (default: 0.5)

#### Rotation
```cpp
.rotation(probability, max_angle_degrees)
```
**Parameters:**
- `probability`: Probability of applying the transformation (default: 0.5)
- `max_angle_degrees`: Maximum rotation angle in degrees (default: 15.0)

#### Random Crop
```cpp
.random_crop(probability, padding)
```
**Parameters:**
- `probability`: Probability of applying the transformation (default: 0.5)
- `padding`: Number of pixels to pad before cropping (default: 4)

### Color Transformations

#### Brightness Adjustment
```cpp
.brightness(probability, range)
```
**Parameters:**
- `probability`: Probability of applying the transformation (default: 0.5)
- `range`: Maximum brightness adjustment range (default: 0.2)

#### Contrast Adjustment
```cpp
.contrast(probability, range)
```
**Parameters:**
- `probability`: Probability of applying the transformation (default: 0.5)
- `range`: Maximum contrast adjustment range (default: 0.2)

### Noise and Occlusion

#### Gaussian Noise
```cpp
.gaussian_noise(probability, std_dev)
```
**Parameters:**
- `probability`: Probability of applying the transformation (default: 0.3)
- `std_dev`: Standard deviation of Gaussian noise (default: 0.05)

#### Cutout
```cpp
.cutout(probability, cutout_size)
```
**Parameters:**
- `probability`: Probability of applying the transformation (default: 0.5)
- `cutout_size`: Size of the square cutout region (default: 8)

## Advanced Usage

### Strategy Management
```cpp
AugmentationStrategy<float> strategy;

// Add multiple augmentations
strategy.add_augmentation(std::make_unique<HorizontalFlipAugmentation<float>>(0.5f));
strategy.add_augmentation(std::make_unique<RotationAugmentation<float>>(0.3f, 10.0f));

// Remove by index
strategy.remove_augmentation(0);

// Remove by name
strategy.remove_augmentation("HorizontalFlip");

// Clear all augmentations
strategy.clear_augmentations();

// Get number of augmentations
size_t count = strategy.size();
```

### Custom Augmentation Integration
```cpp
// Add custom augmentation to builder
auto strategy = AugmentationBuilder<float>()
    .horizontal_flip(0.5f)
    .custom_augmentation(std::make_unique<MyCustomAugmentation<float>>(params))
    .brightness(0.3f, 0.15f)
    .build();
```

### Conditional Augmentation
```cpp
// Different strategies for different scenarios
auto light_aug = AugmentationBuilder<float>()
    .horizontal_flip(0.5f)
    .brightness(0.3f, 0.1f)
    .build();

auto heavy_aug = AugmentationBuilder<float>()
    .horizontal_flip(0.5f)
    .vertical_flip(0.2f)
    .rotation(0.4f, 20.0f)
    .brightness(0.4f, 0.2f)
    .contrast(0.4f, 0.2f)
    .gaussian_noise(0.3f, 0.05f)
    .cutout(0.3f, 12)
    .build();
```

## Custom Augmentation Implementation

### Basic Template
```cpp
#pragma once

#include "augmentation.hpp"
#include <random>

namespace data_augmentation {

template <typename T = float>
class MyCustomAugmentation : public Augmentation<T> {
public:
    explicit MyCustomAugmentation(float probability = 0.5f, /* other params */)
        : probability_(probability) /* initialize other members */ {
        this->name_ = "MyCustomAugmentation";
    }

    void apply(Tensor<T> &data, Tensor<T> &labels) override {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        const auto shape = data.shape();
        if (shape.size() != 4) return; // Expected: [batch, channels, height, width]
        
        const size_t batch_size = shape[0];
        const size_t channels = shape[1];
        const size_t height = shape[2];
        const size_t width = shape[3];
        
        for (size_t b = 0; b < batch_size; ++b) {
            if (dist(this->rng_) < probability_) {
                // Apply your transformation here
                apply_transformation(data, b, channels, height, width);
            }
        }
    }

    std::unique_ptr<Augmentation<T>> clone() const override {
        return std::make_unique<MyCustomAugmentation<T>>(probability_ /* other params */);
    }

private:
    float probability_;
    // Other parameters
    
    void apply_transformation(Tensor<T> &data, size_t batch_idx, 
                            size_t channels, size_t height, size_t width) {
        // Implementation details
    }
};

} // namespace data_augmentation
```

### Implementation Guidelines

#### 1. Inherit from Augmentation Base Class
```cpp
template <typename T = float>
class MyAugmentation : public Augmentation<T> {
    // Implementation
};
```

#### 2. Set Augmentation Name
```cpp
MyAugmentation() {
    this->name_ = "UniqueAugmentationName";
}
```

#### 3. Implement Required Methods
```cpp
// Apply transformation to data and labels
void apply(Tensor<T> &data, Tensor<T> &labels) override;

// Create a copy of the augmentation
std::unique_ptr<Augmentation<T>> clone() const override;
```

#### 4. Use Random Number Generator
```cpp
// Access the inherited RNG
std::uniform_real_distribution<float> dist(0.0f, 1.0f);
if (dist(this->rng_) < probability_) {
    // Apply transformation
}
```

#### 5. Handle Tensor Shape Validation
```cpp
const auto shape = data.shape();
if (shape.size() != 4) return; // Expected format validation

// Extract dimensions
const size_t batch_size = shape[0];
const size_t channels = shape[1];
const size_t height = shape[2];
const size_t width = shape[3];
```

### Integration with Builder Pattern

#### 1. Add to AugmentationBuilder
```cpp
// In augmentation.hpp AugmentationBuilder class
AugmentationBuilder &my_custom_augmentation(float probability = 0.5f, /* other params */) {
    strategy_.add_augmentation(std::make_unique<MyCustomAugmentation<T>>(probability /* other params */));
    return *this;
}
```

#### 2. Include Header
```cpp
// In augmentation.hpp after other includes
#include "my_custom_augmentation.hpp"
```

### Example: Color Inversion Augmentation
```cpp
#pragma once

#include "augmentation.hpp"
#include <random>

namespace data_augmentation {

template <typename T = float>
class ColorInversionAugmentation : public Augmentation<T> {
public:
    explicit ColorInversionAugmentation(float probability = 0.3f)
        : probability_(probability) {
        this->name_ = "ColorInversion";
    }

    void apply(Tensor<T> &data, Tensor<T> &labels) override {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        const auto shape = data.shape();
        if (shape.size() != 4) return;
        
        const size_t batch_size = shape[0];
        
        for (size_t b = 0; b < batch_size; ++b) {
            if (dist(this->rng_) < probability_) {
                // Invert pixel values (assuming normalized data [0,1])
                for (size_t i = 0; i < data.size() / batch_size; ++i) {
                    size_t idx = b * (data.size() / batch_size) + i;
                    data.data()[idx] = static_cast<T>(1.0) - data.data()[idx];
                }
            }
        }
    }

    std::unique_ptr<Augmentation<T>> clone() const override {
        return std::make_unique<ColorInversionAugmentation<T>>(probability_);
    }

private:
    float probability_;
};

} // namespace data_augmentation
```

## Performance Considerations

### Memory Usage
- Augmentations operate in-place on tensor data
- No additional memory allocation during transformation
- Cloning creates lightweight copies of augmentation objects

### Threading
- Each augmentation instance has its own random number generator
- Thread-safe when used with separate strategy instances
- Avoid sharing strategy instances across threads

### Computational Overhead
- Probability checks are performed per batch item
- Transformations only applied when probability condition is met
- Consider probability values based on training requirements

### Best Practices
1. **Order Matters**: Place computationally expensive augmentations with lower probabilities
2. **Probability Tuning**: Start with conservative probabilities and adjust based on validation performance
3. **Data Range**: Ensure transformations maintain expected data ranges (e.g., [0,1] for normalized images)
4. **Validation**: Test custom augmentations with known inputs to verify correctness