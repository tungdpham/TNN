#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <memory>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>
#include "../tensor/tensor.hpp"

namespace data_loading {

/**
 * Abstract base class for all data loaders
 * Provides common interface and functionality for training neural networks
 */
template<typename T = float>
class BaseDataLoader {
public:
    virtual ~BaseDataLoader() = default;
    
    
    
    /**
     * Load data from file(s)
     * @param source Data source (file path, directory, etc.)
     * @return true if successful, false otherwise
     */
    virtual bool load_data(const std::string& source) = 0;
    
    /**
     * Get the next batch of data
     * @param batch_data Output tensor for features/input data
     * @param batch_labels Output tensor for labels/targets
     * @return true if batch was retrieved, false if no more data
     */
    virtual bool get_next_batch(Tensor<T>& batch_data, Tensor<T>& batch_labels) = 0;
    
    /**
     * Get a specific batch size
     * @param batch_size Number of samples per batch
     * @param batch_data Output tensor for features/input data
     * @param batch_labels Output tensor for labels/targets
     * @return true if batch was retrieved, false if no more data
     */
    virtual bool get_batch(int batch_size, Tensor<T>& batch_data, Tensor<T>& batch_labels) = 0;
    
    /**
     * Reset iterator to beginning of dataset
     */
    virtual void reset() = 0;
    
    /**
     * Shuffle the dataset
     */
    virtual void shuffle() = 0;
    
    /**
     * Get the total number of samples in the dataset
     */
    virtual size_t size() const = 0;
    
    
    
    /**
     * Prepare batches for efficient training
     * @param batch_size Size of each batch
     */
    virtual void prepare_batches(int batch_size) {
        if (size() == 0) {
            std::cerr << "Warning: Cannot prepare batches - no data loaded" << std::endl;
            return;
        }
        
        batch_size_ = batch_size;
        batched_prepared_ = true;
        current_batch_index_ = 0;
        
        std::cout << "Preparing batches with size " << batch_size 
                  << " for " << size() << " samples..." << std::endl;
    }
    
    /**
     * Get number of batches when using prepared batches
     */
    virtual size_t num_batches() const {
        if (!batched_prepared_ || size() == 0) return 0;
        return (size() + batch_size_ - 1) / batch_size_; 
    }
    
    /**
     * Check if batches are prepared
     */
    virtual bool are_batches_prepared() const {
        return batched_prepared_;
    }
    
    /**
     * Get current batch size
     */
    virtual int get_batch_size() const {
        return batch_size_;
    }
    
    /**
     * Set random seed for reproducible shuffling
     */
    virtual void set_seed(unsigned int seed) {
        rng_.seed(seed);
    }
    
    /**
     * Get random number generator for derived classes
     */
    std::mt19937& get_rng() { return rng_; }
    const std::mt19937& get_rng() const { return rng_; }

protected:
    
    size_t current_index_ = 0;
    size_t current_batch_index_ = 0;
    int batch_size_ = 32;
    bool batched_prepared_ = false;
    mutable std::mt19937 rng_{std::random_device{}()};
    
    /**
     * Utility function to shuffle indices
     */
    std::vector<size_t> generate_shuffled_indices(size_t data_size) const {
        std::vector<size_t> indices(data_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng_);
        return indices;
    }
    
    /**
     * Utility function to create one-hot encoded labels
     */
    void create_one_hot_label(Tensor<T>& label_tensor, int batch_idx, int class_idx, int num_classes) {
        
        label_tensor(batch_idx, class_idx, 0, 0) = static_cast<T>(1.0);
    }
    
    /**
     * Utility function to normalize data to [0, 1] range
     */
    void normalize_to_unit_range(std::vector<T>& data, T max_value = static_cast<T>(255.0)) {
        for (auto& value : data) {
            value /= max_value;
        }
    }
};

/**
 * Specialized base class for image classification datasets
 * Provides common functionality for image-based datasets like MNIST, CIFAR, etc.
 */
template<typename T = float>
class ImageClassificationDataLoader : public BaseDataLoader<T> {
public:
    virtual ~ImageClassificationDataLoader() = default;
    
    /**
     * Get image dimensions
     */
    virtual std::vector<size_t> get_image_shape() const = 0;
    
    /**
     * Get number of classes
     */
    virtual int get_num_classes() const = 0;
    
    /**
     * Get class names (optional)
     */
    virtual std::vector<std::string> get_class_names() const {
        std::vector<std::string> names;
        int num_classes = get_num_classes();
        names.reserve(num_classes);
        for (int i = 0; i < num_classes; ++i) {
            names.push_back("class_" + std::to_string(i));
        }
        return names;
    }

protected:
    using BaseDataLoader<T>::current_index_;
    using BaseDataLoader<T>::batch_size_;
    using BaseDataLoader<T>::rng_;
    
    /**
     * Utility to copy image data to tensor with proper channel ordering
     */
    void copy_image_to_tensor(Tensor<T>& tensor, int batch_idx, 
                             const std::vector<T>& image_data,
                             const std::vector<size_t>& shape) {
        size_t channels = shape[0];
        size_t height = shape[1];
        size_t width = shape[2];
        
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    size_t idx = c * height * width + h * width + w;
                    tensor(batch_idx, c, h, w) = image_data[idx];
                }
            }
        }
    }
};

/**
 * Specialized base class for regression datasets
 * Provides common functionality for continuous target prediction
 */
template<typename T = float>
class RegressionDataLoader : public BaseDataLoader<T> {
public:
    virtual ~RegressionDataLoader() = default;
    
    /**
     * Get number of input features
     */
    virtual size_t get_num_features() const = 0;
    
    /**
     * Get number of output targets
     */
    virtual size_t get_num_outputs() const = 0;
    
    /**
     * Check if data is normalized
     */
    virtual bool is_normalized() const = 0;
    
    /**
     * Get feature normalization statistics (optional)
     */
    virtual std::vector<T> get_feature_means() const { return {}; }
    virtual std::vector<T> get_feature_stds() const { return {}; }
    
    /**
     * Get target normalization statistics (optional)
     */
    virtual std::vector<T> get_target_means() const { return {}; }
    virtual std::vector<T> get_target_stds() const { return {}; }

protected:
    using BaseDataLoader<T>::current_index_;
    using BaseDataLoader<T>::batch_size_;
    using BaseDataLoader<T>::rng_;
};

/**
 * Factory function to create appropriate data loader based on dataset type
 */
template<typename T = float>
std::unique_ptr<BaseDataLoader<T>> create_data_loader(const std::string& dataset_type) {
    
    
    return nullptr;
}

/**
 * Utility functions for common data loading operations
 */
namespace utils {
    /**
     * Split dataset into train/validation sets
     */
    template<typename T>
    std::pair<std::vector<size_t>, std::vector<size_t>> 
    train_val_split(size_t dataset_size, float val_ratio = 0.2f, unsigned int seed = 42) {
        std::vector<size_t> indices(dataset_size);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        size_t val_size = static_cast<size_t>(dataset_size * val_ratio);
        size_t train_size = dataset_size - val_size;
        
        std::vector<size_t> train_indices(indices.begin(), indices.begin() + train_size);
        std::vector<size_t> val_indices(indices.begin() + train_size, indices.end());
        
        return {train_indices, val_indices};
    }
    
    /**
     * Calculate dataset statistics
     */
    template<typename T>
    struct DatasetStats {
        std::vector<T> means;
        std::vector<T> stds;
        T min_val;
        T max_val;
    };
    
    template<typename T>
    DatasetStats<T> calculate_stats(const std::vector<std::vector<T>>& data) {
        if (data.empty()) return {};
        
        size_t num_features = data[0].size();
        size_t num_samples = data.size();
        
        DatasetStats<T> stats;
        stats.means.resize(num_features, 0);
        stats.stds.resize(num_features, 0);
        stats.min_val = std::numeric_limits<T>::max();
        stats.max_val = std::numeric_limits<T>::lowest();
        
        
        for (const auto& sample : data) {
            for (size_t i = 0; i < num_features; ++i) {
                stats.means[i] += sample[i];
                stats.min_val = std::min(stats.min_val, sample[i]);
                stats.max_val = std::max(stats.max_val, sample[i]);
            }
        }
        
        for (auto& mean : stats.means) {
            mean /= static_cast<T>(num_samples);
        }
        
        
        for (const auto& sample : data) {
            for (size_t i = 0; i < num_features; ++i) {
                T diff = sample[i] - stats.means[i];
                stats.stds[i] += diff * diff;
            }
        }
        
        for (auto& std_val : stats.stds) {
            std_val = std::sqrt(std_val / static_cast<T>(num_samples));
        }
        
        return stats;
    }
} 

} 
