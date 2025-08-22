#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <omp.h>
#include "data_loader.hpp"
#include "../tensor/tensor.hpp"

// Constants for MNIST dataset
namespace mnist_constants {
    constexpr size_t IMAGE_HEIGHT = 28;
    constexpr size_t IMAGE_WIDTH = 28;
    constexpr size_t IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
    constexpr size_t NUM_CLASSES = 10;
    constexpr size_t NUM_CHANNELS = 1;
    constexpr float NORMALIZATION_FACTOR = 255.0f;
}

namespace data_loading {

/**
 * Enhanced MNIST data loader for CSV format adapted for CNN (2D images)
 * Extends ImageClassificationDataLoader for proper inheritance
 */
template<typename T = float>
class MNISTDataLoader : public ImageClassificationDataLoader<T> {
private:
    std::vector<std::vector<T>> data_;
    std::vector<int> labels_;
    
    // Pre-computed batch storage for efficiency
    std::vector<Tensor<T>> batched_data_;
    std::vector<Tensor<T>> batched_labels_;
    bool batches_prepared_;

public:
    MNISTDataLoader() : ImageClassificationDataLoader<T>(), batches_prepared_(false) {
        // Reserve space to reduce allocations
        data_.reserve(60000); // MNIST train set size
        labels_.reserve(60000);
    }

    virtual ~MNISTDataLoader() = default;

    /**
     * Load MNIST data from CSV file
     * @param source Path to CSV file (train.csv or test.csv)
     * @return true if successful, false otherwise
     */
    bool load_data(const std::string& source) override {
        std::ifstream file{source};
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << source << std::endl;
            return false;
        }

        std::string line;
        line.reserve(3136); // Reserve space for MNIST line (784 pixels + label)
        
        // Skip header line
        if (!std::getline(file, line)) {
            std::cerr << "Error: Empty file " << source << std::endl;
            return false;
        }

        data_.clear();
        labels_.clear();

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;

            // First column is label
            if (!std::getline(ss, cell, ',')) continue;
            labels_.push_back(std::stoi(cell));

            // Remaining columns are pixel values - reserve space for efficiency
            std::vector<T> row;
            row.reserve(mnist_constants::IMAGE_SIZE);
            
            while (std::getline(ss, cell, ',')) {
                row.push_back(static_cast<T>(std::stod(cell) / mnist_constants::NORMALIZATION_FACTOR));
            }
            
            // Validate image size
            if (row.size() != mnist_constants::IMAGE_SIZE) {
                std::cerr << "Warning: Invalid image size " << row.size() 
                          << " expected " << mnist_constants::IMAGE_SIZE << std::endl;
                continue;
            }
            
            data_.push_back(std::move(row));
        }

        this->current_index_ = 0;
        std::cout << "Loaded " << data_.size() << " samples from " << source << std::endl;
        return !data_.empty();
    }

    /**
     * Get the next batch of data using pre-computed batches
     */
    bool get_next_batch(Tensor<T>& batch_data, Tensor<T>& batch_labels) override {
        if (!batches_prepared_) {
            std::cerr << "Error: Batches not prepared! Call prepare_batches() first." << std::endl;
            return false;
        }
        
        if (this->current_batch_index_ >= batched_data_.size()) {
            return false; // No more batches
        }
        
        batch_data = batched_data_[this->current_batch_index_];
        batch_labels = batched_labels_[this->current_batch_index_];
        ++this->current_batch_index_;
        
        return true;
    }

    /**
     * Get a specific batch size (supports both pre-computed and on-demand batches)
     */
    bool get_batch(int batch_size, Tensor<T>& batch_data, Tensor<T>& batch_labels) override {
        // If batches are prepared and batch size matches, use fast path
        if (batches_prepared_ && batch_size == this->batch_size_) {
            return get_next_batch(batch_data, batch_labels);
        }
        
        // Fallback to original implementation for different batch sizes
        if (this->current_index_ >= data_.size()) {
            return false; // No more data
        }

        const int actual_batch_size = std::min(batch_size, static_cast<int>(data_.size() - this->current_index_));

        // Create batch data tensor for CNN: (batch_size, channels=1, height=28, width=28)
        batch_data = Tensor<T>(std::vector<size_t>{
            static_cast<size_t>(actual_batch_size), 
            mnist_constants::NUM_CHANNELS,
            mnist_constants::IMAGE_HEIGHT, 
            mnist_constants::IMAGE_WIDTH
        });

        // Create batch labels tensor (batch_size, 10, 1, 1) - one-hot encoded
        batch_labels = Tensor<T>(std::vector<size_t>{
            static_cast<size_t>(actual_batch_size), 
            mnist_constants::NUM_CLASSES, 
            1, 1
        });
        batch_labels.fill(static_cast<T>(0.0));

        // Parallelize batch processing for better performance
        #pragma omp parallel for if(actual_batch_size > 16)
        for (int i = 0; i < actual_batch_size; ++i) {
            const auto& image_data = data_[this->current_index_ + i];
            
            // Copy pixel data and reshape to 28x28 with better cache locality
            for (int h = 0; h < static_cast<int>(mnist_constants::IMAGE_HEIGHT); ++h) {
                for (int w = 0; w < static_cast<int>(mnist_constants::IMAGE_WIDTH); ++w) {
                    batch_data(i, 0, h, w) = image_data[h * mnist_constants::IMAGE_WIDTH + w];
                }
            }

            // Set one-hot label
            const int label = labels_[this->current_index_ + i];
            if (label >= 0 && label < static_cast<int>(mnist_constants::NUM_CLASSES)) {
                batch_labels(i, label, 0, 0) = static_cast<T>(1.0);
            }
        }

        this->current_index_ += actual_batch_size;
        return true;
    }

    /**
     * Reset iterator to beginning of dataset
     */
    void reset() override {
        this->current_index_ = 0;
        this->current_batch_index_ = 0;
    }

    /**
     * Shuffle the dataset
     */
    void shuffle() override {
        if (data_.empty()) return;
        
        std::vector<size_t> indices = this->generate_shuffled_indices(data_.size());

        // Use move semantics for better performance
        std::vector<std::vector<T>> shuffled_data;
        std::vector<int> shuffled_labels;
        shuffled_data.reserve(data_.size());
        shuffled_labels.reserve(labels_.size());

        for (const auto& idx : indices) {
            shuffled_data.emplace_back(std::move(data_[idx]));
            shuffled_labels.emplace_back(labels_[idx]);
        }

        data_ = std::move(shuffled_data);
        labels_ = std::move(shuffled_labels);
        this->current_index_ = 0;
        
        // Invalidate pre-computed batches since data order changed
        batches_prepared_ = false;
    }

    /**
     * Get the total number of samples in the dataset
     */
    size_t size() const override {
        return data_.size();
    }

    /**
     * Get image dimensions (channels, height, width)
     */
    std::vector<size_t> get_image_shape() const override {
        return {mnist_constants::NUM_CHANNELS, mnist_constants::IMAGE_HEIGHT, mnist_constants::IMAGE_WIDTH};
    }

    /**
     * Get number of classes
     */
    int get_num_classes() const override {
        return static_cast<int>(mnist_constants::NUM_CLASSES);
    }

    /**
     * Get class names for MNIST (digits 0-9)
     */
    std::vector<std::string> get_class_names() const override {
        std::vector<std::string> names;
        names.reserve(mnist_constants::NUM_CLASSES);
        for (int i = 0; i < static_cast<int>(mnist_constants::NUM_CLASSES); ++i) {
            names.push_back("digit_" + std::to_string(i));
        }
        return names;
    }

    /**
     * Pre-compute all batches for efficient training
     */
    void prepare_batches(int batch_size) override {
        if (data_.empty()) {
            std::cerr << "Warning: No data loaded, cannot prepare batches!" << std::endl;
            return;
        }
        
        this->batch_size_ = batch_size;
        this->batched_prepared_ = true;
        batched_data_.clear();
        batched_labels_.clear();
        
        const size_t num_samples = data_.size();
        const size_t num_batches = (num_samples + batch_size - 1) / batch_size; // Ceiling division
        
        batched_data_.reserve(num_batches);
        batched_labels_.reserve(num_batches);
        
        std::cout << "Preparing " << num_batches << " batches of size " << batch_size << "..." << std::endl;
        
        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            const size_t start_idx = batch_idx * batch_size;
            const size_t end_idx = std::min(start_idx + batch_size, num_samples);
            const int actual_batch_size = static_cast<int>(end_idx - start_idx);
            assert(actual_batch_size > 0);
            
            // Create batch tensors
            Tensor<T> batch_data(std::vector<size_t>{
                static_cast<size_t>(actual_batch_size), 
                mnist_constants::NUM_CHANNELS,
                mnist_constants::IMAGE_HEIGHT, 
                mnist_constants::IMAGE_WIDTH
            });
            
            Tensor<T> batch_labels(std::vector<size_t>{
                static_cast<size_t>(actual_batch_size), 
                mnist_constants::NUM_CLASSES, 
                1, 1
            });
            batch_labels.fill(T(0.0)); 
            
            // Fill batch data in parallel
            #pragma omp parallel for if(actual_batch_size > 16)
            for (int i = 0; i < actual_batch_size; ++i) {
                const size_t sample_idx = start_idx + i;
                const auto& image_data = data_[sample_idx];
                
                // Copy pixel data and reshape to 28x28 with better cache locality
                for (int h = 0; h < static_cast<int>(mnist_constants::IMAGE_HEIGHT); ++h) {
                    for (int w = 0; w < static_cast<int>(mnist_constants::IMAGE_WIDTH); ++w) {
                        batch_data(i, 0, h, w) = image_data[h * mnist_constants::IMAGE_WIDTH + w];
                    }
                }
                
                // Set one-hot label
                const int label = labels_[sample_idx];
                if (label >= 0 && label < static_cast<int>(mnist_constants::NUM_CLASSES)) {
                    batch_labels(i, label, 0, 0) = static_cast<T>(1.0);
                }
            }
            
            batched_data_.emplace_back(std::move(batch_data));
            batched_labels_.emplace_back(std::move(batch_labels));
        }
        
        this->current_batch_index_ = 0;
        batches_prepared_ = true;
        std::cout << "Batch preparation completed!" << std::endl;
        std::cout << "Prepared " << batched_data_.size() << " batches "
                  << "of size " << batch_size << " each." << std::endl;
    }

    /**
     * Get number of batches when using prepared batches
     */
    size_t num_batches() const override {
        return batches_prepared_ ? batched_data_.size() : BaseDataLoader<T>::num_batches();
    }

    /**
     * Check if batches are prepared
     */
    bool are_batches_prepared() const override {
        return batches_prepared_;
    }
};

// Type alias for convenience
using MNISTDataLoaderFloat = MNISTDataLoader<float>;
using MNISTDataLoaderDouble = MNISTDataLoader<double>;

} // namespace data_loading
