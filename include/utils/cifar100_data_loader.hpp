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

// Constants for CIFAR-100 dataset
namespace cifar100_constants {
    constexpr size_t IMAGE_HEIGHT = 32;
    constexpr size_t IMAGE_WIDTH = 32;
    constexpr size_t IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3; // RGB channels
    constexpr size_t NUM_CLASSES = 100;
    constexpr size_t NUM_COARSE_CLASSES = 20; // Superclasses
    constexpr size_t NUM_CHANNELS = 3;
    constexpr float NORMALIZATION_FACTOR = 255.0f;
    constexpr size_t RECORD_SIZE = 1 + 1 + IMAGE_SIZE; // 1 byte coarse label + 1 byte fine label + 3072 bytes pixel data
}

namespace data_loading {

/**
 * Enhanced CIFAR-100 data loader for binary format adapted for CNN (2D RGB images)
 * Extends ImageClassificationDataLoader for proper inheritance
 */
template<typename T = float>
class CIFAR100DataLoader : public ImageClassificationDataLoader<T> {
private:
    std::vector<std::vector<T>> data_;
    std::vector<int> fine_labels_;  // Fine-grained labels (100 classes)
    std::vector<int> coarse_labels_; // Coarse-grained labels (20 superclasses)
    
    // Pre-computed batch storage for efficiency
    std::vector<Tensor<T>> batched_data_;
    std::vector<Tensor<T>> batched_fine_labels_;
    std::vector<Tensor<T>> batched_coarse_labels_;
    bool batches_prepared_;
    bool use_coarse_labels_; // Flag to use coarse or fine labels

    // CIFAR-100 fine class names
    std::vector<std::string> fine_class_names_ = {
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
        "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
        "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
        "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
        "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
        "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
        "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
        "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
        "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
        "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
        "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
        "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
    };

    // CIFAR-100 coarse class names (superclasses)
    std::vector<std::string> coarse_class_names_ = {
        "aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
        "household_electrical_devices", "household_furniture", "insects", "large_carnivores",
        "large_man-made_outdoor_things", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores",
        "medium_mammals", "non-insect_invertebrates", "people", "reptiles", "small_mammals",
        "trees", "vehicles_1", "vehicles_2"
    };

public:
    CIFAR100DataLoader(bool use_coarse_labels = false) 
        : ImageClassificationDataLoader<T>(), 
          batches_prepared_(false), 
          use_coarse_labels_(use_coarse_labels) {
        // Reserve space to reduce allocations
        data_.reserve(50000); // CIFAR-100 train set size
        fine_labels_.reserve(50000);
        coarse_labels_.reserve(50000);
    }

    virtual ~CIFAR100DataLoader() = default;

    /**
     * Load CIFAR-100 data from binary file(s)
     * @param source Path to binary file or directory containing multiple files
     * @return true if successful, false otherwise
     */
    bool load_data(const std::string& source) override {
        // Check if source is a single file or multiple files
        std::vector<std::string> filenames;
        
        // If source ends with .bin, treat as single file
        if (source.find(".bin") != std::string::npos) {
            filenames.push_back(source);
        } else {
            // Otherwise, assume it's a directory pattern or multiple files
            // For now, we'll expect the user to call load_multiple_files instead
            std::cerr << "Error: For multiple files, use load_multiple_files() method" << std::endl;
            return false;
        }
        
        return load_multiple_files(filenames);
    }

    /**
     * Load CIFAR-100 data from multiple binary files
     * @param filenames Vector of file paths to load
     * @return true if successful, false otherwise
     */
    bool load_multiple_files(const std::vector<std::string>& filenames) {
        data_.clear();
        fine_labels_.clear();
        coarse_labels_.clear();

        for (const auto& filename : filenames) {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open file " << filename << std::endl;
                return false;
            }

            // CIFAR-100 binary format: [1-byte coarse label][1-byte fine label][3072 bytes of pixel data]
            char buffer[cifar100_constants::RECORD_SIZE];
            size_t records_loaded = 0;

            while (file.read(buffer, cifar100_constants::RECORD_SIZE)) {
                // First byte is the coarse label
                coarse_labels_.push_back(static_cast<int>(static_cast<unsigned char>(buffer[0])));
                
                // Second byte is the fine label
                fine_labels_.push_back(static_cast<int>(static_cast<unsigned char>(buffer[1])));

                // Remaining bytes are pixel data - reserve space for efficiency
                std::vector<T> image_data;
                image_data.reserve(cifar100_constants::IMAGE_SIZE);
                
                for (size_t i = 2; i < cifar100_constants::RECORD_SIZE; ++i) {
                    image_data.push_back(static_cast<T>(static_cast<unsigned char>(buffer[i]) / cifar100_constants::NORMALIZATION_FACTOR));
                }
                
                data_.push_back(std::move(image_data));
                records_loaded++;
            }

            std::cout << "Loaded " << records_loaded << " samples from " << filename << std::endl;
        }

        this->current_index_ = 0;
        std::cout << "Total loaded: " << data_.size() << " samples" << std::endl;
        std::cout << "Using " << (use_coarse_labels_ ? "coarse" : "fine") << " labels" << std::endl;
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
        
        batch_data = batched_data_[this->current_batch_index_].clone();
        if (use_coarse_labels_) {
            batch_labels = batched_coarse_labels_[this->current_batch_index_].clone();
        } else {
            batch_labels = batched_fine_labels_[this->current_batch_index_].clone();
        }
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
        const int num_classes = use_coarse_labels_ ? cifar100_constants::NUM_COARSE_CLASSES : cifar100_constants::NUM_CLASSES;

        // Create batch data tensor for CNN: (batch_size, channels=3, height=32, width=32)
        batch_data = Tensor<T>(
            static_cast<size_t>(actual_batch_size), 
            cifar100_constants::NUM_CHANNELS,
            cifar100_constants::IMAGE_HEIGHT, 
            cifar100_constants::IMAGE_WIDTH
        );

        // Create batch labels tensor (batch_size, num_classes, 1, 1) - one-hot encoded
        batch_labels = Tensor<T>(
            static_cast<size_t>(actual_batch_size), 
            static_cast<size_t>(num_classes), 
            1, 1
        );
        batch_labels.fill(static_cast<T>(0.0));

        // Parallelize batch processing for better performance
        #pragma omp parallel for if(actual_batch_size > 16)
        for (int i = 0; i < actual_batch_size; ++i) {
            const auto& image_data = data_[this->current_index_ + i];
            
            // Copy pixel data and reshape to 3x32x32
            // CIFAR-100 data is stored in channel-major order: RRR...GGG...BBB...
            for (int c = 0; c < static_cast<int>(cifar100_constants::NUM_CHANNELS); ++c) {
                for (int h = 0; h < static_cast<int>(cifar100_constants::IMAGE_HEIGHT); ++h) {
                    for (int w = 0; w < static_cast<int>(cifar100_constants::IMAGE_WIDTH); ++w) {
                        size_t pixel_idx = c * cifar100_constants::IMAGE_HEIGHT * cifar100_constants::IMAGE_WIDTH + 
                                         h * cifar100_constants::IMAGE_WIDTH + w;
                        batch_data(i, c, h, w) = image_data[pixel_idx];
                    }
                }
            }

            // Set one-hot label (fine or coarse based on flag)
            const int label = use_coarse_labels_ ? coarse_labels_[this->current_index_ + i] : fine_labels_[this->current_index_ + i];
            if (label >= 0 && label < num_classes) {
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
        std::vector<int> shuffled_fine_labels;
        std::vector<int> shuffled_coarse_labels;
        shuffled_data.reserve(data_.size());
        shuffled_fine_labels.reserve(fine_labels_.size());
        shuffled_coarse_labels.reserve(coarse_labels_.size());

        for (const auto& idx : indices) {
            shuffled_data.emplace_back(std::move(data_[idx]));
            shuffled_fine_labels.emplace_back(fine_labels_[idx]);
            shuffled_coarse_labels.emplace_back(coarse_labels_[idx]);
        }

        data_ = std::move(shuffled_data);
        fine_labels_ = std::move(shuffled_fine_labels);
        coarse_labels_ = std::move(shuffled_coarse_labels);
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
        return {cifar100_constants::NUM_CHANNELS, cifar100_constants::IMAGE_HEIGHT, cifar100_constants::IMAGE_WIDTH};
    }

    /**
     * Get number of classes
     */
    int get_num_classes() const override {
        return use_coarse_labels_ ? static_cast<int>(cifar100_constants::NUM_COARSE_CLASSES) : static_cast<int>(cifar100_constants::NUM_CLASSES);
    }

    /**
     * Get class names for CIFAR-100
     */
    std::vector<std::string> get_class_names() const override {
        return use_coarse_labels_ ? coarse_class_names_ : fine_class_names_;
    }

    /**
     * Set whether to use coarse or fine labels
     */
    void set_use_coarse_labels(bool use_coarse) {
        use_coarse_labels_ = use_coarse;
        // Invalidate pre-computed batches since label type changed
        batches_prepared_ = false;
    }

    /**
     * Check if using coarse labels
     */
    bool is_using_coarse_labels() const {
        return use_coarse_labels_;
    }

    /**
     * Get fine and coarse labels for a sample
     */
    std::pair<int, int> get_labels(size_t index) const {
        if (index >= data_.size()) {
            return {-1, -1};
        }
        return {fine_labels_[index], coarse_labels_[index]};
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
        batched_fine_labels_.clear();
        batched_coarse_labels_.clear();
        
        const size_t num_samples = data_.size();
        const size_t num_batches = (num_samples + batch_size - 1) / batch_size; // Ceiling division
        const int num_fine_classes = cifar100_constants::NUM_CLASSES;
        const int num_coarse_classes = cifar100_constants::NUM_COARSE_CLASSES;
        
        batched_data_.reserve(num_batches);
        batched_fine_labels_.reserve(num_batches);
        batched_coarse_labels_.reserve(num_batches);
        
        std::cout << "Preparing " << num_batches << " batches of size " << batch_size << "..." << std::endl;
        
        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            const size_t start_idx = batch_idx * batch_size;
            const size_t end_idx = std::min(start_idx + batch_size, num_samples);
            const int actual_batch_size = static_cast<int>(end_idx - start_idx);
            
            // Create batch tensors
            Tensor<T> batch_data(
                static_cast<size_t>(actual_batch_size), 
                cifar100_constants::NUM_CHANNELS,
                cifar100_constants::IMAGE_HEIGHT, 
                cifar100_constants::IMAGE_WIDTH
            );
            
            Tensor<T> batch_fine_labels(
                static_cast<size_t>(actual_batch_size), 
                static_cast<size_t>(num_fine_classes), 
                1, 1
            );
            batch_fine_labels.fill(static_cast<T>(0.0));
            
            Tensor<T> batch_coarse_labels(std::vector<size_t>{
                static_cast<size_t>(actual_batch_size), 
                static_cast<size_t>(num_coarse_classes), 
                1, 1
            });
            batch_coarse_labels.fill(static_cast<T>(0.0));
            
            // Fill batch data in parallel
            #pragma omp parallel for if(actual_batch_size > 16)
            for (int i = 0; i < actual_batch_size; ++i) {
                const size_t sample_idx = start_idx + i;
                const auto& image_data = data_[sample_idx];
                
                // Copy pixel data and reshape to 3x32x32
                // CIFAR-100 data is stored in channel-major order: RRR...GGG...BBB...
                for (int c = 0; c < static_cast<int>(cifar100_constants::NUM_CHANNELS); ++c) {
                    for (int h = 0; h < static_cast<int>(cifar100_constants::IMAGE_HEIGHT); ++h) {
                        for (int w = 0; w < static_cast<int>(cifar100_constants::IMAGE_WIDTH); ++w) {
                            size_t pixel_idx = c * cifar100_constants::IMAGE_HEIGHT * cifar100_constants::IMAGE_WIDTH + 
                                             h * cifar100_constants::IMAGE_WIDTH + w;
                            batch_data(i, c, h, w) = image_data[pixel_idx];
                        }
                    }
                }
                
                // Set one-hot fine label
                const int fine_label = fine_labels_[sample_idx];
                if (fine_label >= 0 && fine_label < num_fine_classes) {
                    batch_fine_labels(i, fine_label, 0, 0) = static_cast<T>(1.0);
                }
                
                // Set one-hot coarse label
                const int coarse_label = coarse_labels_[sample_idx];
                if (coarse_label >= 0 && coarse_label < num_coarse_classes) {
                    batch_coarse_labels(i, coarse_label, 0, 0) = static_cast<T>(1.0);
                }
            }
            
            batched_data_.emplace_back(std::move(batch_data));
            batched_fine_labels_.emplace_back(std::move(batch_fine_labels));
            batched_coarse_labels_.emplace_back(std::move(batch_coarse_labels));
        }
        
        this->current_batch_index_ = 0;
        batches_prepared_ = true;
        std::cout << "Batch preparation completed!" << std::endl;
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

    /**
     * Get data statistics for debugging
     */
    void print_data_stats() const {
        if (data_.empty()) {
            std::cout << "No data loaded" << std::endl;
            return;
        }

        // Calculate fine label distribution
        std::vector<int> fine_label_counts(cifar100_constants::NUM_CLASSES, 0);
        for (const auto& label : fine_labels_) {
            if (label >= 0 && label < static_cast<int>(cifar100_constants::NUM_CLASSES)) {
                fine_label_counts[label]++;
            }
        }

        // Calculate coarse label distribution
        std::vector<int> coarse_label_counts(cifar100_constants::NUM_COARSE_CLASSES, 0);
        for (const auto& label : coarse_labels_) {
            if (label >= 0 && label < static_cast<int>(cifar100_constants::NUM_COARSE_CLASSES)) {
                coarse_label_counts[label]++;
            }
        }

        std::cout << "CIFAR-100 Dataset Statistics:" << std::endl;
        std::cout << "Total samples: " << data_.size() << std::endl;
        std::cout << "Image shape: " << cifar100_constants::NUM_CHANNELS << "x" 
                  << cifar100_constants::IMAGE_HEIGHT << "x" << cifar100_constants::IMAGE_WIDTH << std::endl;
        
        std::cout << "Fine class distribution (first 10):" << std::endl;
        for (int i = 0; i < std::min(10, static_cast<int>(cifar100_constants::NUM_CLASSES)); ++i) {
            std::cout << "  " << fine_class_names_[i] << " (" << i << "): " << fine_label_counts[i] << " samples" << std::endl;
        }
        
        std::cout << "Coarse class distribution:" << std::endl;
        for (int i = 0; i < static_cast<int>(cifar100_constants::NUM_COARSE_CLASSES); ++i) {
            std::cout << "  " << coarse_class_names_[i] << " (" << i << "): " << coarse_label_counts[i] << " samples" << std::endl;
        }

        // Calculate pixel value statistics for first image
        if (!data_.empty()) {
            T min_val = *std::min_element(data_[0].begin(), data_[0].end());
            T max_val = *std::max_element(data_[0].begin(), data_[0].end());
            T sum = std::accumulate(data_[0].begin(), data_[0].end(), static_cast<T>(0.0));
            T mean = sum / data_[0].size();
            
            std::cout << "Pixel value range: [" << min_val << ", " << max_val << "]" << std::endl;
            std::cout << "First image mean pixel value: " << mean << std::endl;
        }
    }
};

// Type aliases for convenience
using CIFAR100DataLoaderFloat = CIFAR100DataLoader<float>;
using CIFAR100DataLoaderDouble = CIFAR100DataLoader<double>;

} // namespace data_loading
