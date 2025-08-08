#pragma once

#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "../tensor/tensor.hpp"

namespace layers {

// Forward declaration for WiFiDataLoader (needed for DistanceLoss)
class WiFiDataLoader;

// Base class for all loss functions
template <typename T = float>
class Loss {
public:
    virtual ~Loss() = default;
    
    // Pure virtual functions that must be implemented by derived classes
    virtual T compute_loss(const Tensor<T>& predictions, const Tensor<T>& targets) = 0;
    virtual Tensor<T> compute_gradient(const Tensor<T>& predictions, const Tensor<T>& targets) = 0;
    
    // Optional: Get the name of the loss function
    virtual std::string name() const = 0;
    
    // Optional: Get number of parameters (for losses that have learnable parameters)
    virtual size_t num_parameters() const { return 0; }
    
    // Optional: Reset any internal state (useful for stateful losses)
    virtual void reset() {}
};

// Cross-entropy loss for multi-class classification
template <typename T = float>
class CrossEntropyLoss : public Loss<T> {
public:
    explicit CrossEntropyLoss(T epsilon = static_cast<T>(1e-15)) : epsilon_(epsilon) {}
    
    T compute_loss(const Tensor<T>& predictions, const Tensor<T>& targets) override {
        const size_t batch_size = predictions.shape()[0];
        const size_t num_classes = predictions.shape()[1];
        
        double total_loss = 0.0;
        
#ifdef _OPENMP
#pragma omp parallel for reduction(+:total_loss)
#endif
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < num_classes; ++j) {
                if (targets(i, j, 0, 0) > static_cast<T>(0.5)) {
                    const T pred = std::clamp(predictions(i, j, 0, 0), epsilon_, 
                                            static_cast<T>(1.0) - epsilon_);
                    total_loss -= std::log(pred);
                    break; // Found the true class
                }
            }
        }
        
        return static_cast<T>(total_loss / batch_size);
    }
    
    Tensor<T> compute_gradient(const Tensor<T>& predictions, const Tensor<T>& targets) override {
        Tensor<T> gradient = predictions;
        const size_t batch_size = predictions.shape()[0];
        const size_t num_classes = predictions.shape()[1];
        const T inv_batch_size = static_cast<T>(1.0) / static_cast<T>(batch_size);
        
#ifdef _OPENMP
#pragma omp parallel for if(batch_size > 32)
#endif
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < num_classes; ++j) {
                gradient(i, j, 0, 0) = (predictions(i, j, 0, 0) - targets(i, j, 0, 0)) * inv_batch_size;
            }
        }
        
        return gradient;
    }
    
    std::string name() const override {
        return "CrossEntropyLoss";
    }
    
private:
    T epsilon_;
};

// Mean Squared Error loss for regression tasks
template <typename T = float>
class MSELoss : public Loss<T> {
public:
    MSELoss() = default;
    
    T compute_loss(const Tensor<T>& predictions, const Tensor<T>& targets) override {
        const size_t batch_size = predictions.shape()[0];
        const size_t output_size = predictions.shape()[1];
        
        double total_loss = 0.0;
        
#ifdef _OPENMP
#pragma omp parallel for reduction(+:total_loss) if(batch_size > 32)
#endif
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                const T diff = predictions(i, j, 0, 0) - targets(i, j, 0, 0);
                total_loss += static_cast<double>(diff * diff);
            }
        }
        
        return static_cast<T>(total_loss / (batch_size * output_size));
    }
    
    Tensor<T> compute_gradient(const Tensor<T>& predictions, const Tensor<T>& targets) override {
        Tensor<T> gradient = predictions;
        const size_t batch_size = predictions.shape()[0];
        const size_t output_size = predictions.shape()[1];
        const T scale = static_cast<T>(2.0) / static_cast<T>(batch_size * output_size);
        
#ifdef _OPENMP
#pragma omp parallel for if(batch_size > 32)
#endif
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                gradient(i, j, 0, 0) = (predictions(i, j, 0, 0) - targets(i, j, 0, 0)) * scale;
            }
        }
        
        return gradient;
    }
    
    std::string name() const override {
        return "MSELoss";
    }
};

// Mean Absolute Error loss for regression tasks
template <typename T = float>
class MAELoss : public Loss<T> {
public:
    MAELoss() = default;
    
    T compute_loss(const Tensor<T>& predictions, const Tensor<T>& targets) override {
        const size_t batch_size = predictions.shape()[0];
        const size_t output_size = predictions.shape()[1];
        
        double total_loss = 0.0;
        
#ifdef _OPENMP
#pragma omp parallel for reduction(+:total_loss) if(batch_size > 32)
#endif
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                total_loss += std::abs(predictions(i, j, 0, 0) - targets(i, j, 0, 0));
            }
        }
        
        return static_cast<T>(total_loss / (batch_size * output_size));
    }
    
    Tensor<T> compute_gradient(const Tensor<T>& predictions, const Tensor<T>& targets) override {
        Tensor<T> gradient = predictions;
        const size_t batch_size = predictions.shape()[0];
        const size_t output_size = predictions.shape()[1];
        const T scale = static_cast<T>(1.0) / static_cast<T>(batch_size * output_size);
        
#ifdef _OPENMP
#pragma omp parallel for if(batch_size > 32)
#endif
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                const T diff = predictions(i, j, 0, 0) - targets(i, j, 0, 0);
                gradient(i, j, 0, 0) = (diff > static_cast<T>(0) ? scale : -scale);
            }
        }
        
        return gradient;
    }
    
    std::string name() const override {
        return "MAELoss";
    }
};

// Huber loss for robust regression (less sensitive to outliers than MSE)
template <typename T = float>
class HuberLoss : public Loss<T> {
public:
    explicit HuberLoss(T delta = static_cast<T>(1.0)) : delta_(delta) {}
    
    T compute_loss(const Tensor<T>& predictions, const Tensor<T>& targets) override {
        const size_t batch_size = predictions.shape()[0];
        const size_t output_size = predictions.shape()[1];
        
        double total_loss = 0.0;
        
#ifdef _OPENMP
#pragma omp parallel for reduction(+:total_loss) if(batch_size > 32)
#endif
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                const T diff = std::abs(predictions(i, j, 0, 0) - targets(i, j, 0, 0));
                if (diff <= delta_) {
                    total_loss += static_cast<double>(0.5 * diff * diff);
                } else {
                    total_loss += static_cast<double>(delta_ * diff - 0.5 * delta_ * delta_);
                }
            }
        }
        
        return static_cast<T>(total_loss / (batch_size * output_size));
    }
    
    Tensor<T> compute_gradient(const Tensor<T>& predictions, const Tensor<T>& targets) override {
        Tensor<T> gradient = predictions;
        const size_t batch_size = predictions.shape()[0];
        const size_t output_size = predictions.shape()[1];
        const T scale = static_cast<T>(1.0) / static_cast<T>(batch_size * output_size);
        
#ifdef _OPENMP
#pragma omp parallel for if(batch_size > 32)
#endif
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                const T diff = predictions(i, j, 0, 0) - targets(i, j, 0, 0);
                const T abs_diff = std::abs(diff);
                
                if (abs_diff <= delta_) {
                    gradient(i, j, 0, 0) = diff * scale;
                } else {
                    gradient(i, j, 0, 0) = (diff > static_cast<T>(0) ? delta_ : -delta_) * scale;
                }
            }
        }
        
        return gradient;
    }
    
    std::string name() const override {
        return "HuberLoss";
    }
    
    void set_delta(T delta) { delta_ = delta; }
    T get_delta() const { return delta_; }
    
private:
    T delta_;
};

// Factory class for creating loss functions
template <typename T = float>
class LossFactory {
public:
    static std::unique_ptr<Loss<T>> create(const std::string& loss_type) {
        if (loss_type == "crossentropy" || loss_type == "ce") {
            return std::make_unique<CrossEntropyLoss<T>>();
        }
        if (loss_type == "mse" || loss_type == "mean_squared_error") {
            return std::make_unique<MSELoss<T>>();
        }
        if (loss_type == "mae" || loss_type == "mean_absolute_error") {
            return std::make_unique<MAELoss<T>>();
        }
        if (loss_type == "huber") {
            return std::make_unique<HuberLoss<T>>();
        }
        throw std::invalid_argument("Unknown loss type: " + loss_type);
    }
    
    static std::unique_ptr<Loss<T>> create_crossentropy(T epsilon = static_cast<T>(1e-15)) {
        return std::make_unique<CrossEntropyLoss<T>>(epsilon);
    }
    
    static std::unique_ptr<Loss<T>> create_mse() {
        return std::make_unique<MSELoss<T>>();
    }
    
    static std::unique_ptr<Loss<T>> create_mae() {
        return std::make_unique<MAELoss<T>>();
    }
    
    static std::unique_ptr<Loss<T>> create_huber(T delta = static_cast<T>(1.0)) {
        return std::make_unique<HuberLoss<T>>(delta);
    }
};

} // namespace layers
