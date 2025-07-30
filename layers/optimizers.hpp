#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <cmath>
#include "../tensor/tensor.hpp"

namespace layers {

// Base class for all optimizers
template <typename T = float>
class Optimizer {
public:
    explicit Optimizer(float learning_rate) : learning_rate_(learning_rate) {}
    virtual ~Optimizer() = default;

    virtual void update(std::vector<Tensor<T>*>& params, const std::vector<Tensor<T>*>& grads) = 0;

    void set_learning_rate(float lr) { learning_rate_ = lr; }
    float get_learning_rate() const { return learning_rate_; }

protected:
    float learning_rate_;
};

// Stochastic Gradient Descent (SGD) optimizer
template <typename T = float>
class SGD : public Optimizer<T> {
public:
    SGD(float learning_rate = 0.01f, float momentum = 0.0f)
        : Optimizer<T>(learning_rate), momentum_(momentum), initialized_(false) {}

    void update(std::vector<Tensor<T>*>& params, const std::vector<Tensor<T>*>& grads) override {
        if (momentum_ > 0.0f && !initialized_) {
            velocities_.resize(params.size());
            for (size_t i = 0; i < params.size(); ++i) {
                velocities_[i] = Tensor<T>(params[i]->shape());
                velocities_[i].fill(0.0f);
            }
            initialized_ = true;
        }

        for (size_t i = 0; i < params.size(); ++i) {
            if (momentum_ > 0.0f) {
                // v = momentum * v - learning_rate * grad
                velocities_[i] *= momentum_;
                Tensor<T> scaled_grad = (*grads[i]) * this->learning_rate_;
                velocities_[i] -= scaled_grad;
                // param += v
                (*params[i]) += velocities_[i];
            } else {
                // param -= learning_rate * grad
                Tensor<T> scaled_grad = (*grads[i]) * this->learning_rate_;
                (*params[i]) -= scaled_grad;
            }
        }
    }

private:
    float momentum_;
    bool initialized_;
    std::vector<Tensor<T>> velocities_;
};

// Adam optimizer
template <typename T = float>
class Adam : public Optimizer<T> {
public:
    Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : Optimizer<T>(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0), initialized_(false) {}

    void update(std::vector<Tensor<T>*>& params, const std::vector<Tensor<T>*>& grads) override {
        // for(auto& params : params) {
        //     printf("Parameter shape: %s\n", params->shape_str().c_str());
        // }
        // for (auto& grad : grads) {
        //     printf("Gradient shape: %s\n", grad->shape_str().c_str());
        // }
        if (!initialized_) {
            m_.resize(params.size());
            v_.resize(params.size());
            for (size_t i = 0; i < params.size(); ++i) {
                m_[i] = Tensor<T>(params[i]->shape());
                m_[i].fill(0.0f);
                v_[i] = Tensor<T>(params[i]->shape());
                v_[i].fill(0.0f);
            }
            initialized_ = true;
        }

        t_++;

        for (size_t i = 0; i < params.size(); ++i) {
            // m = beta1 * m + (1 - beta1) * grad
            m_[i] *= beta1_;
            Tensor<T> grad_term = (*grads[i]) * (1.0f - beta1_);
            m_[i] += grad_term;

            // v = beta2 * v + (1 - beta2) * grad^2
            Tensor<T> grad_sq = (*grads[i]);
            grad_sq *= (*grads[i]); // Element-wise square
            v_[i] *= beta2_;
            grad_sq *= (1.0f - beta2_);
            v_[i] += grad_sq;

            // Bias correction
            Tensor<T> m_hat = m_[i] / (1.0f - std::pow(beta1_, t_));
            Tensor<T> v_hat = v_[i] / (1.0f - std::pow(beta2_, t_));

            // Update parameters
            // param -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
            T* param_data = params[i]->data();
            const T* m_hat_data = m_hat.data();
            const T* v_hat_data = v_hat.data();

            for (size_t j = 0; j < params[i]->size(); ++j) {
                param_data[j] -= this->learning_rate_ * m_hat_data[j] / (std::sqrt(v_hat_data[j]) + epsilon_);
            }
        }
    }

private:
    float beta1_;
    float beta2_;
    float epsilon_;
    unsigned long t_;
    bool initialized_;
    std::vector<Tensor<T>> m_; // 1st moment vector
    std::vector<Tensor<T>> v_; // 2nd moment vector
};

// Optimizer Factory
template <typename T = float>
class OptimizerFactory {
public:
    static std::unique_ptr<Optimizer<T>> create(const std::string& name, float learning_rate, float momentum = 0.9f) {
        if (name == "sgd") {
            return std::make_unique<SGD<T>>(learning_rate, momentum);
        }
        if (name == "adam") {
            return std::make_unique<Adam<T>>(learning_rate);
        }
        throw std::invalid_argument("Unknown optimizer type: " + name);
    }
};

} // namespace layers
