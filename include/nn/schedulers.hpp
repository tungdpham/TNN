/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/config.hpp"
#include "optimizers.hpp"

namespace tnn {

using SchedulerConfig = TConfig;

/**
 * @brief Base class for learning rate schedulers.
 * @tparam T The data type (default: float).
 */
class Scheduler {
public:
  explicit Scheduler(Optimizer *optimizer) : optimizer_(optimizer), current_step_(0) {
    if (optimizer_) {
      base_lr_ = optimizer_->get_learning_rate();
    }
  }
  virtual ~Scheduler() = default;

  /**
   * @brief Update the learning rate based on the current step/epoch.
   */
  virtual void step() = 0;

  /**
   * @brief Get the current learning rate.
   */
  float get_lr() const { return optimizer_ ? optimizer_->get_learning_rate() : base_lr_; }

  /**
   * @brief Get the base learning rate.
   */
  float get_base_lr() const { return base_lr_; }

  /**
   * @brief Get the current step count.
   */
  size_t get_current_step() const { return current_step_; }

  /**
   * @brief Reset the scheduler state.
   */
  virtual void reset() {
    current_step_ = 0;
    if (optimizer_) {
      optimizer_->set_learning_rate(base_lr_);
    }
  }

  virtual std::string name() const = 0;
  virtual SchedulerConfig get_config() const = 0;
  virtual std::unique_ptr<Scheduler> clone(Optimizer *optimizer) const = 0;

protected:
  Optimizer *optimizer_;
  float base_lr_;
  size_t current_step_;

  void set_lr(float lr) {
    if (optimizer_) {
      optimizer_->set_learning_rate(lr);
    }
  }
};

/**
 * @brief No-op scheduler - learning rate remains constant.
 */
class NoOpScheduler : public Scheduler {
public:
  NoOpScheduler(Optimizer *optimizer) : Scheduler(optimizer) {}

  void step() override {
    this->current_step_++;
    // No-op: do nothing, keep learning rate constant
  }

  std::string name() const override { return "NoOpScheduler"; }

  SchedulerConfig get_config() const override {
    SchedulerConfig config;
    config.type = "no_op";
    config.name = "NoOpScheduler";
    config.set("base_lr", this->base_lr_);
    return config;
  }

  std::unique_ptr<Scheduler> clone(Optimizer *optimizer) const override {
    return std::make_unique<NoOpScheduler>(optimizer);
  }
};

/**
 * @brief Step decay scheduler - reduces LR by a factor every N steps.
 */
class StepLR : public Scheduler {
public:
  StepLR(Optimizer *optimizer, size_t step_size, float gamma = 0.1f)
      : Scheduler(optimizer), step_size_(step_size), gamma_(gamma) {}

  void step() override {
    this->current_step_++;
    if (this->current_step_ % step_size_ == 0) {
      float new_lr = this->get_lr() * gamma_;
      this->set_lr(new_lr);
    }
  }

  std::string name() const override { return "StepLR"; }

  SchedulerConfig get_config() const override {
    SchedulerConfig config;
    config.type = "step_lr";
    config.name = "StepLR";
    config.set("step_size", step_size_);
    config.set("gamma", gamma_);
    config.set("base_lr", this->base_lr_);
    return config;
  }

  std::unique_ptr<Scheduler> clone(Optimizer *optimizer) const override {
    return std::make_unique<StepLR>(optimizer, step_size_, gamma_);
  }

private:
  size_t step_size_;
  float gamma_;
};

/**
 * @brief Multi-step decay scheduler - reduces LR at specified milestones.
 */
class MultiStepLR : public Scheduler {
public:
  MultiStepLR(Optimizer *optimizer, std::vector<size_t> milestones, float gamma = 0.1f)
      : Scheduler(optimizer), milestones_(std::move(milestones)), gamma_(gamma), milestone_idx_(0) {
    std::sort(milestones_.begin(), milestones_.end());
  }

  void step() override {
    this->current_step_++;
    if (milestone_idx_ < milestones_.size() && this->current_step_ >= milestones_[milestone_idx_]) {
      float new_lr = this->get_lr() * gamma_;
      this->set_lr(new_lr);
      milestone_idx_++;
    }
  }

  void reset() override {
    Scheduler::reset();
    milestone_idx_ = 0;
  }

  std::string name() const override { return "MultiStepLR"; }

  SchedulerConfig get_config() const override {
    SchedulerConfig config;
    config.type = "multi_step_lr";
    config.name = "MultiStepLR";
    config.set("milestones", milestones_);
    config.set("gamma", gamma_);
    config.set("base_lr", this->base_lr_);
    return config;
  }

  std::unique_ptr<Scheduler> clone(Optimizer *optimizer) const override {
    return std::make_unique<MultiStepLR>(optimizer, milestones_, gamma_);
  }

private:
  std::vector<size_t> milestones_;
  float gamma_;
  size_t milestone_idx_;
};

/**
 * @brief Exponential decay scheduler - reduces LR by gamma every step.
 */
class ExponentialLR : public Scheduler {
public:
  ExponentialLR(Optimizer *optimizer, float gamma = 0.95f) : Scheduler(optimizer), gamma_(gamma) {}

  void step() override {
    this->current_step_++;
    float new_lr = this->get_lr() * gamma_;
    this->set_lr(new_lr);
  }

  std::string name() const override { return "ExponentialLR"; }

  SchedulerConfig get_config() const override {
    SchedulerConfig config;
    config.type = "exponential_lr";
    config.name = "ExponentialLR";
    config.set("gamma", gamma_);
    config.set("base_lr", this->base_lr_);
    return config;
  }

  std::unique_ptr<Scheduler> clone(Optimizer *optimizer) const override {
    return std::make_unique<ExponentialLR>(optimizer, gamma_);
  }

private:
  float gamma_;
};

/**
 * @brief Cosine annealing scheduler - follows cosine curve from base_lr to min_lr.
 */
class CosineAnnealingLR : public Scheduler {
public:
  CosineAnnealingLR(Optimizer *optimizer, size_t T_max, float eta_min = 0.0f)
      : Scheduler(optimizer), T_max_(T_max), eta_min_(eta_min) {}

  void step() override {
    this->current_step_++;
    size_t step = this->current_step_ % T_max_;
    float new_lr = eta_min_ + (this->base_lr_ - eta_min_) *
                                  (1.0f + std::cos(static_cast<float>(M_PI) * step / T_max_)) /
                                  2.0f;
    this->set_lr(new_lr);
  }

  std::string name() const override { return "CosineAnnealingLR"; }

  SchedulerConfig get_config() const override {
    SchedulerConfig config;
    config.type = "cosine_annealing_lr";
    config.name = "CosineAnnealingLR";
    config.set("T_max", T_max_);
    config.set("eta_min", eta_min_);
    config.set("base_lr", this->base_lr_);
    return config;
  }

  std::unique_ptr<Scheduler> clone(Optimizer *optimizer) const override {
    return std::make_unique<CosineAnnealingLR>(optimizer, T_max_, eta_min_);
  }

private:
  size_t T_max_;
  float eta_min_;
};

/**
 * @brief Cosine annealing with warm restarts.
 */
class CosineAnnealingWarmRestarts : public Scheduler {
public:
  CosineAnnealingWarmRestarts(Optimizer *optimizer, size_t T_0, size_t T_mult = 1,
                              float eta_min = 0.0f)
      : Scheduler(optimizer), T_0_(T_0), T_mult_(T_mult), eta_min_(eta_min), T_cur_(0), T_i_(T_0) {}

  void step() override {
    this->current_step_++;
    T_cur_++;

    if (T_cur_ >= T_i_) {
      T_cur_ = 0;
      T_i_ *= T_mult_;
    }

    float new_lr = eta_min_ + (this->base_lr_ - eta_min_) *
                                  (1.0f + std::cos(static_cast<float>(M_PI) * T_cur_ / T_i_)) /
                                  2.0f;
    this->set_lr(new_lr);
  }

  void reset() override {
    Scheduler::reset();
    T_cur_ = 0;
    T_i_ = T_0_;
  }

  std::string name() const override { return "CosineAnnealingWarmRestarts"; }

  SchedulerConfig get_config() const override {
    SchedulerConfig config;
    config.type = "cosine_annealing_warm_restarts";
    config.name = "CosineAnnealingWarmRestarts";
    config.set("T_0", T_0_);
    config.set("T_mult", T_mult_);
    config.set("eta_min", eta_min_);
    config.set("base_lr", this->base_lr_);
    return config;
  }

  std::unique_ptr<Scheduler> clone(Optimizer *optimizer) const override {
    return std::make_unique<CosineAnnealingWarmRestarts>(optimizer, T_0_, T_mult_, eta_min_);
  }

private:
  size_t T_0_;
  size_t T_mult_;
  float eta_min_;
  size_t T_cur_;
  size_t T_i_;
};

/**
 * @brief Linear warmup scheduler - linearly increases LR from start_lr to base_lr.
 */
class LinearWarmup : public Scheduler {
public:
  LinearWarmup(Optimizer *optimizer, size_t warmup_steps, float start_lr = 0.0f)
      : Scheduler(optimizer), warmup_steps_(warmup_steps), start_lr_(start_lr) {
    // Start at start_lr
    this->set_lr(start_lr_);
  }

  void step() override {
    this->current_step_++;
    if (this->current_step_ <= warmup_steps_) {
      float progress = static_cast<float>(this->current_step_) / warmup_steps_;
      float new_lr = start_lr_ + progress * (this->base_lr_ - start_lr_);
      this->set_lr(new_lr);
    }
  }

  bool is_warmup_complete() const { return this->current_step_ >= warmup_steps_; }

  std::string name() const override { return "LinearWarmup"; }

  SchedulerConfig get_config() const override {
    SchedulerConfig config;
    config.type = "linear_warmup";
    config.name = "LinearWarmup";
    config.set("warmup_steps", warmup_steps_);
    config.set("start_lr", start_lr_);
    config.set("base_lr", this->base_lr_);
    return config;
  }

  std::unique_ptr<Scheduler> clone(Optimizer *optimizer) const override {
    return std::make_unique<LinearWarmup>(optimizer, warmup_steps_, start_lr_);
  }

private:
  size_t warmup_steps_;
  float start_lr_;
};

/**
 * @brief Linear warmup followed by cosine annealing decay.
 */
class WarmupCosineAnnealing : public Scheduler {
public:
  WarmupCosineAnnealing(Optimizer *optimizer, size_t warmup_steps, size_t total_steps,
                        float start_lr = 0.0f, float eta_min = 0.0f)
      : Scheduler(optimizer),
        warmup_steps_(warmup_steps),
        total_steps_(total_steps),
        start_lr_(start_lr),
        eta_min_(eta_min) {
    this->set_lr(start_lr_);
  }

  void step() override {
    this->current_step_++;

    if (this->current_step_ <= warmup_steps_) {
      // Warmup phase
      float progress = static_cast<float>(this->current_step_) / warmup_steps_;
      float new_lr = start_lr_ + progress * (this->base_lr_ - start_lr_);
      this->set_lr(new_lr);
    } else {
      // Cosine annealing phase
      size_t decay_steps = total_steps_ - warmup_steps_;
      size_t current_decay_step = this->current_step_ - warmup_steps_;
      float progress = static_cast<float>(current_decay_step) / decay_steps;
      progress = std::min(progress, 1.0f);
      float new_lr = eta_min_ + (this->base_lr_ - eta_min_) *
                                    (1.0f + std::cos(static_cast<float>(M_PI) * progress)) / 2.0f;
      this->set_lr(new_lr);
    }
  }

  std::string name() const override { return "WarmupCosineAnnealing"; }

  SchedulerConfig get_config() const override {
    SchedulerConfig config;
    config.type = "warmup_cosine_annealing";
    config.name = "WarmupCosineAnnealing";
    config.set("warmup_steps", warmup_steps_);
    config.set("total_steps", total_steps_);
    config.set("start_lr", start_lr_);
    config.set("eta_min", eta_min_);
    config.set("base_lr", this->base_lr_);
    return config;
  }

  std::unique_ptr<Scheduler> clone(Optimizer *optimizer) const override {
    return std::make_unique<WarmupCosineAnnealing>(optimizer, warmup_steps_, total_steps_,
                                                   start_lr_, eta_min_);
  }

private:
  size_t warmup_steps_;
  size_t total_steps_;
  float start_lr_;
  float eta_min_;
};

/**
 * @brief Reduce LR on plateau - reduces LR when a metric has stopped improving.
 */
class ReduceLROnPlateau : public Scheduler {
public:
  enum class Mode { MIN, MAX };

  ReduceLROnPlateau(Optimizer *optimizer, Mode mode = Mode::MIN, float factor = 0.1f,
                    size_t patience = 10, float threshold = 1e-4f, float min_lr = 0.0f)
      : Scheduler(optimizer),
        mode_(mode),
        factor_(factor),
        patience_(patience),
        threshold_(threshold),
        min_lr_(min_lr),
        best_value_(mode == Mode::MIN ? 1e10f : -1e10f),
        bad_epochs_(0) {}

  void step() override {
    // This scheduler doesn't do anything on regular step()
    // Use step(metric) instead
    this->current_step_++;
  }

  /**
   * @brief Step with a metric value to monitor.
   * @param metric The current value of the metric to monitor.
   */
  void step(float metric) {
    this->current_step_++;

    bool is_better = false;
    if (mode_ == Mode::MIN) {
      is_better = metric < (best_value_ - threshold_);
    } else {
      is_better = metric > (best_value_ + threshold_);
    }

    if (is_better) {
      best_value_ = metric;
      bad_epochs_ = 0;
    } else {
      bad_epochs_++;
      if (bad_epochs_ >= patience_) {
        float new_lr = std::max(this->get_lr() * factor_, min_lr_);
        this->set_lr(new_lr);
        bad_epochs_ = 0;
      }
    }
  }

  void reset() override {
    Scheduler::reset();
    best_value_ = (mode_ == Mode::MIN) ? 1e10f : -1e10f;
    bad_epochs_ = 0;
  }

  std::string name() const override { return "ReduceLROnPlateau"; }

  SchedulerConfig get_config() const override {
    SchedulerConfig config;
    config.type = "reduce_lr_on_plateau";
    config.name = "ReduceLROnPlateau";
    config.set("mode", (mode_ == Mode::MIN) ? std::string("min") : std::string("max"));
    config.set("factor", factor_);
    config.set("patience", patience_);
    config.set("threshold", threshold_);
    config.set("min_lr", min_lr_);
    config.set("base_lr", this->base_lr_);
    return config;
  }

  std::unique_ptr<Scheduler> clone(Optimizer *optimizer) const override {
    return std::make_unique<ReduceLROnPlateau>(optimizer, mode_, factor_, patience_, threshold_,
                                               min_lr_);
  }

private:
  Mode mode_;
  float factor_;
  size_t patience_;
  float threshold_;
  float min_lr_;
  float best_value_;
  size_t bad_epochs_;
};

/**
 * @brief Polynomial decay scheduler.
 */
class PolynomialLR : public Scheduler {
public:
  PolynomialLR(Optimizer *optimizer, size_t total_steps, float power = 1.0f, float end_lr = 0.0f)
      : Scheduler(optimizer), total_steps_(total_steps), power_(power), end_lr_(end_lr) {}

  void step() override {
    this->current_step_++;
    float progress = static_cast<float>(this->current_step_) / total_steps_;
    progress = std::min(progress, 1.0f);
    float new_lr = (this->base_lr_ - end_lr_) * std::pow(1.0f - progress, power_) + end_lr_;
    this->set_lr(new_lr);
  }

  std::string name() const override { return "PolynomialLR"; }

  SchedulerConfig get_config() const override {
    SchedulerConfig config;
    config.type = "polynomial_lr";
    config.name = "PolynomialLR";
    config.set("total_steps", total_steps_);
    config.set("power", power_);
    config.set("end_lr", end_lr_);
    config.set("base_lr", this->base_lr_);
    return config;
  }

  std::unique_ptr<Scheduler> clone(Optimizer *optimizer) const override {
    return std::make_unique<PolynomialLR>(optimizer, total_steps_, power_, end_lr_);
  }

private:
  size_t total_steps_;
  float power_;
  float end_lr_;
};

/**
 * @brief One Cycle scheduler - follows the 1cycle policy.
 */
class OneCycleLR : public Scheduler {
public:
  OneCycleLR(Optimizer *optimizer, float max_lr, size_t total_steps, float pct_start = 0.3f,
             float div_factor = 25.0f, float final_div_factor = 1e4f)
      : Scheduler(optimizer),
        max_lr_(max_lr),
        total_steps_(total_steps),
        pct_start_(pct_start),
        div_factor_(div_factor),
        final_div_factor_(final_div_factor) {
    initial_lr_ = max_lr_ / div_factor_;
    min_lr_ = initial_lr_ / final_div_factor_;
    step_up_ = static_cast<size_t>(total_steps_ * pct_start_);
    step_down_ = total_steps_ - step_up_;
    this->set_lr(initial_lr_);
  }

  void step() override {
    this->current_step_++;

    float new_lr;
    if (this->current_step_ <= step_up_) {
      // Increase phase
      float progress = static_cast<float>(this->current_step_) / step_up_;
      new_lr = initial_lr_ + progress * (max_lr_ - initial_lr_);
    } else {
      // Decrease phase (cosine annealing)
      size_t current_down = this->current_step_ - step_up_;
      float progress = static_cast<float>(current_down) / step_down_;
      new_lr = min_lr_ +
               (max_lr_ - min_lr_) * (1.0f + std::cos(static_cast<float>(M_PI) * progress)) / 2.0f;
    }
    this->set_lr(new_lr);
  }

  std::string name() const override { return "OneCycleLR"; }

  SchedulerConfig get_config() const override {
    SchedulerConfig config;
    config.type = "one_cycle_lr";
    config.name = "OneCycleLR";
    config.set("max_lr", max_lr_);
    config.set("total_steps", total_steps_);
    config.set("pct_start", pct_start_);
    config.set("div_factor", div_factor_);
    config.set("final_div_factor", final_div_factor_);
    return config;
  }

  std::unique_ptr<Scheduler> clone(Optimizer *optimizer) const override {
    return std::make_unique<OneCycleLR>(optimizer, max_lr_, total_steps_, pct_start_, div_factor_,
                                        final_div_factor_);
  }

private:
  float max_lr_;
  size_t total_steps_;
  float pct_start_;
  float div_factor_;
  float final_div_factor_;
  float initial_lr_;
  float min_lr_;
  size_t step_up_;
  size_t step_down_;
};

/**
 * @brief Factory class for creating schedulers.
 */
class SchedulerFactory {
public:
  /**
   * @brief Static method to create a no-op scheduler (learning rate unchanged).
   */
  static std::unique_ptr<Scheduler> create_no_op(Optimizer *optimizer) {
    return std::make_unique<NoOpScheduler>(optimizer);
  }

  /**
   * @brief Static method to create a StepLR scheduler.
   */
  static std::unique_ptr<Scheduler> create_step_lr(Optimizer *optimizer, size_t step_size,
                                                   float gamma = 0.1f) {
    return std::make_unique<StepLR>(optimizer, step_size, gamma);
  }

  /**
   * @brief Static method to create a CosineAnnealingLR scheduler.
   */
  static std::unique_ptr<Scheduler> create_cosine_annealing(Optimizer *optimizer, size_t T_max,
                                                            float eta_min = 0.0f) {
    return std::make_unique<CosineAnnealingLR>(optimizer, T_max, eta_min);
  }

  /**
   * @brief Static method to create a linear warmup followed by cosine annealing scheduler.
   */
  static std::unique_ptr<Scheduler> create_warmup_cosine(Optimizer *optimizer, size_t warmup_steps,
                                                         size_t total_steps, float start_lr = 0.0f,
                                                         float eta_min = 0.0f) {
    return std::make_unique<WarmupCosineAnnealing>(optimizer, warmup_steps, total_steps, start_lr,
                                                   eta_min);
  }

  static std::unique_ptr<Scheduler> create(
      const std::string &name, Optimizer *optimizer,
      const std::unordered_map<std::string, float> &params = {}) {
    if (name == "step_lr") {
      size_t step_size =
          static_cast<size_t>(params.count("step_size") ? params.at("step_size") : 10);
      float gamma = params.count("gamma") ? params.at("gamma") : 0.1f;
      return std::make_unique<StepLR>(optimizer, step_size, gamma);
    }
    if (name == "exponential_lr") {
      float gamma = params.count("gamma") ? params.at("gamma") : 0.95f;
      return std::make_unique<ExponentialLR>(optimizer, gamma);
    }
    if (name == "cosine_annealing_lr") {
      size_t T_max = static_cast<size_t>(params.count("T_max") ? params.at("T_max") : 100);
      float eta_min = params.count("eta_min") ? params.at("eta_min") : 0.0f;
      return std::make_unique<CosineAnnealingLR>(optimizer, T_max, eta_min);
    }
    if (name == "polynomial_lr") {
      size_t total_steps =
          static_cast<size_t>(params.count("total_steps") ? params.at("total_steps") : 100);
      float power = params.count("power") ? params.at("power") : 1.0f;
      float end_lr = params.count("end_lr") ? params.at("end_lr") : 0.0f;
      return std::make_unique<PolynomialLR>(optimizer, total_steps, power, end_lr);
    }
    throw std::invalid_argument("Unknown scheduler type: " + name);
  }

  static std::unique_ptr<Scheduler> create_from_config(const SchedulerConfig &config,
                                                       Optimizer *optimizer) {
    if (config.type == "step_lr") {
      size_t step_size = config.get<size_t>("step_size", 10);
      float gamma = config.get<float>("gamma", 0.1f);
      return std::make_unique<StepLR>(optimizer, step_size, gamma);
    }
    if (config.type == "multi_step_lr") {
      auto milestones = config.get<std::vector<size_t>>("milestones", {});
      float gamma = config.get<float>("gamma", 0.1f);
      return std::make_unique<MultiStepLR>(optimizer, milestones, gamma);
    }
    if (config.type == "exponential_lr") {
      float gamma = config.get<float>("gamma", 0.95f);
      return std::make_unique<ExponentialLR>(optimizer, gamma);
    }
    if (config.type == "cosine_annealing_lr") {
      size_t T_max = config.get<size_t>("T_max", 100);
      float eta_min = config.get<float>("eta_min", 0.0f);
      return std::make_unique<CosineAnnealingLR>(optimizer, T_max, eta_min);
    }
    if (config.type == "cosine_annealing_warm_restarts") {
      size_t T_0 = config.get<size_t>("T_0", 10);
      size_t T_mult = config.get<size_t>("T_mult", 1);
      float eta_min = config.get<float>("eta_min", 0.0f);
      return std::make_unique<CosineAnnealingWarmRestarts>(optimizer, T_0, T_mult, eta_min);
    }
    if (config.type == "linear_warmup") {
      size_t warmup_steps = config.get<size_t>("warmup_steps", 100);
      float start_lr = config.get<float>("start_lr", 0.0f);
      return std::make_unique<LinearWarmup>(optimizer, warmup_steps, start_lr);
    }
    if (config.type == "warmup_cosine_annealing") {
      size_t warmup_steps = config.get<size_t>("warmup_steps", 100);
      size_t total_steps = config.get<size_t>("total_steps", 1000);
      float start_lr = config.get<float>("start_lr", 0.0f);
      float eta_min = config.get<float>("eta_min", 0.0f);
      return std::make_unique<WarmupCosineAnnealing>(optimizer, warmup_steps, total_steps, start_lr,
                                                     eta_min);
    }
    if (config.type == "reduce_lr_on_plateau") {
      std::string mode_str = config.get<std::string>("mode", "min");
      auto mode = (mode_str == "max") ? ReduceLROnPlateau::Mode::MAX : ReduceLROnPlateau::Mode::MIN;
      float factor = config.get<float>("factor", 0.1f);
      size_t patience = config.get<size_t>("patience", 10);
      float threshold = config.get<float>("threshold", 1e-4f);
      float min_lr = config.get<float>("min_lr", 0.0f);
      return std::make_unique<ReduceLROnPlateau>(optimizer, mode, factor, patience, threshold,
                                                 min_lr);
    }
    if (config.type == "polynomial_lr") {
      size_t total_steps = config.get<size_t>("total_steps", 100);
      float power = config.get<float>("power", 1.0f);
      float end_lr = config.get<float>("end_lr", 0.0f);
      return std::make_unique<PolynomialLR>(optimizer, total_steps, power, end_lr);
    }
    if (config.type == "one_cycle_lr") {
      float max_lr = config.get<float>("max_lr", 0.1f);
      size_t total_steps = config.get<size_t>("total_steps", 100);
      float pct_start = config.get<float>("pct_start", 0.3f);
      float div_factor = config.get<float>("div_factor", 25.0f);
      float final_div_factor = config.get<float>("final_div_factor", 1e4f);
      return std::make_unique<OneCycleLR>(optimizer, max_lr, total_steps, pct_start, div_factor,
                                          final_div_factor);
    }
    throw std::invalid_argument("Unknown scheduler type: " + config.type);
  }
};

}  // namespace tnn
