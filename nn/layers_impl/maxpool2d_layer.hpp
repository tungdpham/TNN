#pragma once

#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../tensor/tensor.hpp"
#include "../parallel_for.hpp"

namespace tnn {

// Forward declarations
template <typename T> class Layer;
template <typename T> class StatelessLayer;
struct LayerConfig;

template <typename T = float> class MaxPool2DLayer : public StatelessLayer<T> {
private:
  size_t pool_h_;
  size_t pool_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;

  // Use more efficient storage for mask indices
  std::unordered_map<int, std::vector<size_t>> micro_batch_mask_indices_;
  std::unordered_map<int, Tensor<T>> micro_batch_inputs_;

  // Pre-computed stride values for faster access
  mutable size_t input_stride_n_, input_stride_c_, input_stride_h_,
      input_stride_w_;
  mutable size_t output_stride_n_, output_stride_c_, output_stride_h_,
      output_stride_w_;

public:
  MaxPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h = 0,
                 size_t stride_w = 0, size_t pad_h = 0, size_t pad_w = 0,
                 const std::string &name = "maxpool2d")
      : StatelessLayer<T>(name), pool_h_(pool_h), pool_w_(pool_w),
        stride_h_(stride_h == 0 ? pool_h : stride_h),
        stride_w_(stride_w == 0 ? pool_w : stride_w), pad_h_(pad_h),
        pad_w_(pad_w), input_stride_n_(0), input_stride_c_(0),
        input_stride_h_(0), input_stride_w_(0), output_stride_n_(0),
        output_stride_c_(0), output_stride_h_(0), output_stride_w_(0) {

    // Validate parameters
    if (pool_h_ == 0 || pool_w_ == 0) {
      throw std::invalid_argument("Pool dimensions must be positive");
    }
    if (stride_h_ == 0 || stride_w_ == 0) {
      throw std::invalid_argument("Stride dimensions must be positive");
    }
  }

  // Add method to clear cached data for memory management
  void clear_cache(int micro_batch_id = -1) {
    if (micro_batch_id < 0) {
      // Clear all cached data
      micro_batch_inputs_.clear();
      micro_batch_mask_indices_.clear();
    } else {
      // Clear specific micro-batch data
      micro_batch_inputs_.erase(micro_batch_id);
      micro_batch_mask_indices_.erase(micro_batch_id);
    }
  }

  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) override {
    // Store input for backward pass
    micro_batch_inputs_[micro_batch_id] = input;

    const size_t batch_size = input.batch_size();
    const size_t channels = input.channels();
    const size_t input_h = input.height();
    const size_t input_w = input.width();

    // Calculate output dimensions
    const size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
    const size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

    // Create output tensor
    Tensor<T> output(
        std::vector<size_t>{batch_size, channels, output_h, output_w});

    // Pre-compute strides for efficient memory access
    input_stride_n_ = input.stride(0);
    input_stride_c_ = input.stride(1);
    input_stride_h_ = input.stride(2);
    input_stride_w_ = input.stride(3);

    output_stride_n_ = output.stride(0);
    output_stride_c_ = output.stride(1);
    output_stride_h_ = output.stride(2);
    output_stride_w_ = output.stride(3);

    // Store mask indices more efficiently
    const size_t total_outputs = batch_size * channels * output_h * output_w;
    std::vector<size_t> mask_indices(total_outputs);

    const T *input_data = input.data();
    T *output_data = output.data();

    // Unified pooling implementation - handles all cases cleanly
#if defined(USE_TBB)
    tnn::parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
      const T *input_channel =
          input_data + n * input_stride_n_ + c * input_stride_c_;
      T *output_channel =
          output_data + n * output_stride_n_ + c * output_stride_c_;

      for (size_t out_h = 0; out_h < output_h; ++out_h) {
        for (size_t out_w = 0; out_w < output_w; ++out_w) {
          T max_val = -std::numeric_limits<T>::infinity();
          size_t max_idx = 0;

          // Pool over the kernel region
          for (size_t ph = 0; ph < pool_h_; ++ph) {
            for (size_t pw = 0; pw < pool_w_; ++pw) {
              const int h_idx = static_cast<int>(out_h * stride_h_ + ph) -
                                static_cast<int>(pad_h_);
              const int w_idx = static_cast<int>(out_w * stride_w_ + pw) -
                                static_cast<int>(pad_w_);

              // Check bounds (handles padding naturally)
              if (h_idx >= 0 && h_idx < static_cast<int>(input_h) &&
                  w_idx >= 0 && w_idx < static_cast<int>(input_w)) {
                T val = input_channel[h_idx * input_stride_h_ +
                                      w_idx * input_stride_w_];
                if (val > max_val) {
                  max_val = val;
                  max_idx = h_idx * input_w + w_idx;
                }
              }
            }
          }

          output_channel[out_h * output_stride_h_ + out_w * output_stride_w_] =
              max_val;
          const size_t output_idx =
              ((n * channels + c) * output_h + out_h) * output_w + out_w;
          mask_indices[output_idx] = max_idx;
        }
      }
    });
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        const T *input_channel =
            input_data + n * input_stride_n_ + c * input_stride_c_;
        T *output_channel =
            output_data + n * output_stride_n_ + c * output_stride_c_;

        for (size_t out_h = 0; out_h < output_h; ++out_h) {
          for (size_t out_w = 0; out_w < output_w; ++out_w) {
            T max_val = -std::numeric_limits<T>::infinity();
            size_t max_idx = 0;

            // Pool over the kernel region
            for (size_t ph = 0; ph < pool_h_; ++ph) {
              for (size_t pw = 0; pw < pool_w_; ++pw) {
                const int h_idx = static_cast<int>(out_h * stride_h_ + ph) -
                                  static_cast<int>(pad_h_);
                const int w_idx = static_cast<int>(out_w * stride_w_ + pw) -
                                  static_cast<int>(pad_w_);

                // Check bounds (handles padding naturally)
                if (h_idx >= 0 && h_idx < static_cast<int>(input_h) &&
                    w_idx >= 0 && w_idx < static_cast<int>(input_w)) {
                  T val = input_channel[h_idx * input_stride_h_ +
                                        w_idx * input_stride_w_];
                  if (val > max_val) {
                    max_val = val;
                    max_idx = h_idx * input_w + w_idx;
                  }
                }
              }
            }

            output_channel[out_h * output_stride_h_ +
                           out_w * output_stride_w_] = max_val;
            const size_t output_idx =
                ((n * channels + c) * output_h + out_h) * output_w + out_w;
            mask_indices[output_idx] = max_idx;
          }
        }
      }
    }
#endif

    micro_batch_mask_indices_[micro_batch_id] = std::move(mask_indices);
    return output;
  }

  Tensor<T> backward(const Tensor<T> &grad_output,
                     int micro_batch_id = 0) override {
    auto it_input = micro_batch_inputs_.find(micro_batch_id);
    auto it_mask = micro_batch_mask_indices_.find(micro_batch_id);

    if (it_input == micro_batch_inputs_.end()) {
      throw std::runtime_error(
          "No cached input found for micro-batch ID in MaxPool2DLayer: " +
          std::to_string(micro_batch_id));
    }
    if (it_mask == micro_batch_mask_indices_.end()) {
      throw std::runtime_error(
          "No cached mask found for micro-batch ID in MaxPool2DLayer: " +
          std::to_string(micro_batch_id));
    }

    const Tensor<T> &last_input = it_input->second;
    const std::vector<size_t> &mask_indices = it_mask->second;

    const size_t batch_size = last_input.batch_size();
    const size_t channels = last_input.channels();
    const size_t input_h = last_input.height();
    const size_t input_w = last_input.width();
    const size_t output_h = grad_output.height();
    const size_t output_w = grad_output.width();

    Tensor<T> grad_input(
        std::vector<size_t>{batch_size, channels, input_h, input_w});
    grad_input.fill(T(0));

    const T *grad_output_data = grad_output.data();
    T *grad_input_data = grad_input.data();

    const size_t total_outputs = batch_size * channels * output_h * output_w;

    // Unified backward pass - handles all cases cleanly
#if defined(USE_TBB)
    tnn::parallel_for_range<size_t>(0, total_outputs, [&](size_t i) {
      const size_t output_hw = output_h * output_w;
      const size_t output_chw = channels * output_hw;

      const size_t n = i / output_chw;
      const size_t c = (i % output_chw) / output_hw;
      const size_t out_h = (i % output_hw) / output_w;
      const size_t out_w = i % output_w;

      const size_t max_idx = mask_indices[i];
      const size_t max_h = max_idx / input_w;
      const size_t max_w = max_idx % input_w;

      // Bounds check (handles edge cases)
      if (max_h < input_h && max_w < input_w) {
        const T grad_val =
            grad_output_data[n * output_stride_n_ + c * output_stride_c_ +
                             out_h * output_stride_h_ +
                             out_w * output_stride_w_];

        grad_input_data[n * input_stride_n_ + c * input_stride_c_ +
                        max_h * input_stride_h_ + max_w * input_stride_w_] +=
            grad_val;
      }
    });
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < total_outputs; ++i) {
      const size_t output_hw = output_h * output_w;
      const size_t output_chw = channels * output_hw;

      const size_t n = i / output_chw;
      const size_t c = (i % output_chw) / output_hw;
      const size_t out_h = (i % output_hw) / output_w;
      const size_t out_w = i % output_w;

      const size_t max_idx = mask_indices[i];
      const size_t max_h = max_idx / input_w;
      const size_t max_w = max_idx % input_w;

      // Bounds check (handles edge cases)
      if (max_h < input_h && max_w < input_w) {
        const T grad_val =
            grad_output_data[n * output_stride_n_ + c * output_stride_c_ +
                             out_h * output_stride_h_ +
                             out_w * output_stride_w_];

        grad_input_data[n * input_stride_n_ + c * input_stride_c_ +
                        max_h * input_stride_h_ + max_w * input_stride_w_] +=
            grad_val;
      }
    }
#endif

    return grad_input;
  }

  std::string type() const override { return "maxpool2d"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["pool_h"] = pool_h_;
    config.parameters["pool_w"] = pool_w_;
    config.parameters["stride_h"] = stride_h_;
    config.parameters["stride_w"] = stride_w_;
    config.parameters["pad_h"] = pad_h_;
    config.parameters["pad_w"] = pad_w_;
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    return std::make_unique<MaxPool2DLayer<T>>(
        pool_h_, pool_w_, stride_h_, stride_w_, pad_h_, pad_w_, this->name_);
  }

  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const override {
    if (input_shape.size() != 4) {
      throw std::invalid_argument("MaxPool2DLayer expects 4D input");
    }

    size_t output_h = (input_shape[2] + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
    size_t output_w = (input_shape[3] + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

    return {input_shape[0], input_shape[1], output_h, output_w};
  }
};

} // namespace tnn