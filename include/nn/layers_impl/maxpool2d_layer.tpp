/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "maxpool2d_layer.hpp"

#include <limits>
#include <stdexcept>

#include "utils/parallel_for.hpp"

namespace tnn {

template <typename T>
MaxPool2DLayer<T>::MaxPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                                  size_t pad_h, size_t pad_w, const std::string &name)
    : StatelessLayer<T>(name), pool_h_(pool_h), pool_w_(pool_w),
      stride_h_(stride_h == 0 ? pool_h : stride_h), stride_w_(stride_w == 0 ? pool_w : stride_w),
      pad_h_(pad_h), pad_w_(pad_w) {

  if (pool_h_ == 0 || pool_w_ == 0) {
    throw std::invalid_argument("Pool dimensions must be positive");
  }
  if (stride_h_ == 0 || stride_w_ == 0) {
    throw std::invalid_argument("Stride dimensions must be positive");
  }
}

template <typename T>
Tensor<T> MaxPool2DLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {

  const size_t batch_size = input.batch_size();
  const size_t channels = input.channels();

  const Tensor<T> *padded_input_ptr;
  std::unique_ptr<Tensor<T>> padded_input_storage;

  if (pad_h_ > 0 || pad_w_ > 0) {
    padded_input_storage = std::make_unique<Tensor<T>>(input.pad(pad_h_, pad_w_, T(0)));
    padded_input_ptr = padded_input_storage.get();
  } else {
    padded_input_ptr = &input;
  }

  const size_t padded_h = padded_input_ptr->height();
  const size_t padded_w = padded_input_ptr->width();

  const size_t output_h = (padded_h - pool_h_) / stride_h_ + 1;
  const size_t output_w = (padded_w - pool_w_) / stride_w_ + 1;

  Tensor<T> output(batch_size, channels, output_h, output_w, nullptr);

  const size_t total_outputs = batch_size * channels * output_h * output_w;
  std::vector<size_t> mask_indices(total_outputs);

  const T *padded_data = padded_input_ptr->data();
  T *output_data = output.data();

  compute_max_pool_forward(padded_data, output_data, batch_size, channels, padded_h, padded_w,
                           output_h, output_w, mask_indices);

  micro_batch_mask_indices_[micro_batch_id] = std::move(mask_indices);
  micro_batch_inputs_[micro_batch_id] = padded_input_ptr->clone();

  return output;
}

template <typename T>
Tensor<T> MaxPool2DLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
  auto it_input = micro_batch_inputs_.find(micro_batch_id);
  auto it_mask = micro_batch_mask_indices_.find(micro_batch_id);

  if (it_input == micro_batch_inputs_.end()) {
    throw std::runtime_error("No cached input found for micro-batch ID in MaxPool2DLayer: " +
                             std::to_string(micro_batch_id));
  }
  if (it_mask == micro_batch_mask_indices_.end()) {
    throw std::runtime_error("No cached mask found for micro-batch ID in MaxPool2DLayer: " +
                             std::to_string(micro_batch_id));
  }

  const Tensor<T> &cached_padded_input = it_input->second;
  const std::vector<size_t> &mask_indices = it_mask->second;

  const size_t batch_size = cached_padded_input.batch_size();
  const size_t channels = cached_padded_input.channels();
  const size_t output_h = gradient.height();
  const size_t output_w = gradient.width();

  Tensor<T> grad_padded_input(cached_padded_input.shape());

  const T *gradient_data = gradient.data();
  T *grad_padded_data = grad_padded_input.data();

  compute_max_pool_backward(gradient_data, grad_padded_data, batch_size, channels, output_h,
                            output_w, mask_indices);

  if (pad_h_ > 0 || pad_w_ > 0) {
    return grad_padded_input.unpad(pad_h_, pad_w_);
  } else {
    return grad_padded_input;
  }
}

template <typename T>
void MaxPool2DLayer<T>::compute_max_pool_forward(const T *input_data, T *output_data,
                                                 size_t batch_size, size_t channels, size_t input_h,
                                                 size_t input_w, size_t output_h, size_t output_w,
                                                 std::vector<size_t> &mask_indices) const {
  const T MIN_VALUE = std::numeric_limits<T>::lowest();

  utils::parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {
        T max_val = MIN_VALUE;
        size_t max_idx = 0;
        for (size_t ph = 0; ph < pool_h_; ++ph) {
          for (size_t pw = 0; pw < pool_w_; ++pw) {
            const size_t h_idx = out_h * stride_h_ + ph;
            const size_t w_idx = out_w * stride_w_ + pw;

            const size_t target_padded_idx =
                ((n * channels + c) * input_h + h_idx) * input_w + w_idx;
            T val = input_data[target_padded_idx];
            if (val > max_val) {
              max_val = val;
              max_idx = target_padded_idx;
            }
          }
        }

        const size_t output_idx = ((n * channels + c) * output_h + out_h) * output_w + out_w;
        output_data[output_idx] = max_val;
        mask_indices[output_idx] = max_idx;
      }
    }
  });
}

template <typename T>
void MaxPool2DLayer<T>::compute_max_pool_backward(const T *gradient_data, T *grad_input_data,
                                                  size_t batch_size, size_t channels,
                                                  size_t output_h, size_t output_w,
                                                  const std::vector<size_t> &mask_indices) const {
  utils::parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {
        const size_t output_idx = ((n * channels + c) * output_h + out_h) * output_w + out_w;
        const T grad_val = gradient_data[output_idx];
        const size_t input_idx = mask_indices[output_idx];
        grad_input_data[input_idx] += grad_val;
      }
    }
  });
}

template <typename T> std::string MaxPool2DLayer<T>::type() const { return "maxpool2d"; }

template <typename T> LayerConfig MaxPool2DLayer<T>::get_config() const {
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

template <typename T> std::unique_ptr<Layer<T>> MaxPool2DLayer<T>::clone() const {
  return std::make_unique<MaxPool2DLayer<T>>(pool_h_, pool_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                                             this->name_);
}

template <typename T>
std::vector<size_t>
MaxPool2DLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("MaxPool2DLayer expects 4D input");
  }

  size_t output_h = (input_shape[2] + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[3] + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

  return {input_shape[0], input_shape[1], output_h, output_w};
}

template <typename T>
std::unique_ptr<Layer<T>> MaxPool2DLayer<T>::create_from_config(const LayerConfig &config) {
  size_t pool_h = config.get<size_t>("pool_h");
  size_t pool_w = config.get<size_t>("pool_w");
  size_t stride_h = config.get<size_t>("stride_h");
  size_t stride_w = config.get<size_t>("stride_w");
  size_t pad_h = config.get<size_t>("pad_h");
  size_t pad_w = config.get<size_t>("pad_w");

  return std::make_unique<MaxPool2DLayer<T>>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                             config.name);
}

template <typename T>
uint32_t MaxPool2DLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  // Forward pass: for each output element, we do pool_h * pool_w comparisons
  // Total operations = batch_size * channels * output_h * output_w * (pool_h * pool_w)
  size_t batch_size = input_shape[0];
  size_t channels = input_shape[1];
  size_t input_h = input_shape[2];
  size_t input_w = input_shape[3];

  size_t padded_h = input_h + 2 * pad_h_;
  size_t padded_w = input_w + 2 * pad_w_;

  size_t output_h = (padded_h - pool_h_) / stride_h_ + 1;
  size_t output_w = (padded_w - pool_w_) / stride_w_ + 1;

  size_t total_operations = batch_size * channels * output_h * output_w * (pool_h_ * pool_w_);

  return static_cast<uint32_t>(total_operations);
}

template <typename T>
uint32_t MaxPool2DLayer<T>::backward_complexity(const std::vector<size_t> &gradient_shape) {
  assert(gradient_shape.size() == 4 && "Gradient shape must be 4D");
  // Backward pass: for each output gradient element, we do one addition to the input gradient
  // Total operations = batch_size * channels * output_h * output_w
  size_t batch_size = gradient_shape[0];
  size_t channels = gradient_shape[1];
  size_t output_h = gradient_shape[2];
  size_t output_w = gradient_shape[3];

  size_t total_operations = batch_size * channels * output_h * output_w;

  return static_cast<uint32_t>(total_operations);
}

} // namespace tnn
