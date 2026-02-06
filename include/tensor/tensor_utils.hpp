#include <iomanip>

#include "tensor.hpp"

namespace tnn {

template <typename T>
void check_nan_and_inf(const T *data, size_t size, const std::string &tensor_name = "") {
  for (size_t i = 0; i < size; ++i) {
    if (std::isnan(data[i]) || std::isinf(data[i])) {
      std::cerr << "TypedTensor " << tensor_name << " contains NaN or Inf at index " << i
                << std::endl;
      return;
    }
  }
}

template <typename T>
void check_nan_and_inf(const TypedTensor<T> &tensor, const std::string &tensor_name = "") {
  auto cpu_tensor = std::dynamic_pointer_cast<TypedTensor<T>>(tensor.to_cpu());
  size_t total_elements = cpu_tensor->size();
  T *data = cpu_tensor->data_ptr().template get<T>();
  check_nan_and_inf(data, total_elements, tensor_name);
}

inline void check_nan_and_inf(const ConstTensor &tensor, const std::string &tensor_name = "") {
  DType_t dtype = tensor->data_type();
  switch (dtype) {
    case DType_t::FP32: {
      auto typed_tensor = std::dynamic_pointer_cast<TypedTensor<float>>(tensor);
      check_nan_and_inf<float>(*typed_tensor, tensor_name);
      break;
    }
    case DType_t::FP64: {
      auto typed_tensor = std::dynamic_pointer_cast<TypedTensor<double>>(tensor);
      check_nan_and_inf<double>(*typed_tensor, tensor_name);
      break;
    }
    case DType_t::FP16: {
      throw std::runtime_error("check_nan_and_inf not implemented for FP16 tensors");
      break;
    }
    default:
      throw std::runtime_error("Unsupported data type for check_nan_and_inf");
  }
}

// Prints data density at ranges (2^-32, 2^-31, ..., 2^31, 2^32)
inline void print_data_distribution(const ConstTensor &tensor,
                                    const std::string &tensor_name = "") {
  if (!tensor) {
    std::cerr << "Cannot print distribution of null tensor" << std::endl;
    return;
  }

  Tensor cpu_tensor = tensor->to_cpu();
  DType_t dtype = cpu_tensor->data_type();

  constexpr int min_exp = -32;
  constexpr int max_exp = 32;
  constexpr int num_buckets = max_exp - min_exp + 1;

  // buckets[0] = values < 2^-32 (including zeros)
  // buckets[1..num_buckets] = values in [2^exp, 2^(exp+1))
  // buckets[num_buckets+1] = values >= 2^32
  std::vector<size_t> buckets(num_buckets + 2, 0);

  auto process_data = [&]<typename T>() {
    auto typed_tensor = std::dynamic_pointer_cast<TypedTensor<T>>(cpu_tensor);
    if (!typed_tensor) {
      throw std::runtime_error("Failed to cast tensor in print_data_distribution");
    }

    T *data = typed_tensor->data_ptr().template get<T>();
    size_t size = typed_tensor->size();

    for (size_t i = 0; i < size; ++i) {
      T val = data[i];
      double abs_val = std::abs(static_cast<double>(val));

      if (abs_val == 0.0 || abs_val < std::pow(2.0, min_exp)) {
        buckets[0]++;
      } else if (abs_val >= std::pow(2.0, max_exp + 1)) {
        buckets[num_buckets + 1]++;
      } else {
        double log2_val = std::log2(abs_val);
        int exp = static_cast<int>(std::floor(log2_val));

        exp = std::max(min_exp, std::min(max_exp, exp));
        int bucket_idx = exp - min_exp + 1;
        buckets[bucket_idx]++;
      }
    }
  };

  switch (dtype) {
    case DType_t::FP32:
      process_data.template operator()<float>();
      break;
    case DType_t::FP64:
      process_data.template operator()<double>();
      break;
    case DType_t::FP16:
      process_data.template operator()<fp16>();
      break;
    default:
      std::cerr << "Unsupported data type for print_data_distribution" << std::endl;
      return;
  }

  // Print distribution
  size_t total = cpu_tensor->size();
  std::cout << "\nData Distribution for tensor: " << tensor_name << " (shape "
            << cpu_tensor->shape_str() << ", " << total << " elements):\n";
  std::cout << std::setw(20) << "Range" << std::setw(15) << "Count" << std::setw(15)
            << "Percentage\n";
  std::cout << std::string(50, '-') << "\n";

  // Zero/very small values
  if (buckets[0] > 0) {
    double pct = 100.0 * buckets[0] / total;
    std::cout << std::setw(20) << "< 2^-32 (or zero)" << std::setw(15) << buckets[0]
              << std::setw(14) << std::fixed << std::setprecision(2) << pct << "%\n";
  }

  // Regular buckets - only show non-empty buckets
  for (int exp = min_exp; exp <= max_exp; ++exp) {
    int bucket_idx = exp - min_exp + 1;
    if (buckets[bucket_idx] > 0) {
      double pct = 100.0 * buckets[bucket_idx] / total;
      std::ostringstream range;
      range << "[2^" << exp << ", 2^" << (exp + 1) << ")";
      std::cout << std::setw(20) << range.str() << std::setw(15) << buckets[bucket_idx]
                << std::setw(14) << std::fixed << std::setprecision(2) << pct << "%\n";
    }
  }

  // Very large values
  if (buckets[num_buckets + 1] > 0) {
    double pct = 100.0 * buckets[num_buckets + 1] / total;
    std::cout << std::setw(20) << ">= 2^33" << std::setw(15) << buckets[num_buckets + 1]
              << std::setw(14) << std::fixed << std::setprecision(2) << pct << "%\n";
  }

  std::cout << std::endl;
}
}  // namespace tnn