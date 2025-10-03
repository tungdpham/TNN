#include "data_loader.hpp"

namespace data_loading {
/**
 * Specialized base class for regression datasets
 * Provides common functionality for continuous target prediction
 */
template <typename T = float> class RegressionDataLoader : public BaseDataLoader<T> {
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

} // namespace data_loading