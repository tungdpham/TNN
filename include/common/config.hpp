/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <any>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

namespace tnn {

/**
 * @brief Unified configuration class with JSON serialization.
 *
 * This class stores configuration parameters with their exact types and provides
 * serialization to/from JSON. Values are stored directly in JSON format without
 * explicit type tags.
 *
 * JSON Format Example:
 * {
 *   "name": "MyConfig",
 *   "type": "layer",
 *   "parameters": {
 *     "learning_rate": 0.001,
 *     "size": 256,
 *     "momentum": 0.9
 *   }
 * }
 */
class TConfig {
public:
  TConfig() = default;

  /**
   * @brief Get a parameter value with type checking.
   * @tparam T The expected type of the parameter.
   * @param key The parameter key.
   * @param default_value The default value if the parameter doesn't exist or type mismatches.
   * @return The parameter value or default value.
   */
  template <typename T>
  T get(const std::string &key, const T &default_value = T{}) const;

  /**
   * @brief Set a parameter value.
   * @tparam T The type of the parameter.
   * @param key The parameter key.
   * @param value The parameter value.
   */
  template <typename T>
  void set(const std::string &key, const T &value);

  /**
   * @brief Check if a parameter exists.
   * @param key The parameter key.
   * @return True if the parameter exists, false otherwise.
   */
  bool has(const std::string &key) const;

  /**
   * @brief Remove a parameter.
   * @param key The parameter key.
   * @return True if the parameter was removed, false if it didn't exist.
   */
  bool remove(const std::string &key);

  /**
   * @brief Clear all parameters.
   */
  void clear_parameters();

  /**
   * @brief Serialize to JSON.
   * @return JSON object with parameters stored as direct values.
   */
  nlohmann::json to_json() const;

  /**
   * @brief Deserialize from JSON.
   * @param j JSON object to deserialize from.
   * @return TConfig object.
   */
  static TConfig from_json(const nlohmann::json &j);

  // Public members for name and type
  std::string name;
  std::string type;

  // Access to parameters (for iteration, debugging, etc.)
  const std::unordered_map<std::string, std::any> &get_parameters() const { return parameters_; }

private:
  std::unordered_map<std::string, std::any> parameters_;

  // Helper methods for type conversion
  static std::string type_name_from_any(const std::any &value);
  static nlohmann::json value_to_json(const std::any &value);
  static std::any json_to_value(const nlohmann::json &type_value_pair);
};

template <typename Archiver>
void archive(Archiver &archiver, const TConfig &config) {
  const std::string json_str = config.to_json().dump();
  archiver(json_str);
}

template <typename Archiver>
void archive(Archiver &archiver, TConfig &config) {
  std::string json_str;
  archiver(json_str);
  if (!json_str.empty()) {
    nlohmann::json j = nlohmann::json::parse(json_str);
    config = TConfig::from_json(j);
  } else {
    config = TConfig();  // Reset to default if empty JSON is provided
  }
}

}  // namespace tnn
