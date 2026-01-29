/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nlohmann/json.hpp"
#include <string>
#include <unordered_map>

namespace tnn {

class Communicator;

enum class CommunicationType { IN_PROCESS, TCP, ROCE, NONE };

struct Endpoint {
private:
  CommunicationType type_;
  std::unordered_map<std::string, std::any> parameters_;

public:
  Endpoint() = default;

  explicit Endpoint(CommunicationType comm_type) : type_(comm_type) {}

  CommunicationType type() const { return type_; }

  template <typename T> T get_parameter(const std::string &key) const {
    auto it = parameters_.find(key);
    if (it == parameters_.end()) {
      throw std::runtime_error("Parameter " + key + " not found");
    }
    try {
      return std::any_cast<T>(it->second);
    } catch (const std::bad_any_cast &) {
      throw std::runtime_error("Parameter type mismatch for key: " + key);
    }
  }

  template <typename T> void set_parameter(const std::string &key, T value) {
    parameters_[key] = std::move(value);
  }

  bool is_empty() const { return type_ == CommunicationType::NONE; }

  size_t hash() const {
    size_t h = std::hash<int>{}(static_cast<int>(type_));
    // Combine hashes of parameters
    for (const auto &[key, value] : parameters_) {
      // Hash the key
      h ^= std::hash<std::string>{}(key) + 0x9e3779b9 + (h << 6) + (h >> 2);
      // Hash the value based on its type
      if (value.type() == typeid(std::string)) {
        h ^= std::hash<std::string>{}(std::any_cast<std::string>(value)) + 0x9e3779b9 + (h << 6) +
             (h >> 2);
      } else if (value.type() == typeid(int)) {
        h ^= std::hash<int>{}(std::any_cast<int>(value)) + 0x9e3779b9 + (h << 6) + (h >> 2);
      } else if (value.type() == typeid(double)) {
        h ^= std::hash<double>{}(std::any_cast<double>(value)) + 0x9e3779b9 + (h << 6) + (h >> 2);
      } else if (value.type() == typeid(float)) {
        h ^= std::hash<float>{}(std::any_cast<float>(value)) + 0x9e3779b9 + (h << 6) + (h >> 2);
      } else if (value.type() == typeid(bool)) {
        h ^= std::hash<bool>{}(std::any_cast<bool>(value)) + 0x9e3779b9 + (h << 6) + (h >> 2);
      } else if (value.type() == typeid(Communicator *)) {
        h ^= std::hash<Communicator *>{}(std::any_cast<Communicator *>(value)) + 0x9e3779b9 +
             (h << 6) + (h >> 2);
      }
    }
    return h;
  }

  std::string id() const {
    switch (type_) {
    case CommunicationType::IN_PROCESS:
      return "IN_PROCESS";
    case CommunicationType::TCP:
      return get_parameter<std::string>("host") + ":" + std::to_string(get_parameter<int>("port"));
    case CommunicationType::ROCE:
      return get_parameter<std::string>("host") + ":" + std::to_string(get_parameter<int>("port"));
    case CommunicationType::NONE:
      return "NONE";
    default:
      return "UNKNOWN";
    }
  }

  bool operator==(const Endpoint &other) const {
    if (type_ != other.type_) {
      return false;
    }
    if (parameters_.size() != other.parameters_.size()) {
      return false;
    }
    for (const auto &[key, value] : parameters_) {
      auto it = other.parameters_.find(key);
      if (it == other.parameters_.end()) {
        return false;
      }
      const auto &other_value = it->second;
      // Compare std::any values by checking types first
      if (value.type() != other_value.type()) {
        return false;
      }
      // Compare based on the actual type
      if (value.type() == typeid(std::string)) {
        if (std::any_cast<std::string>(value) != std::any_cast<std::string>(other_value)) {
          return false;
        }
      } else if (value.type() == typeid(int)) {
        if (std::any_cast<int>(value) != std::any_cast<int>(other_value)) {
          return false;
        }
      } else if (value.type() == typeid(double)) {
        if (std::any_cast<double>(value) != std::any_cast<double>(other_value)) {
          return false;
        }
      } else if (value.type() == typeid(float)) {
        if (std::any_cast<float>(value) != std::any_cast<float>(other_value)) {
          return false;
        }
      } else if (value.type() == typeid(bool)) {
        if (std::any_cast<bool>(value) != std::any_cast<bool>(other_value)) {
          return false;
        }
      } else if (value.type() == typeid(Communicator *)) {
        if (std::any_cast<Communicator *>(value) != std::any_cast<Communicator *>(other_value)) {
          return false;
        }
      }
      // For other types, we skip comparison (assume equal if same type)
    }
    return true;
  }

  static Endpoint empty() { return Endpoint(CommunicationType::NONE); }

  static Endpoint tcp(const std::string &host, int port) {
    Endpoint endpoint(CommunicationType::TCP);
    endpoint.set_parameter("host", host);
    endpoint.set_parameter("port", port);
    return endpoint;
  }

  static Endpoint roce(const std::string &host, int port, const std::string &device_name,
                       int gid_index) {
    Endpoint endpoint(CommunicationType::ROCE);
    endpoint.set_parameter("host", host);
    endpoint.set_parameter("port", port);
    endpoint.set_parameter("device_name", device_name);
    endpoint.set_parameter("gid_index", gid_index);
    return endpoint;
  }

  static Endpoint in_process(Communicator *comm) {
    Endpoint endpoint(CommunicationType::IN_PROCESS);
    endpoint.set_parameter("communicator", comm);
    return endpoint;
  }

  nlohmann::json to_json() const {
    nlohmann::json j;
    j["type_"] = static_cast<int>(type_);
    nlohmann::json param_json = nlohmann::json::object();

    for (const auto &pair : parameters_) {
      const auto &key = pair.first;
      const auto &val = pair.second;

      if (val.type() == typeid(std::string)) {
        param_json[key] = std::any_cast<std::string>(val);
      } else if (val.type() == typeid(const char *)) {
        param_json[key] = std::string(std::any_cast<const char *>(val));
      } else if (val.type() == typeid(int)) {
        param_json[key] = std::any_cast<int>(val);
      } else if (val.type() == typeid(double)) {
        param_json[key] = std::any_cast<double>(val);
      } else if (val.type() == typeid(float)) {
        param_json[key] = std::any_cast<float>(val);
      }
    }

    j["parameters_"] = param_json;
    return j;
  }

  static Endpoint from_json(const nlohmann::json &j) {
    Endpoint endpoint;
    endpoint.type_ = static_cast<CommunicationType>(j.at("type_").get<int>());

    if (j.contains("parameters_")) {
      for (auto &[key, value] : j["parameters_"].items()) {
        if (value.is_string()) {
          endpoint.parameters_[key] = value.get<std::string>();
        } else if (value.is_number_integer()) {
          endpoint.parameters_[key] = value.get<int>();
        } else if (value.is_number_float()) {
          endpoint.parameters_[key] = value.get<double>();
        } else if (value.is_boolean()) {
          endpoint.parameters_[key] = value.get<bool>();
        }
      }
    }
    return endpoint;
  }
};

} // namespace tnn

// Hash function specialization for tnn::Endpoint
namespace std {
template <> struct hash<tnn::Endpoint> {
  size_t operator()(const tnn::Endpoint &endpoint) const noexcept { return endpoint.hash(); }
};
} // namespace std