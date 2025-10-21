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
#include <variant>

namespace tpipeline {

class Communicator;

struct Endpoint {
private:
  std::string communication_type_;
  std::unordered_map<std::string, std::any> parameters_;

public:
  Endpoint() = default;

  explicit Endpoint(std::string comm_type) : communication_type_(std::move(comm_type)) {}

  const std::string &communication_type() const { return communication_type_; }

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

  static Endpoint network(const std::string &host, int port) {
    Endpoint endpoint("tcp");
    endpoint.set_parameter("host", host);
    endpoint.set_parameter("port", std::to_string(port));
    return endpoint;
  }

  static Endpoint in_process(Communicator *comm) {
    Endpoint endpoint("in_process");
    endpoint.set_parameter("communicator", comm);
    return endpoint;
  }

  nlohmann::json to_json() const {
    nlohmann::json j;
    j["communication_type_"] = communication_type_;
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
    endpoint.communication_type_ = j.at("communication_type_").get<std::string>();

    if (j.contains("parameters_")) {
      for (auto &[key, value] : j["parameters_"].items()) {
        endpoint.parameters_[key] = value.get<std::string>();
      }
    }
    return endpoint;
  }
};

} // namespace tpipeline