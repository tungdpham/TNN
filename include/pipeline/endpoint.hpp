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

namespace tpipeline {

struct Endpoint {
  std::string communication_type;
  std::unordered_map<std::string, std::string> parameters;

  Endpoint() = default;

  Endpoint(std::string comm_type) : communication_type(std::move(comm_type)) {}

  const std::string &get_parameter(const std::string &key) const {
    auto it = parameters.find(key);
    if (it != parameters.end()) {
      return it->second;
    } else {
      throw std::runtime_error("Parameter " + key + " not found in endpoint");
    }
  }

  static Endpoint in_process() { return Endpoint("in_process"); }

  static Endpoint network(const std::string &host, int port) {
    Endpoint endpoint("tcp");
    endpoint.parameters["host"] = host;
    endpoint.parameters["port"] = std::to_string(port);
    return endpoint;
  }

  nlohmann::json to_json() const {
    nlohmann::json j;
    j["communication_type"] = communication_type;
    j["parameters"] = parameters;
    return j;
  }

  static Endpoint from_json(const nlohmann::json &j) {
    Endpoint endpoint;
    endpoint.communication_type = j["communication_type"];
    endpoint.parameters = j["parameters"].get<std::unordered_map<std::string, std::string>>();
    return endpoint;
  }
};

} // namespace tpipeline