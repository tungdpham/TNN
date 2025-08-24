#pragma once

#include <string>
#include <unordered_map>

namespace tpipeline {

struct StageEndpoint {
  std::string stage_id;
  std::string communication_type;
  std::string address;
  std::unordered_map<std::string, std::string> parameters;

  StageEndpoint() = default;

  StageEndpoint(std::string id, std::string comm_type, std::string addr)
      : stage_id(std::move(id)), communication_type(std::move(comm_type)),
        address(std::move(addr)) {}

  static StageEndpoint in_process(const std::string &id) {
    return StageEndpoint(id, "in_process", "memory");
  }

  static StageEndpoint network(const std::string &id, const std::string &host,
                               int port) {
    StageEndpoint endpoint(id, "tcp", host + ":" + std::to_string(port));
    endpoint.parameters["host"] = host;
    endpoint.parameters["port"] = std::to_string(port);
    return endpoint;
  }
};

} // namespace tpipeline