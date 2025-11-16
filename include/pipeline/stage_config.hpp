#pragma once

#include "endpoint.hpp"
#include <nlohmann/json.hpp>
#include <string>

namespace tnn {
struct StageConfig {
  std::string stage_id;
  nlohmann::json model_config;
  Endpoint next_stage_endpoint;
  Endpoint prev_stage_endpoint;
  Endpoint coordinator_endpoint;

  nlohmann::json to_json() const {
    return nlohmann::json{{"stage_id", stage_id},
                          {"model_config", model_config},
                          {"next_stage_endpoint", next_stage_endpoint.to_json()},
                          {"prev_stage_endpoint", prev_stage_endpoint.to_json()},
                          {"coordinator_endpoint", coordinator_endpoint.to_json()}};
  }

  static StageConfig from_json(const nlohmann::json &j) {
    StageConfig config;
    config.stage_id = j["stage_id"];
    config.model_config = j["model_config"];
    config.next_stage_endpoint = Endpoint::from_json(j["next_stage_endpoint"]);
    config.prev_stage_endpoint = Endpoint::from_json(j["prev_stage_endpoint"]);
    config.coordinator_endpoint = Endpoint::from_json(j["coordinator_endpoint"]);
    return config;
  }
};

} // namespace tnn
