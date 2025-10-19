#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace tpipeline {
struct StageConfig {
  std::string stage_id;
  nlohmann::json model_config;
  std::string next_stage_endpoint;
  std::string prev_stage_endpoint;
  std::string coordinator_endpoint;

  nlohmann::json to_json() const {
    return nlohmann::json{{"stage_id", stage_id},
                          {"model_config", model_config},
                          {"next_stage_endpoint", next_stage_endpoint},
                          {"prev_stage_endpoint", prev_stage_endpoint},
                          {"coordinator_endpoint", coordinator_endpoint}};
  }

  static StageConfig from_json(const nlohmann::json &j) {
    StageConfig config;
    config.stage_id = j["stage_id"];
    config.model_config = j["model_config"];
    config.next_stage_endpoint = j["next_stage_endpoint"];
    config.prev_stage_endpoint = j["prev_stage_endpoint"];
    config.coordinator_endpoint = j["coordinator_endpoint"];
    return config;
  }
};

} // namespace tpipeline
