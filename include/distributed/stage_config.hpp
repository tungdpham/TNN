#pragma once

#include <nlohmann/json.hpp>

#include "endpoint.hpp"
#include "nn/layer.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"

namespace tnn {
struct StageConfig {
  LayerConfig model_config;
  OptimizerConfig optimizer_config;
  SchedulerConfig scheduler_config;
  Endpoint next_stage_endpoint;
  Endpoint prev_stage_endpoint;
  Endpoint coordinator_endpoint;

  nlohmann::json to_json() const {
    return nlohmann::json{{"model_config", model_config.to_json()},
                          {"optimizer_config", optimizer_config.to_json()},
                          {"scheduler_config", scheduler_config.to_json()},
                          {"next_stage_endpoint", next_stage_endpoint.to_json()},
                          {"prev_stage_endpoint", prev_stage_endpoint.to_json()},
                          {"coordinator_endpoint", coordinator_endpoint.to_json()}};
  }

  static StageConfig from_json(const nlohmann::json &j) {
    StageConfig config;
    config.model_config = LayerConfig::from_json(j["model_config"]);
    config.optimizer_config = OptimizerConfig::from_json(j["optimizer_config"]);
    config.scheduler_config = SchedulerConfig::from_json(j["scheduler_config"]);
    config.next_stage_endpoint = Endpoint::from_json(j["next_stage_endpoint"]);
    config.prev_stage_endpoint = Endpoint::from_json(j["prev_stage_endpoint"]);
    config.coordinator_endpoint = Endpoint::from_json(j["coordinator_endpoint"]);
    return config;
  }
};

}  // namespace tnn
