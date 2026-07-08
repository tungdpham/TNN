#pragma once

#include <nlohmann/json.hpp>
#include <string>

#include "endpoint.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"

namespace tunx {
struct StageConfig {
  std::string graph_state;
  Vec<std::string> input_uids;
  Vec<std::string> output_uids;
  OptimizerConfig optimizer_config;
  SchedulerConfig scheduler_config;
  Endpoint next_stage_endpoint;
  Endpoint prev_stage_endpoint;
  Endpoint coordinator_endpoint;

  nlohmann::json to_json() const {
    return nlohmann::json{{"graph_state_bytes", graph_state.size()},
                          {"input_uids", input_uids},
                          {"output_uids", output_uids},
                          {"optimizer_config", optimizer_config.to_json()},
                          {"scheduler_config", scheduler_config.to_json()},
                          {"next_stage_endpoint", next_stage_endpoint.to_json()},
                          {"prev_stage_endpoint", prev_stage_endpoint.to_json()},
                          {"coordinator_endpoint", coordinator_endpoint.to_json()}};
  }

  static StageConfig from_json(const nlohmann::json &j) {
    StageConfig config;
    if (j.contains("graph_state")) {
      config.graph_state = j["graph_state"].get<std::string>();
    }
    if (j.contains("input_uids")) {
      config.input_uids = j["input_uids"].get<Vec<std::string>>();
    }
    if (j.contains("output_uids")) {
      config.output_uids = j["output_uids"].get<Vec<std::string>>();
    }
    config.optimizer_config = OptimizerConfig::from_json(j["optimizer_config"]);
    config.scheduler_config = SchedulerConfig::from_json(j["scheduler_config"]);
    config.next_stage_endpoint = Endpoint::from_json(j["next_stage_endpoint"]);
    config.prev_stage_endpoint = Endpoint::from_json(j["prev_stage_endpoint"]);
    config.coordinator_endpoint = Endpoint::from_json(j["coordinator_endpoint"]);
    return config;
  }
};

template <typename Archiver>
void archive(Archiver &archiver, const StageConfig &config) {
  archiver(config.graph_state, config.input_uids, config.output_uids, config.optimizer_config,
           config.scheduler_config, config.next_stage_endpoint, config.prev_stage_endpoint,
           config.coordinator_endpoint);
}

template <typename Archiver>
void archive(Archiver &archiver, StageConfig &config) {
  archiver(config.graph_state, config.input_uids, config.output_uids, config.optimizer_config,
           config.scheduler_config, config.next_stage_endpoint, config.prev_stage_endpoint,
           config.coordinator_endpoint);
}
}  // namespace tunx
