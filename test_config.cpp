#include <any>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "common/config.hpp"

using namespace tnn;

int main() {
  TConfig config;
  std::vector<std::string> vec = {"hello", "world"};
  config.set("test_vector", vec);

  TConfig sub_config;
  sub_config.set("foo", 123);

  // Attempt to set TConfig directly
  config.set("sub_config", sub_config);

  nlohmann::json j = config.to_json();
  std::cout << j.dump(4) << std::endl;

  return 0;
}
