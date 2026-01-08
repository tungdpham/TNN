#pragma once

#include "parser.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

namespace tnn {

class EnvLoader {
public:
  EnvLoader(std::string file_path = "./.env") { load_env_file(file_path); }

  /**
   * Load environment variables from a .env file
   * @param file_path Path to the .env file (default: "./.env")
   * @return true if file was loaded successfully, false otherwise
   */
  bool load_env_file(const std::string &file_path = "./.env") {
    std::ifstream file(file_path);
    if (!file.is_open()) {
      std::cerr << "Warning: Could not open .env file: " << file_path << std::endl;
      return false;
    }

    std::string line;

    while (std::getline(file, line)) {
      size_t first = line.find_first_not_of(" \t\r\n");
      if (first == std::string::npos)
        continue;
      line.erase(0, first);
      line.erase(line.find_last_not_of(" \t\r\n") + 1);

      if (line.empty() || line[0] == '#') {
        continue;
      }

      size_t equals_pos = line.find('=');
      if (equals_pos == std::string::npos) {
        continue;
      }

      std::string key = line.substr(0, equals_pos);
      std::string value = line.substr(equals_pos + 1);

      // 2. Trim key and value again after split
      auto trim = [](std::string &s) {
        size_t first = s.find_first_not_of(" \t");
        if (first == std::string::npos) {
          s.clear();
          return;
        }
        s.erase(0, first);
        s.erase(s.find_last_not_of(" \t") + 1);
      };
      trim(key);
      trim(value);

      // 3. Robust quote stripping
      if (value.size() >= 2) {
        if ((value.front() == '"' && value.back() == '"') ||
            (value.front() == '\'' && value.back() == '\'')) {
          value = value.substr(1, value.size() - 2);
        }
      }

      // 4. Validate Key (Security)
      if (key.empty() || key.find_first_of("= \t") != std::string::npos) {
        continue; // Skip keys with invalid characters
      }

      env_vars_[key] = value;

      // set in system environment (note: this may not work on all systems)
#ifdef _WIN32
      _putenv_s(key.c_str(), value.c_str());
#else
      setenv(key.c_str(), value.c_str(), 1);
#endif
    }

    return true;
  }

  template <typename T = std::string> T get(const std::string &env_var, const T &default_value) {
    auto it = env_vars_.find(env_var);
    if (it != env_vars_.end()) {
      return from_str<T>(it->second);
    }

#ifdef _WIN32
#ifdef _MSC_VER
    char *env_value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&env_value, &len, env_var.c_str()) == 0 && env_value != nullptr) {
      std::string result(env_value);
      free(env_value);
      return from_str<T>(result);
    }
    return default_value;
#else
    const char *env_value = std::getenv(env_var.c_str());
    if (env_value) {
      return from_str<T>(std::string(env_value));
    }
    return default_value;
#endif
#else
    const char *env_value = std::getenv(env_var.c_str());
    if (env_value) {
      return from_str<T>(std::string(env_value));
    }
    return default_value;
#endif
  }

private:
  std::unordered_map<std::string, std::string> env_vars_;
};

class Env {
public:
  static EnvLoader &instance() {
    static EnvLoader instance;
    return instance;
  }

  template <typename T = std::string>
  static T get(const std::string &env_var, const T &default_value) {
    return instance().get<T>(env_var, default_value);
  }
};
} // namespace tnn