#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace tnn {

// Static storage for loaded environment variables from .env file
inline std::unordered_map<std::string, std::string> loaded_env_vars;
inline bool env_file_loaded = false;

namespace detail {
template <typename T> T convert_from_string(const std::string &str) {
  if constexpr (std::is_same_v<T, std::string>) {
    return str;
  } else if constexpr (std::is_arithmetic_v<T>) {
    std::istringstream iss(str);
    T value;
    if (iss >> value && iss.eof()) {
      return value;
    }
    throw std::invalid_argument("Cannot convert '" + str + "' to numeric type");
  } else {
    static_assert(std::is_same_v<T, std::string> || std::is_arithmetic_v<T>,
                  "get_env only supports string and arithmetic types");
    return T{};
  }
}
} // namespace detail

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
  size_t line_number = 0;

  while (std::getline(file, line)) {
    ++line_number;

    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Find the equals sign
    size_t equals_pos = line.find('=');
    if (equals_pos == std::string::npos) {
      std::cerr << "Warning: Invalid line " << line_number << " in " << file_path << ": " << line
                << std::endl;
      continue;
    }

    std::string key = line.substr(0, equals_pos);
    std::string value = line.substr(equals_pos + 1);

    // Trim whitespace from key
    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);

    // Handle quoted values
    if (!value.empty()) {
      // Remove surrounding quotes if present
      if ((value.front() == '"' && value.back() == '"') ||
          (value.front() == '\'' && value.back() == '\'')) {
        value = value.substr(1, value.length() - 2);
      }
    }

    // Store in our map and set in system environment
    loaded_env_vars[key] = value;

    // Set in system environment (note: this may not work on all systems)
#ifdef _WIN32
    _putenv_s(key.c_str(), value.c_str());
#else
    setenv(key.c_str(), value.c_str(), 1);
#endif
  }

  env_file_loaded = true;
  std::cout << "Loaded " << loaded_env_vars.size() << " environment variables from " << file_path
            << std::endl;
  return true;
}

template <typename T = std::string> T get_env(const std::string &env_var, const T &default_value) {
  // First check if we have it in our loaded env vars
  auto it = loaded_env_vars.find(env_var);
  if (it != loaded_env_vars.end()) {
    return detail::convert_from_string<T>(it->second);
  }

  // Fall back to system environment variables
#ifdef _WIN32
#ifdef _MSC_VER
  char *env_value = nullptr;
  size_t len = 0;
  if (_dupenv_s(&env_value, &len, env_var.c_str()) == 0 && env_value != nullptr) {
    std::string result(env_value);
    free(env_value);
    return detail::convert_from_string<T>(result);
  }
  return default_value;
#else
  const char *env_value = std::getenv(env_var.c_str());
  if (env_value) {
    return detail::convert_from_string<T>(std::string(env_value));
  }
  return default_value;
#endif
#else
  const char *env_value = std::getenv(env_var.c_str());
  if (env_value) {
    return detail::convert_from_string<T>(std::string(env_value));
  }
  return default_value;
#endif
}

/**
 * Automatically load .env file if it hasn't been loaded yet
 * This is called internally by get_env_auto but can be called manually
 */
void ensure_env_loaded(const std::string &file_path = "./.env") {
  if (!env_file_loaded) {
    load_env_file(file_path);
  }
}

/**
 * Get environment variable with automatic .env file loading
 * This version will automatically try to load ./.env if it hasn't been loaded yet
 */
template <typename T = std::string>
T get_env_auto(const std::string &env_var, const T &default_value,
               const std::string &env_file = "./.env") {
  ensure_env_loaded(env_file);
  return get_env<T>(env_var, default_value);
}

} // namespace tnn