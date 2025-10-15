#pragma once

#include <cstdlib>
#include <string>

namespace utils {

std::string get_env(const std::string &env_var, const std::string &default_host) {
#ifdef _WIN32
#ifdef _MSC_VER

  char *env_value = nullptr;
  size_t len = 0;
  if (_dupenv_s(&env_value, &len, env_var.c_str()) == 0 && env_value != nullptr) {
    std::string result(env_value);
    free(env_value);
    return result;
  }
  return default_host;
#else
  const char *env_value = std::getenv(env_var.c_str());
  return env_value ? std::string(env_value) : default_host;
#endif
#else
  const char *env_value = std::getenv(env_var.c_str());
  return env_value ? std::string(env_value) : default_host;
#endif
}

} // namespace utils