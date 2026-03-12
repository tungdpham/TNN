/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "common/config.hpp"

#include <iostream>
#include <stdexcept>
#include <typeinfo>

namespace tnn {

template <typename T>
T TConfig::get(const std::string &key, const T &default_value) const {
  auto it = parameters_.find(key);
  if (it != parameters_.end()) {
    try {
      return std::any_cast<T>(it->second);
    } catch (const std::bad_any_cast &) {
      std::cerr << "Warning: TConfig parameter '" << key
                << "' type mismatch. Returning default value." << std::endl;
      return default_value;
    }
  }
  return default_value;
}

// Explicit template instantiations for common types
template int TConfig::get<int>(const std::string &, const int &) const;
template long TConfig::get<long>(const std::string &, const long &) const;
template long long TConfig::get<long long>(const std::string &, const long long &) const;
template unsigned int TConfig::get<unsigned int>(const std::string &, const unsigned int &) const;
template unsigned long TConfig::get<unsigned long>(const std::string &,
                                                   const unsigned long &) const;
template unsigned long long TConfig::get<unsigned long long>(const std::string &,
                                                             const unsigned long long &) const;
template float TConfig::get<float>(const std::string &, const float &) const;
template double TConfig::get<double>(const std::string &, const double &) const;
template bool TConfig::get<bool>(const std::string &, const bool &) const;
template std::string TConfig::get<std::string>(const std::string &, const std::string &) const;
template nlohmann::json TConfig::get<nlohmann::json>(const std::string &,
                                                     const nlohmann::json &) const;
template std::vector<size_t> TConfig::get<std::vector<size_t>>(const std::string &,
                                                               const std::vector<size_t> &) const;
template std::vector<int> TConfig::get<std::vector<int>>(const std::string &,
                                                         const std::vector<int> &) const;
template std::vector<float> TConfig::get<std::vector<float>>(const std::string &,
                                                             const std::vector<float> &) const;
template std::vector<double> TConfig::get<std::vector<double>>(const std::string &,
                                                               const std::vector<double> &) const;
template std::vector<std::string> TConfig::get<std::vector<std::string>>(
    const std::string &, const std::vector<std::string> &) const;

template <typename T>
void TConfig::set(const std::string &key, const T &value) {
  parameters_[key] = value;
}

// Explicit template instantiations for set
template void TConfig::set<int>(const std::string &, const int &);
template void TConfig::set<long>(const std::string &, const long &);
template void TConfig::set<long long>(const std::string &, const long long &);
template void TConfig::set<unsigned int>(const std::string &, const unsigned int &);
template void TConfig::set<unsigned long>(const std::string &, const unsigned long &);
template void TConfig::set<unsigned long long>(const std::string &, const unsigned long long &);
template void TConfig::set<float>(const std::string &, const float &);
template void TConfig::set<double>(const std::string &, const double &);
template void TConfig::set<bool>(const std::string &, const bool &);
template void TConfig::set<std::string>(const std::string &, const std::string &);
template void TConfig::set<nlohmann::json>(const std::string &, const nlohmann::json &);
template void TConfig::set<std::vector<size_t>>(const std::string &, const std::vector<size_t> &);
template void TConfig::set<std::vector<int>>(const std::string &, const std::vector<int> &);
template void TConfig::set<std::vector<float>>(const std::string &, const std::vector<float> &);
template void TConfig::set<std::vector<double>>(const std::string &, const std::vector<double> &);
template void TConfig::set<std::vector<std::string>>(const std::string &,
                                                     const std::vector<std::string> &);

bool TConfig::has(const std::string &key) const {
  return parameters_.find(key) != parameters_.end();
}

bool TConfig::remove(const std::string &key) { return parameters_.erase(key) > 0; }

void TConfig::clear_parameters() { parameters_.clear(); }

std::string TConfig::type_name_from_any(const std::any &value) {
  if (value.type() == typeid(int)) {
    return "int";
  } else if (value.type() == typeid(long)) {
    return "long";
  } else if (value.type() == typeid(long long)) {
    return "long long";
  } else if (value.type() == typeid(unsigned int)) {
    return "unsigned int";
  } else if (value.type() == typeid(unsigned long)) {
    return "unsigned long";
  } else if (value.type() == typeid(unsigned long long)) {
    return "unsigned long long";
  } else if (value.type() == typeid(size_t)) {
    return "size_t";
  } else if (value.type() == typeid(float)) {
    return "float";
  } else if (value.type() == typeid(double)) {
    return "double";
  } else if (value.type() == typeid(bool)) {
    return "bool";
  } else if (value.type() == typeid(std::string)) {
    return "string";
  } else if (value.type() == typeid(nlohmann::json)) {
    return "json";
  } else if (value.type() == typeid(std::vector<size_t>)) {
    return "vector<size_t>";
  } else if (value.type() == typeid(std::vector<int>)) {
    return "vector<int>";
  } else if (value.type() == typeid(std::vector<float>)) {
    return "vector<float>";
  } else if (value.type() == typeid(std::vector<double>)) {
    return "vector<double>";
  } else if (value.type() == typeid(std::vector<std::string>)) {
    return "vector<string>";
  }
  return "unknown";
}

template <typename T>
static void assign_value(nlohmann::json &j, const std::any &value) {
  T val = std::any_cast<T>(value);
  j["value"] = val;
}

nlohmann::json TConfig::value_to_json(const std::any &value) {
  nlohmann::json result;
  std::string type_name = type_name_from_any(value);
  result["type"] = type_name;

  if (value.type() == typeid(int)) {
    assign_value<int>(result, value);
  } else if (value.type() == typeid(long)) {
    assign_value<long>(result, value);
  } else if (value.type() == typeid(long long)) {
    assign_value<long long>(result, value);
  } else if (value.type() == typeid(unsigned int)) {
    assign_value<unsigned int>(result, value);
  } else if (value.type() == typeid(unsigned long)) {
    assign_value<unsigned long>(result, value);
  } else if (value.type() == typeid(unsigned long long)) {
    assign_value<unsigned long long>(result, value);
  } else if (value.type() == typeid(size_t)) {
    assign_value<size_t>(result, value);
  } else if (value.type() == typeid(float)) {
    assign_value<float>(result, value);
  } else if (value.type() == typeid(double)) {
    assign_value<double>(result, value);
  } else if (value.type() == typeid(bool)) {
    assign_value<bool>(result, value);
  } else if (value.type() == typeid(std::string)) {
    assign_value<std::string>(result, value);
  } else if (value.type() == typeid(nlohmann::json)) {
    assign_value<nlohmann::json>(result, value);
  } else if (value.type() == typeid(std::vector<size_t>)) {
    assign_value<std::vector<size_t>>(result, value);
  } else if (value.type() == typeid(std::vector<int>)) {
    assign_value<std::vector<int>>(result, value);
  } else if (value.type() == typeid(std::vector<float>)) {
    assign_value<std::vector<float>>(result, value);
  } else if (value.type() == typeid(std::vector<double>)) {
    assign_value<std::vector<double>>(result, value);
  } else if (value.type() == typeid(std::vector<std::string>)) {
    assign_value<std::vector<std::string>>(result, value);
  }

  return result;
}

std::any TConfig::json_to_value(const nlohmann::json &type_value_pair) {
  if (!type_value_pair.contains("type") || !type_value_pair.contains("value")) {
    throw std::runtime_error("Invalid type-value pair in JSON");
  }

  std::string type_name = type_value_pair["type"];
  const auto &value = type_value_pair["value"];

  if (type_name == "int") {
    return value.get<int>();
  } else if (type_name == "long") {
    return value.get<long>();
  } else if (type_name == "long long") {
    return value.get<long long>();
  } else if (type_name == "unsigned int") {
    return value.get<unsigned int>();
  } else if (type_name == "unsigned long") {
    return value.get<unsigned long>();
  } else if (type_name == "unsigned long long") {
    return value.get<unsigned long long>();
  } else if (type_name == "size_t") {
    return value.get<size_t>();
  } else if (type_name == "float") {
    return value.get<float>();
  } else if (type_name == "double") {
    return value.get<double>();
  } else if (type_name == "bool") {
    return value.get<bool>();
  } else if (type_name == "string") {
    return value.get<std::string>();
  } else if (type_name == "json") {
    return value.get<nlohmann::json>();
  } else if (type_name == "vector<size_t>") {
    return value.get<std::vector<size_t>>();
  } else if (type_name == "vector<int>") {
    return value.get<std::vector<int>>();
  } else if (type_name == "vector<float>") {
    return value.get<std::vector<float>>();
  } else if (type_name == "vector<double>") {
    return value.get<std::vector<double>>();
  } else if (type_name == "vector<string>") {
    return value.get<std::vector<std::string>>();
  }

  throw std::runtime_error("Unknown type in JSON: " + type_name);
}

nlohmann::json TConfig::to_json() const {
  nlohmann::json j;
  j["name"] = name;
  j["type"] = type;

  nlohmann::json param_json = nlohmann::json::object();
  for (const auto &[key, value] : parameters_) {
    param_json[key] = value_to_json(value);
  }
  j["parameters"] = param_json;

  return j;
}

TConfig TConfig::from_json(const nlohmann::json &j) {
  TConfig config;
  config.name = j.value("name", "");
  config.type = j.value("type", "");

  if (j.contains("parameters") && j["parameters"].is_object()) {
    for (const auto &[key, type_value_pair] : j["parameters"].items()) {
      try {
        config.parameters_[key] = json_to_value(type_value_pair);
      } catch (const std::exception &e) {
        std::cerr << "Warning: Failed to deserialize parameter '" << key << "': " << e.what()
                  << std::endl;
      }
    }
  }

  return config;
}

}  // namespace tnn
