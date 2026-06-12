#pragma once

namespace synet {

/**
 * @brief Error codes for SYNET library
 */
enum class ec {
  SUCCESS = 0,

  // General errors
  INVALID_ARGUMENT = 1,
  OUT_OF_MEMORY = 2,
  NOT_IMPLEMENTED = 3,
  INTERNAL_ERROR = 4,

  // Graph API errors
  DUPLICATE_NODE_UID = 100,
  NULL_INPUT_DATA = 101,
  NULL_OUTPUT_GRADIENT = 102,
  MISSING_INPUT = 103,
  MISSING_OUTPUT = 104,
  INTERNAL_GRAPH_ERROR = 105,

};

}  // namespace synet