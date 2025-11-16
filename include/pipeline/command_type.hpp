/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#pragma once

#include <cstdint>

namespace tnn {

/**
 * @brief Enumeration of all possible command types in the pipeline system.
 * If you want to modify its contents, please also update the COUNT to have the
 * highest value, and START to be lowest. Ordering the enum by priority is
 * advised.
 */
enum CommandType : uint16_t {
  _START,

  // core commands
  FORWARD_JOB,
  BACKWARD_JOB,
  UPDATE_PARAMETERS,

  // mode switching
  TRAIN_MODE,
  EVAL_MODE,
  SHUTDOWN,

  // configuration
  CONFIG_TRANSFER,
  CONFIG_RECEIVED,
  LOAD_PARAMS,
  PARAMS_LOADED,
  SEND_PARAMS,
  PARAMS_TRANSFER,

  // status and monitoring
  STATUS_REQUEST,
  STATUS_RESPONSE,
  PARAMETERS_UPDATED,
  HEALTH_CHECK,

  // error handling
  ERROR_REPORT,
  JOB_FAILURE,

  // advanced features
  BARRIER_SYNC,
  CHECKPOINT_REQUEST,
  CHECKPOINT_COMPLETE,

  // load balancing and resource management
  UPDATE_LOAD,
  REPORT_LOAD,
  LOAD_REPORT,

  // profiling and debugging
  PRINT_PROFILING,
  PROFILING_PRINTED,
  CLEAR_PROFILING,
  PROFILING_CLEARED,

  _COUNT
};

} // namespace tnn
