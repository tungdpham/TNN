/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#pragma once

namespace tpipeline {

/**
 * @brief Enumeration of all possible command types in the pipeline system. 
 * If you want to modify its contents, please also update the COUNT to have the highest value, and START to be lowest. 
 * Ordering the enum by priority is advised.
 */
enum class CommandType {
  // START (DO NOT REMOVE)
  _START,

  // Pipeline Tasks
  FORWARD_TASK,
  BACKWARD_TASK,
  UPDATE_PARAMETERS,

  // Control Commands
  TRAIN_MODE,
  EVAL_MODE,
  SHUTDOWN,

  // Network Management
  HANDSHAKE_REQUEST,
  HANDSHAKE_RESPONSE,
  CONFIG_TRANSFER,
  CONFIG_RECEIVED,
  WEIGHTS_TRANSFER,
  WEIGHTS_RECEIVED,

  // System State
  STATUS_REQUEST,
  STATUS_RESPONSE,
  PARAMETERS_UPDATED,
  HEALTH_CHECK,

  // Error Handling
  ERROR_REPORT,
  TASK_FAILURE,

  // Synchronization
  BARRIER_SYNC,
  CHECKPOINT_REQUEST,
  CHECKPOINT_COMPLETE,

  // Resource Management
  MEMORY_REPORT,
  RESOURCE_REQUEST,

  // Stage Management
  QUERY_STAGE_INFO,
  STAGE_INFO_RESPONSE,

  // Debugging and Profiling
  PRINT_PROFILING,
  CLEAR_PROFILING,

  // IMPORTANT: ALWAYS LAST
  _COUNT // track the number of command types
};

} // namespace tpipeline
