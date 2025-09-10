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
 * If you want to modify its contents, please also update the COUNT to have the
 * highest value, and START to be lowest. Ordering the enum by priority is
 * advised.
 */
enum class CommandType {

  _START,

  FORWARD_TASK,
  BACKWARD_TASK,
  UPDATE_PARAMETERS,

  TRAIN_MODE,
  EVAL_MODE,
  SHUTDOWN,

  HANDSHAKE_REQUEST,
  HANDSHAKE_RESPONSE,
  CONFIG_TRANSFER,
  CONFIG_RECEIVED,
  WEIGHTS_TRANSFER,
  WEIGHTS_RECEIVED,

  STATUS_REQUEST,
  STATUS_RESPONSE,
  PARAMETERS_UPDATED,
  HEALTH_CHECK,

  ERROR_REPORT,
  TASK_FAILURE,

  BARRIER_SYNC,
  CHECKPOINT_REQUEST,
  CHECKPOINT_COMPLETE,

  MEMORY_REPORT,
  RESOURCE_REQUEST,

  QUERY_STAGE_INFO,
  STAGE_INFO_RESPONSE,

  PRINT_PROFILING,
  CLEAR_PROFILING,

  _COUNT
};

} // namespace tpipeline
