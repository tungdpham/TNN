/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "pipeline/job.hpp"
#include <pipeline/distributed_coordinator.hpp>
#include <pipeline/in_process_coordinator.hpp>
#include <pipeline/network_serialization.hpp>
#include <pipeline/network_stage_worker.hpp>

namespace tnn {
/**
 * Template instantiations for commonly used types. Uncomment as needed.
 */
template struct Job<float>;

} // namespace tnn