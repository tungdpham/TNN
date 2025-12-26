/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "distributed/in_process_coordinator.hpp"
#include "distributed/job.hpp"
#include "distributed/network_coordinator.hpp"
#include "distributed/network_serialization.hpp"
#include "distributed/network_worker.hpp"

namespace tnn {
/**
 * Template instantiations for commonly used types. Uncomment as needed.
 */
template struct Job<float>;

} // namespace tnn