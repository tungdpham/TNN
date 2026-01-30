/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "distributed/binary_serializer.hpp"
#include "distributed/job.hpp"
#include "distributed/tcp_coordinator.hpp"
#include "distributed/tcp_worker.hpp"

namespace tnn {

// Static member variable definition
bool BinarySerializer::deserialize_to_gpu_ = false;

} // namespace tnn