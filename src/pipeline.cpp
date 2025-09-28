/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <pipeline/distributed_coordinator.hpp>
#include <pipeline/in_process_coordinator.hpp>
#include <pipeline/network_serialization.hpp>
#include <pipeline/network_stage_worker.hpp>

namespace tpipeline {
/**
 * Template instantiations for commonly used types. Uncomment as needed.
 */
template class DistributedPipelineCoordinator<float>;
// template class DistributedPipelineCoordinator<double>;

template class InProcessPipelineCoordinator<float>;
// template class InProcessPipelineCoordinator<double>;

template class NetworkStageWorker<float>;
// template class NetworkStageWorker<double>;
} // namespace tpipeline