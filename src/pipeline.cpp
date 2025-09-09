#include <pipeline/distributed_coordinator.hpp>
#include <pipeline/in_process_coordinator.hpp>
#include <pipeline/network_stage_worker.hpp>
#include <pipeline/network_serialization.hpp>

namespace tpipeline {
  template class DistributedPipelineCoordinator<float>;
  template class DistributedPipelineCoordinator<double>;

  template class InProcessPipelineCoordinator<float>;
  template class InProcessPipelineCoordinator<double>;

  template class NetworkStageWorker<float>;
  template class NetworkStageWorker<double>;
}