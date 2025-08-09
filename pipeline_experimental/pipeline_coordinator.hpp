template <typename T = float> class PipelineCoordinator {
public:
  PipelineCoordinator() = default;

  // Add a stage to the pipeline
  void add_stage(std::unique_ptr<PipelineStage<T>> stage) {
    stages_.push_back(std::move(stage));
  }

  // Start all stages
  void start() {
    for (auto &stage : stages_) {
      stage->start();
    }
  }

  // Stop all stages
  void stop() {
    for (auto &stage : stages_) {
      stage->stop();
    }
  }

  // Process a batch by splitting it into microbatches and sending to first stage
  void process_batch(const Tensor<T> &batch, int num_microbatches = 4) {
    if (stages_.empty()) return;

    // Split batch into microbatches
    std::vector<Tensor<T>> microbatches = split_into_microbatches(batch, num_microbatches);

    // Send each microbatch to the first stage
    auto *first_stage_comm = stages_[0]->get_communicator();
    for (int i = 0; i < microbatches.size(); ++i) {
      tpipeline::Task<T> task(tpipeline::TaskType::Forward, microbatches[i], i);
      first_stage_comm->enqueue_task(task);
    }
  }

protected:
  std::vector<std::unique_ptr<PipelineStage<T>>> stages_;

  // Helper function to split batch into microbatches
  std::vector<Tensor<T>> split_into_microbatches(const Tensor<T> &batch, int num_microbatches) {
    std::vector<Tensor<T>> microbatches;
    size_t batch_size = batch.batch_size() / num_microbatches;

    for (int i = 0; i < num_microbatches; ++i) {
      size_t start = i * batch_size;
      size_t end = (i == num_microbatches - 1) ? batch.batch_size() : start + batch_size;
      microbatches.push_back(batch.slice_batch(start, end));
    }

    return microbatches;
  }
};