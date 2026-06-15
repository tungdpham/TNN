#pragma once

#include "nn/layer.hpp"
#include "nn/node.hpp"

namespace synet {

class EdgeImpl {
public:
  EdgeImpl(std::shared_ptr<LayerImpl> layer, const Vec<Node> &inputs, const Vec<Node> &outputs)
      : layer_(layer),
        producers_(inputs),
        consumers_(outputs) {}

  EdgeImpl(std::shared_ptr<LayerImpl> layer, std::initializer_list<Node> inputs,
           std::initializer_list<Node> outputs)
      : layer_(layer),
        producers_(inputs),
        consumers_(outputs) {}

  const Vec<Node> &producers() const { return producers_; }
  const Vec<Node> &consumers() const { return consumers_; }
  std::shared_ptr<LayerImpl> layer() const { return layer_; }

  Residuals &residuals(size_t mb_id) {
    auto it = residuals_cache_.find(mb_id);
    if (it != residuals_cache_.end()) {
      return it->second;
    }
    throw std::runtime_error("Residuals not found for the given minibatch ID");
  }
  void set_residuals(size_t mb_id, Residuals residuals) {
    residuals_cache_[mb_id] = std::move(residuals);
  }

  void clear_residuals(size_t mb_id) { residuals_cache_.erase(mb_id); }

private:
  std::shared_ptr<LayerImpl> layer_;
  Vec<Node> producers_;
  Vec<Node> consumers_;
  std::unordered_map<size_t, Residuals> residuals_cache_;  // mb_id -> residuals
};

using Edge = std::shared_ptr<EdgeImpl>;

}  // namespace synet