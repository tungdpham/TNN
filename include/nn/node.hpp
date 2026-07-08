#pragma once

#include <map>
#include <memory>
#include <string>

#include "tensor/tensor.hpp"

namespace tunx {
class Graph;

class NodeImpl {
public:
  explicit NodeImpl(Graph *graph, const std::string &uid = "")
      : graph_(graph),
        uid_(uid) {}

  ~NodeImpl() = default;

  Graph *graph() const { return graph_; }

  const std::string &uid() const { return uid_; }
  void set_uid(const std::string &uid) { uid_ = uid; }

  const Tensor &data(size_t pid) const { return data_.at(pid).tensor; }
  int data_ref_count(size_t pid) const { return data_.at(pid).ref_count; }
  void set_data(size_t pid, const Tensor &data, int ref_count) { data_[pid] = {data, ref_count}; }
  void clear_data(size_t pid) { data_.erase(pid); }
  bool decrement_data_ref_count(size_t pid) {
    auto it = data_.find(pid);
    if (it != data_.end() && it->second.ref_count > 0) {
      --it->second.ref_count;
      if (it->second.ref_count == 0) {
        it->second.tensor =
            Tensor();  // Clear tensor data but keep the entry to avoid iterator invalidation
        return true;
      }
    }
    return false;
  }

  const Tensor &grad(size_t pid) const { return grad_.at(pid).tensor; }
  int grad_ref_count(size_t pid) const { return grad_.at(pid).ref_count; }
  void set_grad(size_t pid, const Tensor &grad, int ref_count) { grad_[pid] = {grad, ref_count}; }
  void clear_grad(size_t pid) { grad_[pid].tensor = Tensor(); }
  void accumulate_grad(size_t pid, const Tensor &grad, int ref_count) {
    if (grad_.count(pid) == 0) {
      grad_[pid] = {grad, ref_count};
    } else {
      grad_[pid].tensor += grad;  // Accumulate the new gradient
    }
  }
  bool decrement_grad_ref_count(size_t pid) {
    auto it = grad_.find(pid);
    if (it != grad_.end() && it->second.ref_count > 0) {
      --it->second.ref_count;
      if (it->second.ref_count == 0) {
        it->second.tensor =
            Tensor();  // Clear tensor data but keep the entry to avoid iterator invalidation
        return true;
      }
    }
    return false;
  }
  void zero_grads() { grad_.clear(); }

private:
  struct Entry {
    Tensor tensor;
    int ref_count = 0;
  };
  Graph *graph_;
  std::string uid_;
  std::map<size_t, Entry> data_;
  std::map<size_t, Entry> grad_;
};

using Node = std::shared_ptr<NodeImpl>;

}  // namespace tunx