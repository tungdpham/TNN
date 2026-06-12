#pragma once

#include <string>

#include "tensor/tensor.hpp"

namespace synet {
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

  const Tensor &data(size_t mb_id) const { return data_.at(mb_id).tensor; }
  int data_ref_count(size_t mb_id) const { return data_.at(mb_id).ref_count; }
  void set_data(size_t mb_id, const Tensor &data, int ref_count) {
    data_[mb_id] = {data, ref_count};
  }
  bool decrement_data_ref_count(size_t mb_id) {
    if (data_.at(mb_id).ref_count > 0) {
      --data_[mb_id].ref_count;
      if (data_[mb_id].ref_count == 0) {
        data_[mb_id].tensor = nullptr;  // Release the tensor when ref count reaches zero
        return true;
      }
    }
    return false;
  }

  const Tensor &grad(size_t mb_id) const { return grad_.at(mb_id).tensor; }
  int grad_ref_count(size_t mb_id) const { return grad_.at(mb_id).ref_count; }
  void set_grad(size_t mb_id, const Tensor &grad, int ref_count) {
    grad_[mb_id] = {grad, ref_count};
  }
  void accumulate_grad(size_t mb_id, const Tensor &grad, int ref_count) {
    if (grad_.count(mb_id) == 0 || grad_[mb_id].tensor == nullptr) {
      grad_[mb_id] = {grad, ref_count};
    } else {
      grad_[mb_id].tensor += grad;  // Accumulate the new gradient
    }
  }
  bool decrement_grad_ref_count(size_t mb_id) {
    if (grad_.at(mb_id).ref_count > 0) {
      --grad_[mb_id].ref_count;
      if (grad_[mb_id].ref_count == 0) {
        grad_[mb_id].tensor = nullptr;
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

}  // namespace synet