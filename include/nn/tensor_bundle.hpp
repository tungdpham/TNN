#pragma once

#include <map>
#include <string>

#include "nn/node.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

class TensorBundle {
private:
  std::map<std::string, Tensor> inputs_;  // Map from uid -> Tensor

public:
  TensorBundle() = default;

  TensorBundle(std::initializer_list<std::pair<const std::string, Tensor>> init)
      : inputs_(init) {}

  TensorBundle(std::initializer_list<std::pair<Node, Tensor>> init) {
    for (const auto &pair : init) {
      inputs_[pair.first->uid()] = pair.second;
    }
  }

  auto begin() noexcept { return inputs_.begin(); }
  auto end() noexcept { return inputs_.end(); }

  auto begin() const noexcept { return inputs_.begin(); }
  auto end() const noexcept { return inputs_.end(); }
  auto cbegin() const noexcept { return inputs_.cbegin(); }
  auto cend() const noexcept { return inputs_.cend(); }

  Tensor &operator[](const std::string &name) { return inputs_[name]; }
  const Tensor &operator[](const std::string &name) const { return inputs_.at(name); }

  void set(const std::string &name, const Tensor &tensor) { inputs_[name] = tensor; }
  const Tensor &get(const std::string &name) const { return inputs_.at(name); }
  bool contains(const std::string &name) const { return inputs_.count(name) > 0; }
  size_t size() const { return inputs_.size(); }
  void clear() { inputs_.clear(); }
};

template <typename Archiver>
void archive(Archiver &archiver, const TensorBundle &bundle) {
  uint64_t bundle_size = static_cast<uint64_t>(bundle.size());
  archiver(bundle_size);
  for (const auto &entry : bundle) {
    archiver(entry.first);   // Serialize the UID
    archiver(entry.second);  // Serialize the Tensor
  }
}
}  // namespace tunx