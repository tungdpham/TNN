#pragma once

#include <cstring>
#include <mutex>

#include "distributed/peer_context.hpp"

namespace tnn {

class Channel {
public:
  explicit Channel() {}

  virtual ~Channel() = default;

  virtual void close() = 0;

  PeerContext context() {
    std::lock_guard<std::mutex> lock(context_mutex_);
    if (!context_) {
      std::cerr << "Err: No peer context set for this channel" << std::endl;
      return nullptr;
    }
    return context_;
  }

  void set_context(PeerContext context) {
    std::lock_guard<std::mutex> lock(context_mutex_);
    context_ = context;
  }

private:
  PeerContext context_;
  std::mutex context_mutex_;
};

}  // namespace tnn