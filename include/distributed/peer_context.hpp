#pragma once

#include <mutex>

#include "chunking.hpp"
#include "distributed/endpoint.hpp"

namespace tnn {

// Represents the context for a peer endpoint.
class IPeerContext {
public:
  IPeerContext(Endpoint endpoint, Aggregator &&aggregator, Slicer &&slicer)
      : endpoint_(std::move(endpoint)),
        aggregator_(std::move(aggregator)),
        slicer_(std::move(slicer)) {}

  ~IPeerContext() = default;

  Endpoint endpoint() const { return endpoint_; }

  std::vector<Packet> slice(IBuffer &&buffer) {
    std::unique_lock<std::mutex> lock(mutex_);
    return slicer_->slice(std::move(buffer));
  }

  dptr fetch_packet(const PacketHeader &header) {
    std::unique_lock<std::mutex> lock(mutex_);
    return aggregator_->fetch_packet(header);
  }

  bool commit_packet(const PacketHeader &header) {
    std::unique_lock<std::mutex> lock(mutex_);
    return aggregator_->commit_packet(header);
  }

  dptr finalize(const PacketHeader &header) {
    std::unique_lock<std::mutex> lock(mutex_);
    return aggregator_->finalize(header);
  }

protected:
  Endpoint endpoint_;
  Aggregator aggregator_;
  Slicer slicer_;
  std::mutex mutex_;
};

using PeerContext = std::shared_ptr<IPeerContext>;

inline PeerContext make_peer_context(Endpoint endpoint, Slicer &&slicer, Aggregator &&aggregator) {
  return std::make_shared<IPeerContext>(std::move(endpoint), std::move(aggregator),
                                        std::move(slicer));
}

}  // namespace tnn