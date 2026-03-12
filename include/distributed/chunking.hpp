#pragma once

#include <sys/types.h>

#include <cstdint>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "device/dptr.hpp"
#include "device/iallocator.hpp"
#include "distributed/ibuffer.hpp"
#include "packet.hpp"

namespace tnn {
class ISlicer {
public:
  // Slices a buffer into multiple packets according to the implemented chunking strategy.
  virtual std::vector<Packet> slice(IBuffer &&buffer) = 0;

protected:
  uint64_t get_id() { return current_id_.fetch_add(1); }

private:
  std::atomic<uint64_t> current_id_{101};
};

class IAggregator {
public:
  // Registers a received packet and returns a pointer to the buffer where the packet data should be
  // written. The buffer is guaranteed to be large enough to hold the packet data at the specified
  // offset.
  virtual dptr fetch_packet(const PacketHeader &header) = 0;

  // Commits a received packet. Returns true if this packet completes the message and the message is
  // ready to be processed, or false if more packets are still expected.
  virtual bool commit_packet(const PacketHeader &header) = 0;

  // Finalizes the message and returns a pointer to the complete message buffer.
  virtual dptr finalize(const PacketHeader &header) = 0;
};

using Slicer = std::unique_ptr<ISlicer>;
using Aggregator = std::unique_ptr<IAggregator>;

class BlockSlicer : public ISlicer {
public:
  BlockSlicer(uint64_t block_size)
      : block_size_(block_size) {}

  std::vector<Packet> slice(IBuffer &&buffer) override {
    uint64_t msg_id = get_id();
    uint64_t total_size = buffer.size();
    uint32_t num_packets = static_cast<uint32_t>((total_size + block_size_ - 1) / block_size_);
    std::vector<Packet> packets;
    for (uint32_t i = 0; i < num_packets; ++i) {
      PacketHeader header;
      header.packet_length = i == num_packets - 1 ? total_size - i * block_size_ : block_size_;
      header.msg_length = total_size;
      header.packet_offset = i * block_size_;
      header.msg_serial_id = msg_id;
      header.total_packets = num_packets;
      packets.push_back(Packet{.header = std::move(header),
                               .data = buffer.span(header.packet_offset, header.packet_length)});
    }
    return packets;
  }

private:
  uint64_t block_size_;
};

class NWaySlicer : public ISlicer {
public:
  NWaySlicer(uint64_t num_ways)
      : num_ways_(num_ways) {}

  std::vector<Packet> slice(IBuffer &&buffer) override {
    uint64_t msg_id = this->get_id();
    uint64_t total_size = buffer.size();
    uint32_t num_packets = static_cast<uint32_t>(num_ways_);
    std::vector<Packet> packets;
    uint64_t packet_size = (total_size + num_ways_ - 1) / num_ways_;
    for (uint32_t i = 0; i < num_packets; ++i) {
      PacketHeader header;
      header.packet_length = i == num_packets - 1 ? total_size - i * packet_size : packet_size;
      header.msg_length = total_size;
      header.packet_offset = i * packet_size;
      header.msg_serial_id = msg_id;
      header.total_packets = num_packets;
      packets.push_back(Packet{.header = std::move(header),
                               .data = buffer.span(header.packet_offset, header.packet_length)});
    }
    return packets;
  }

private:
  uint64_t num_ways_;
};

// A simple aggregator that doesn't consider compression
class RawAggregator : public IAggregator {
public:
  RawAggregator(IAllocator &allocator)
      : allocator_(allocator) {}

  dptr fetch_packet(const PacketHeader &header) override {
    std::lock_guard<std::mutex> lock(message_states_mutex_);
    auto it = message_states_.find(header.msg_serial_id);
    // If this is the first packet for this message, allocate a buffer for the entire message
    if (it == message_states_.end()) {
      auto buffer = allocator_.allocate(header.msg_length);
      message_states_[header.msg_serial_id] = MessageState{
          .total_packets = header.total_packets,
          .received_packets = 0,
          .buffer = std::move(buffer),
      };
    }
    auto &state = message_states_[header.msg_serial_id];
    return state.buffer.span(header.packet_offset, header.packet_length);
  }

  bool commit_packet(const PacketHeader &header) override {
    std::lock_guard<std::mutex> lock(message_states_mutex_);
    auto it = message_states_.find(header.msg_serial_id);
    if (it == message_states_.end()) {
      std::cerr << "Error committing packet" << std::endl;
      return false;
    }
    MessageState &state = it->second;
    state.received_packets++;
    return state.received_packets == state.total_packets;
  }

  dptr finalize(const PacketHeader &header) override {
    std::lock_guard<std::mutex> lock(message_states_mutex_);
    auto it = message_states_.find(header.msg_serial_id);
    if (it == message_states_.end()) {
      std::cerr << "Error during finalization" << std::endl;
      return dptr();
    }
    if (it->second.received_packets != it->second.total_packets) {
      std::cerr << "Error: message finalized before all packets are received" << std::endl;
      return dptr();
    }
    dptr data = std::move(it->second.buffer);
    message_states_.erase(it);
    return data;
  }

private:
  struct MessageState {
    uint64_t total_packets = 0;
    uint64_t received_packets = 0;
    dptr buffer;
  };
  IAllocator &allocator_;                                      // allocator for message buffers
  mutable std::mutex message_states_mutex_;                    // protect message_states_
  std::unordered_map<uint64_t, MessageState> message_states_;  // for receiver side
};

}  // namespace tnn