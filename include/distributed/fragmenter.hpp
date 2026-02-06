#pragma once

#include <sys/types.h>

#include <cstdint>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "device/dptr.hpp"
#include "device/iallocator.hpp"
#include "distributed/tbuffer.hpp"
#include "packet.hpp"

namespace tnn {
class Fragmenter {
public:
  Fragmenter(IAllocator &allocator)
      : allocator_(allocator) {}
  ~Fragmenter() = default;

  Fragmenter(const Fragmenter &) = delete;
  Fragmenter &operator=(const Fragmenter &) = delete;

  Fragmenter(Fragmenter &&other)
      : allocator_(other.allocator_) {
    std::lock_guard<std::mutex> msg_lock(other.message_states_mutex_);
    message_states_ = std::move(other.message_states_);
  }

  Fragmenter &operator=(Fragmenter &&other) = delete;

  std::vector<Packet> split(TBuffer &&buffer, uint32_t num_packets) {
    size_t serial_id = cur_serial_id_.fetch_add(1);
    if (num_packets == 0) {
      throw std::invalid_argument("Number of packets must be greater than 0");
    }

    std::vector<Packet> packets;
    size_t packet_size = (buffer.size() + num_packets - 1) / num_packets;
    for (uint32_t i = 0; i < num_packets; ++i) {
      PacketHeader header;
      header.packet_length = i == num_packets - 1 ? buffer.size() - i * packet_size : packet_size;
      header.msg_length = buffer.size();
      header.packet_offset = i * packet_size;
      header.msg_serial_id = serial_id;
      header.total_packets = num_packets;
      packets.emplace_back(Packet{
          .header = std::move(header),
          .data = buffer.span(header.packet_offset, header.packet_length),
      });
    }
    return packets;
  }

  // initiate the packet storage
  void register_packet(const PacketHeader &header) {
    std::lock_guard<std::mutex> lock(message_states_mutex_);
    auto it = message_states_.find(header.msg_serial_id);
    if (it == message_states_.end()) {
      auto ptr = allocator_.allocate(header.msg_length);
      auto buffer = TBuffer(std::move(ptr));
      buffer.resize(header.msg_length);
      message_states_[header.msg_serial_id] = MessageState{
          .total_packets = header.total_packets,
          .received_packets = 0,
          .buffer = std::move(buffer),
      };
      return;
    }
  }

  /**
   * @brief Commit a received packet and return whether the entire message is complete.
   * @note This ensure the add fetch transaction is atomic.
   * @param msg_serial_id The serial ID of the message.
   * @param header The header of the received packet.
   * @return true if the entire message has been received; false otherwise.
   */
  bool commit_packet(const PacketHeader &header) {
    std::lock_guard<std::mutex> lock(message_states_mutex_);
    auto it = message_states_.find(header.msg_serial_id);
    if (it == message_states_.end()) {
      std::cerr << "Error committing packet: Message ID " << header.msg_serial_id << " not found."
                << std::endl;
      return false;
    }
    MessageState &state = it->second;
    state.received_packets++;
    return state.received_packets == state.total_packets;
  }

  dptr get_packet_buffer(const PacketHeader &header) {
    std::lock_guard<std::mutex> lock(message_states_mutex_);
    auto &state = message_states_[header.msg_serial_id];
    return state.buffer.span(header.packet_offset, header.packet_length);
  }

  TBuffer fetch_complete_message(const PacketHeader &header) {
    std::lock_guard<std::mutex> lock(message_states_mutex_);
    auto it = message_states_.find(header.msg_serial_id);
    if (it == message_states_.end()) {
      std::cerr << "Error fetching complete message: Message ID " << header.msg_serial_id
                << " not found." << std::endl;
    }
    MessageState state = std::move(it->second);
    if (state.received_packets < state.total_packets) {
      std::cerr << "Error fetching complete message: Message ID " << header.msg_serial_id
                << " is not complete." << std::endl;
    }
    message_states_.erase(it);
    return std::move(state.buffer);
  }

private:
  struct MessageState {
    uint32_t total_packets = 0;
    uint32_t received_packets = 0;
    TBuffer buffer;
  };

  IAllocator &allocator_;                                      // allocator for message buffers
  static inline std::atomic<uint64_t> cur_serial_id_{101};     // auto increment, for sender side
  mutable std::mutex message_states_mutex_;                    // protect message_states_
  std::unordered_map<uint64_t, MessageState> message_states_;  // for receiver side
};
}  // namespace tnn