#pragma once

#include <sys/types.h>

#include <cstdint>
#include <iostream>
#include <mutex>
#include <unordered_map>

#include "distributed/buffer_pool.hpp"
#include "packet.hpp"

namespace tnn {
struct MessageState {
  uint32_t received_packets = 0;
  uint32_t total_packets = 0;
  PooledBuffer buffer;
};

class Fragmenter {
public:
  Fragmenter() = default;
  ~Fragmenter() = default;
  Fragmenter(const Fragmenter &) = delete;
  Fragmenter &operator=(const Fragmenter &) = delete;

  Fragmenter(Fragmenter &&other) {
    std::lock_guard<std::mutex> msg_lock(other.message_states_mutex_);
    message_states_ = std::move(other.message_states_);
  }
  Fragmenter &operator=(Fragmenter &&other) {
    if (this != &other) {
      std::lock_guard<std::mutex> msg_lock(other.message_states_mutex_);
      message_states_ = std::move(other.message_states_);
    }
    return *this;
  }

  template <typename BufferType>
  std::vector<PacketHeader> get_headers(BufferType &buffer, uint32_t num_packets) {
    size_t serial_id = cur_serial_id_.fetch_add(1);
    if (num_packets == 0) {
      throw std::invalid_argument("Number of packets must be greater than 0");
    }

    std::vector<PacketHeader> headers;
    size_t packet_size = (buffer.size() + num_packets - 1) / num_packets;
    for (uint32_t i = 0; i < num_packets; ++i) {
      PacketHeader header;
      header.length = i == num_packets - 1 ? buffer.size() - i * packet_size : packet_size;
      header.msg_length = buffer.size();
      header.packet_offset = i * packet_size;
      header.msg_serial_id = serial_id;
      header.total_packets = num_packets;
      headers.emplace_back(std::move(header));
    }
    return headers;
  }

  bool message_exists(uint64_t msg_serial_id) const {
    std::lock_guard<std::mutex> lock(message_states_mutex_);
    return message_states_.find(msg_serial_id) != message_states_.end();
  }

  // initiate the packet storage
  void register_packet(uint64_t msg_serial_id, const PacketHeader &header) {
    std::lock_guard<std::mutex> lock(message_states_mutex_);
    auto it = message_states_.find(msg_serial_id);
    if (it == message_states_.end()) {
      MessageState &state = message_states_[msg_serial_id];
      state.total_packets = header.total_packets;
      state.buffer = BufferPool::instance().get_buffer(header.msg_length);
      state.buffer->resize(header.msg_length);
      return;
    }
    it->second.total_packets = header.total_packets;
    if (it->second.buffer->capacity() < header.msg_length) {
      throw std::runtime_error(
          "Packet length mistmatch: " + std::to_string(header.msg_length) +
          ", buffer capacity: " + std::to_string(it->second.buffer->capacity()));
    }
  }

  /**
   * @brief Commit a received packet and return whether the entire message is complete.
   * @note This ensure the add fetch transaction is atomic.
   * @param msg_serial_id The serial ID of the message.
   * @param header The header of the received packet.
   * @return true if the entire message has been received; false otherwise.
   */
  bool commit_packet(uint64_t msg_serial_id, const PacketHeader &header) {
    std::lock_guard<std::mutex> lock(message_states_mutex_);
    auto it = message_states_.find(msg_serial_id);
    if (it == message_states_.end()) {
      std::cerr << "Error committing packet: Message ID " << msg_serial_id << " not found."
                << std::endl;
      return false;
    }
    MessageState &state = it->second;
    state.received_packets++;
    return state.received_packets == state.total_packets;
  }

  PooledBuffer get_packet_buffer(uint64_t msg_serial_id, const PacketHeader &header) {
    std::lock_guard<std::mutex> lock(message_states_mutex_);
    return message_states_[msg_serial_id].buffer;
  }

  MessageState fetch_complete_message(uint64_t msg_serial_id) {
    std::lock_guard<std::mutex> lock(message_states_mutex_);
    auto it = message_states_.find(msg_serial_id);
    if (it == message_states_.end()) {
      std::cerr << "Error fetching complete message: Message ID " << msg_serial_id << " not found."
                << std::endl;
    }
    MessageState state = std::move(it->second);
    if (state.received_packets < state.total_packets) {
      std::cerr << "Error fetching complete message: Message ID " << msg_serial_id
                << " is not complete." << std::endl;
    }
    message_states_.erase(it);
    return state;
  }

private:
  static inline std::atomic<uint64_t> cur_serial_id_{101};     // auto increment, for sender side
  mutable std::mutex message_states_mutex_;                    // protect message_states_
  std::unordered_map<uint64_t, MessageState> message_states_;  // for receiver side
};
}  // namespace tnn