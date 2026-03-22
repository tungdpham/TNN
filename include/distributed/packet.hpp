#pragma once

#include <cstdint>

#include "common/endian.hpp"
#include "device/dptr.hpp"

namespace tnn {

enum class CompressionType : uint8_t { NONE = 0, ZSTD = 1, QUANTIZATION = 2 };

enum class PacketType : uint8_t { DATA_FRAGMENT = 0, MSG_PREPARE = 1, MSG_READY_TO_WRITE = 2 };

// Fixed header at the start of each packet.
struct PacketHeader {
  // Packet information
  uint8_t PROTOCOL_VERSION = 1;
  PacketType type = PacketType::DATA_FRAGMENT;  // Type of packet
  Endianness endianess;                         // 1 for little-endian, 0 for big-endian
  uint64_t packet_length = 0;  // Length of the rest of the packet (excluding fixed header part)

  // For fragmentation
  uint64_t msg_length = 0;     // Total length for the entire message.
  uint64_t msg_serial_id = 0;  // Unique ID for the entire message.
  uint32_t packet_offset = 0;  // packet index for fragmentation
  uint32_t total_packets = 1;  // total packets for fragmentation

  CompressionType compression_type = CompressionType::NONE;

  PacketHeader()
      : endianess(host_endianness) {}

  PacketHeader(PacketType t, uint64_t packet_len, uint64_t msg_len, uint32_t pkt_offset,
               uint32_t total_pkts, CompressionType comp_type = CompressionType::NONE)
      : type(t),
        packet_length(packet_len),
        msg_length(msg_len),
        packet_offset(pkt_offset),
        total_packets(total_pkts),
        compression_type(comp_type) {
    endianess = host_endianness;
  }
};

template <typename Archiver>
void archive(Archiver &archiver, const PacketHeader &header) {
  archiver(header.PROTOCOL_VERSION, header.type, header.endianess, header.packet_length,
           header.msg_length, header.msg_serial_id, header.packet_offset, header.total_packets,
           header.compression_type);
}

template <typename Archiver>
void archive(Archiver &archiver, PacketHeader &header) {
  archiver(header.PROTOCOL_VERSION, header.type, header.endianess, header.packet_length,
           header.msg_length, header.msg_serial_id, header.packet_offset, header.total_packets,
           header.compression_type);
}

struct Packet {
  PacketHeader header;
  dptr data;
};

template <typename Archiver>
void archive(Archiver &archiver, const Packet &packet) {
  archiver(packet.header, packet.data);
}

// for deserialization, dptr is dynamically allocated so we can't know where to allocate.

}  // namespace tnn