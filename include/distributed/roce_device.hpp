#pragma once

#include <infiniband/verbs.h>

#include <iostream>
#include <stdexcept>
#include <string>

constexpr int ROCE_SQ_DEPTH = 32;
constexpr int ROCE_RQ_DEPTH = 32;

namespace tnn {
class RoCEDevice {
public:
  RoCEDevice(const std::string &device_name, int ib_port, int gid_index_override = -1)
      : device_name_(device_name),
        ib_port_(ib_port) {
    int num_devices;
    ibv_device **device_list = ibv_get_device_list(&num_devices);
    if (!device_list) {
      throw std::runtime_error("Failed to get RDMA devices");
    }
    bool found = false;
    for (int i = 0; i < num_devices; ++i) {
      if (device_name == ibv_get_device_name(device_list[i]) || device_name.empty()) {
        context_ = ibv_open_device(device_list[i]);
        found = true;
        break;
      }
    }
    ibv_free_device_list(device_list);
    if (!found) {
      throw std::runtime_error("RDMA device not found: " + device_name);
    }
    pd_ = ibv_alloc_pd(context_);
    if (!pd_) {
      ibv_close_device(context_);
      throw std::runtime_error("Failed to allocate protection domain");
    }
    gid_index_ = (gid_index_override == -1) ? auto_select_gid() : gid_index_override;
  }

  ~RoCEDevice() {
    if (pd_) ibv_dealloc_pd(pd_);
    if (context_) ibv_close_device(context_);
  }

  RoCEDevice(const RoCEDevice &) = delete;
  RoCEDevice &operator=(const RoCEDevice &) = delete;

  RoCEDevice(RoCEDevice &&other) noexcept
      : device_name_(std::move(other.device_name_)),
        ib_port_(other.ib_port_),
        gid_index_(other.gid_index_),
        context_(other.context_),
        pd_(other.pd_) {
    other.context_ = nullptr;
    other.pd_ = nullptr;
  }

  RoCEDevice &operator=(RoCEDevice &&other) noexcept {
    if (this != &other) {
      if (pd_) ibv_dealloc_pd(pd_);
      if (context_) ibv_close_device(context_);
      device_name_ = std::move(other.device_name_);
      ib_port_ = other.ib_port_;
      gid_index_ = other.gid_index_;
      context_ = other.context_;
      pd_ = other.pd_;
      other.context_ = nullptr;
      other.pd_ = nullptr;
    }
    return *this;
  }

  ibv_context *get_context() const { return context_; }
  ibv_pd *get_pd() const { return pd_; }
  const std::string &device_name() const { return device_name_; }
  int get_port() const { return ib_port_; }
  int get_gid_index() const { return gid_index_; }

  void print_gid_table() const {
    struct ibv_port_attr port_attr;
    if (ibv_query_port(context_, ib_port_, &port_attr) == 0) {
      std::cout << "[RoCE] GID Table for device " << device_name_ << ":\n";
      for (int i = 0; i < port_attr.gid_tbl_len; ++i) {
        union ibv_gid gid;
        if (ibv_query_gid(context_, ib_port_, i, &gid) == 0) {
          bool empty = true;
          for (int b = 0; b < 16; ++b)
            if (gid.raw[b] != 0) empty = false;
          if (empty) continue;
          std::cout << "  GID Index " << i << ": ";
          auto old_flags = std::cout.flags();
          for (int b = 0; b < 16; ++b) std::cout << (int)gid.raw[b] << (b < 15 ? ":" : "");
          std::cout.flags(old_flags);
          std::cout << "\n";
        }
      }
    }
  }

private:
  int auto_select_gid() {
    struct ibv_port_attr port_attr;
    int best_gid_index = -1;
    bool found_ipv4 = false;

    if (ibv_query_port(context_, ib_port_, &port_attr) == 0) {
      for (int i = 0; i < port_attr.gid_tbl_len; ++i) {
        union ibv_gid gid;
        if (ibv_query_gid(context_, ib_port_, i, &gid) == 0) {
          bool empty = true;
          for (int b = 0; b < 16; ++b) {
            if (gid.raw[b] != 0) {
              empty = false;
              break;
            }
          }
          if (empty) continue;
          bool is_ipv4 = true;
          for (int b = 0; b < 10; ++b) {
            if (gid.raw[b] != 0) {
              is_ipv4 = false;
              break;
            }
          }
          if (is_ipv4 && gid.raw[10] == 0xff && gid.raw[11] == 0xff) {
            best_gid_index = i;
            found_ipv4 = true;
            break;
          }
          if (best_gid_index == -1) {
            best_gid_index = i;
          }
        }
      }
    }
    if (best_gid_index == -1) {
      throw std::runtime_error("Auto-selection of GID Index failed: No valid GID found on device " +
                               device_name_);
    }
    std::cout << "[RoCE] Auto-selected GID Index: " << best_gid_index
              << (found_ipv4 ? " (IPv4/RoCEv2)" : "") << "\n";
    return best_gid_index;
  }

  std::string device_name_;
  int ib_port_ = 0;
  int gid_index_ = -1;
  ibv_context *context_ = nullptr;
  ibv_pd *pd_ = nullptr;
};
}  // namespace tnn