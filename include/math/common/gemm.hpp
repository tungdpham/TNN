#pragma once

#include <cstddef>

namespace tnn {
struct GemmStats {
  size_t M = 0;
  size_t N = 0;
  size_t K = 0;
  size_t batch_count = 1;
  size_t fwd_workspace_size = 0;
  size_t dgrad_workspace_size = 0;
  size_t wgrad_workspace_size = 0;
};

inline void init_gemm_stats(GemmStats &stats, size_t M, size_t N, size_t K,
                            size_t batch_count = 1) {
  stats.M = M;
  stats.N = N;
  stats.K = K;
  stats.batch_count = batch_count;
  stats.fwd_workspace_size = 0;
  stats.dgrad_workspace_size = 0;
  stats.wgrad_workspace_size = 0;
}

}  // namespace tnn