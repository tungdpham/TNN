#pragma once

#if defined(USE_TBB)
#include <tbb/scalable_allocator.h>
#include <tbb/task_arena.h>
#endif

#ifdef _OPENMP
#include "omp.h"
#endif

namespace tthreads {

struct ThreadingConfig {
  unsigned int num_threads = 1;
};

class ThreadWrapper {
public:
  explicit ThreadWrapper(const ThreadingConfig &config)
      : config_(config), arena_(config.num_threads) {}

  template <typename Function, typename... Args> void execute(Function &&f, Args &&...args) {
#ifdef USE_TBB
    arena_.execute([&]() { f(std::forward<Args>(args)...); });
#elif defined(_OPENMP)
    f(std::forward<Args>(args)...);
  }
#else
    f(std::forward<Args>(args)...);
#endif
  }

  inline void clean_buffers() {
#ifdef USE_TBB
    scalable_allocation_command(TBBMALLOC_CLEAN_ALL_BUFFERS, 0);
#endif
  }

  const ThreadingConfig &get_config() const { return config_; }

private:
  ThreadingConfig config_;
#ifdef USE_TBB
  tbb::task_arena arena_;
#endif
};

} // namespace tthreads
