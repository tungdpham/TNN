/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#ifdef USE_TBB
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tthreads {
inline uint32_t get_num_threads() {
#ifdef _OPENMP
  return static_cast<uint32_t>(omp_get_max_threads());
#elif defined(USE_TBB)
  return static_cast<uint32_t>(tbb::this_task_arena::max_concurrency());
#else
  return 1;
#endif
}

template <typename Index = size_t, typename Func>
static inline void parallel_for(const Index begin, const Index end, Func f) {
  assert(end >= begin && "Invalid range");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
  for (Index i = begin; i < end; ++i) {
    f(i);
  }
#elif defined(USE_TBB)
  tbb::parallel_for(
      tbb::blocked_range<Index>(begin, end),
      [&](const tbb::blocked_range<Index> &r) {
        for (Index i = r.begin(); i != r.end(); ++i)
          f(i);
      },
      tbb::static_partitioner());
#else
  for (Index i = begin; i < end; ++i)
    f(i);
#endif
}

template <typename Index = size_t, typename Func>
inline void parallel_for_2d(const Index dim0, const Index dim1, Func f) {
  assert(dim0 >= 0 && dim1 >= 0 && "Invalid dimensions");
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
  for (Index i = 0; i < dim0; ++i) {
    for (Index j = 0; j < dim1; ++j) {
      f(i, j);
    }
  }
#elif defined(USE_TBB)
  tbb::parallel_for(
      tbb::blocked_range2d<Index>(0, dim0, 0, dim1),
      [&](const tbb::blocked_range2d<Index> &r) {
        for (Index i = r.rows().begin(); i != r.rows().end(); ++i) {
          for (Index j = r.cols().begin(); j != r.cols().end(); ++j) {
            f(i, j);
          }
        }
      },
      tbb::static_partitioner());
#else
  std::cout << "Warning: Running parallel_for_2d in serial mode.\n";
  for (Index i = 0; i < dim0; ++i) {
    for (Index j = 0; j < dim1; ++j) {
      f(i, j);
    }
  }
#endif
}

} // namespace tthreads