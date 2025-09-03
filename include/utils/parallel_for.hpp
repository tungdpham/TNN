/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#pragma once

#include <cstddef>
#include <functional>

#ifdef USE_TBB
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/parallel_for.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace utils {

template <typename Index = size_t, typename Func>
inline void parallel_for_range(const Index begin, const Index end, Func f) {
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
  for (Index i = 0; i < dim0; ++i) {
    for (Index j = 0; j < dim1; ++j) {
      f(i, j);
    }
  }
#endif
}

} // namespace utils
