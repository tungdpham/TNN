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

namespace utils {

template <typename Index = size_t, typename Func>
inline void parallel_for_range(Index begin, Index end, Func f) {
#ifdef USE_TBB
  if (end > begin) {
    tbb::parallel_for(tbb::blocked_range<Index>(begin, end),
                      [&](const tbb::blocked_range<Index> &r) {
                        for (Index i = r.begin(); i != r.end(); ++i)
                          f(i);
                      }, tbb::static_partitioner());
  }
#else
  for (Index i = begin; i < end; ++i)
    f(i);
#endif
}

template <typename Index = size_t, typename Func>
inline void parallel_for_2d(Index dim0, Index dim1, Func f) {
#ifdef USE_TBB
  if (dim0 > 0 && dim1 > 0) {
    tbb::parallel_for(
        tbb::blocked_range2d<Index>(0, dim0, 0, dim1),
        [&](const tbb::blocked_range2d<Index> &r) {
          for (Index i = r.rows().begin(); i != r.rows().end(); ++i) {
            for (Index j = r.cols().begin(); j != r.cols().end(); ++j) {
              f(i, j);
            }
          }
        }, tbb::static_partitioner());
  }
#else
  for (Index i = 0; i < dim0; ++i) {
    for (Index j = 0; j < dim1; ++j) {
      f(i, j);
    }
  }
#endif
}

} // namespace tnn
