// Lightweight parallel-for helpers with optional Intel TBB backend.
#pragma once

#include <cstddef>
#include <functional>

#if defined(USE_TBB) || defined(USE_INTEL_TBB)
// Allow either the project macro `USE_TBB` or `USE_INTEL_TBB` to enable
// Intel/oneAPI TBB headers. Some installations expose headers under
// <oneapi/tbb/...> while others use <tbb/...>, so prefer the oneapi path
// when `USE_INTEL_TBB` is explicitly requested.
#if defined(USE_INTEL_TBB)
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/blocked_range2d.h>
#else
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#endif
#endif

namespace tnn {

// 1D range with optional index type (allows parallel_for_range<Index>(...))
template <typename Index = size_t, typename Func>
inline void parallel_for_range(Index begin, Index end, Func f) {
#if defined(USE_TBB)
  if (end > begin) {
    tbb::parallel_for(tbb::blocked_range<Index>(begin, end),
                      [&](const tbb::blocked_range<Index> &r) {
                        for (Index i = r.begin(); i != r.end(); ++i)
                          f(i);
                      });
  }
#else
  for (Index i = begin; i < end; ++i)
    f(i);
#endif
}

// 2D range: dims are [dim0, dim1], callback gets (i, j)
// 2D range with optional index type
template <typename Index = size_t, typename Func>
inline void parallel_for_2d(Index dim0, Index dim1, Func f) {
#if defined(USE_TBB)
  if (dim0 > 0 && dim1 > 0) {
    tbb::parallel_for(
        tbb::blocked_range2d<Index>(0, dim0, 0, dim1),
        [&](const tbb::blocked_range2d<Index> &r) {
          for (Index i = r.rows().begin(); i != r.rows().end(); ++i) {
            for (Index j = r.cols().begin(); j != r.cols().end(); ++j) {
              f(i, j);
            }
          }
        });
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
