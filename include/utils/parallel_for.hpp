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
inline void parallel_for(const Index begin, const Index end, Func f) {
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

// Strided parallel_for overloads for better memory access patterns
template <typename Index = size_t, typename Func>
inline void parallel_for_strided(const Index begin, const Index end, const Index stride, Func f) {
  assert(end >= begin && stride > 0 && "Invalid range or stride");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
  for (Index i = begin; i < end; i += stride) {
    f(i);
  }
#elif defined(USE_TBB)
  // Calculate number of elements accounting for stride
  const Index num_elements = (end - begin + stride - 1) / stride;
  tbb::parallel_for(
      tbb::blocked_range<Index>(0, num_elements),
      [&](const tbb::blocked_range<Index> &r) {
        for (Index idx = r.begin(); idx != r.end(); ++idx) {
          const Index i = begin + idx * stride;
          if (i < end) {
            f(i);
          }
        }
      },
      tbb::static_partitioner());
#else
  for (Index i = begin; i < end; i += stride) {
    f(i);
  }
#endif
}

template <typename Index = size_t, typename Func>
inline void parallel_for_2d_strided(const Index dim0, const Index dim1, 
                                   const Index stride0, const Index stride1, Func f) {
  assert(dim0 >= 0 && dim1 >= 0 && stride0 > 0 && stride1 > 0 && "Invalid dimensions or strides");
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
  for (Index i = 0; i < dim0; i += stride0) {
    for (Index j = 0; j < dim1; j += stride1) {
      f(i, j);
    }
  }
#elif defined(USE_TBB)
  // Calculate number of elements accounting for strides
  const Index num_elements0 = (dim0 + stride0 - 1) / stride0;
  const Index num_elements1 = (dim1 + stride1 - 1) / stride1;
  
  tbb::parallel_for(
      tbb::blocked_range2d<Index>(0, num_elements0, 0, num_elements1),
      [&](const tbb::blocked_range2d<Index> &r) {
        for (Index idx0 = r.rows().begin(); idx0 != r.rows().end(); ++idx0) {
          for (Index idx1 = r.cols().begin(); idx1 != r.cols().end(); ++idx1) {
            const Index i = idx0 * stride0;
            const Index j = idx1 * stride1;
            if (i < dim0 && j < dim1) {
              f(i, j);
            }
          }
        }
      },
      tbb::static_partitioner());
#else
  std::cout << "Warning: Running parallel_for_2d_strided in serial mode.\n";
  for (Index i = 0; i < dim0; i += stride0) {
    for (Index j = 0; j < dim1; j += stride1) {
      f(i, j);
    }
  }
#endif
}

// Range-based strided parallel_for with custom begin/end for each dimension
template <typename Index = size_t, typename Func>
inline void parallel_for_2d_range_strided(const Index begin0, const Index end0, const Index stride0,
                                         const Index begin1, const Index end1, const Index stride1, Func f) {
  assert(end0 >= begin0 && end1 >= begin1 && stride0 > 0 && stride1 > 0 && "Invalid ranges or strides");
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
  for (Index i = begin0; i < end0; i += stride0) {
    for (Index j = begin1; j < end1; j += stride1) {
      f(i, j);
    }
  }
#elif defined(USE_TBB)
  // Calculate number of elements accounting for strides
  const Index num_elements0 = (end0 - begin0 + stride0 - 1) / stride0;
  const Index num_elements1 = (end1 - begin1 + stride1 - 1) / stride1;
  
  tbb::parallel_for(
      tbb::blocked_range2d<Index>(0, num_elements0, 0, num_elements1),
      [&](const tbb::blocked_range2d<Index> &r) {
        for (Index idx0 = r.rows().begin(); idx0 != r.rows().end(); ++idx0) {
          for (Index idx1 = r.cols().begin(); idx1 != r.cols().end(); ++idx1) {
            const Index i = begin0 + idx0 * stride0;
            const Index j = begin1 + idx1 * stride1;
            if (i < end0 && j < end1) {
              f(i, j);
            }
          }
        }
      },
      tbb::static_partitioner());
#else
  std::cout << "Warning: Running parallel_for_2d_range_strided in serial mode.\n";
  for (Index i = begin0; i < end0; i += stride0) {
    for (Index j = begin1; j < end1; j += stride1) {
      f(i, j);
    }
  }
#endif
}

// Specialized overload for memory-aligned strided access (e.g., processing every 4th, 8th element)
template <typename Index = size_t, typename Func>
inline void parallel_for_aligned_strided(const Index begin, const Index end, 
                                        const Index stride, const Index alignment, Func f) {
  assert(end >= begin && stride > 0 && alignment > 0 && "Invalid parameters");
  // Align the starting position to the specified boundary
  const Index aligned_begin = ((begin + alignment - 1) / alignment) * alignment;
  
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
  for (Index i = aligned_begin; i < end; i += stride) {
    f(i);
  }
#elif defined(USE_TBB)
  const Index num_elements = (end - aligned_begin + stride - 1) / stride;
  tbb::parallel_for(
      tbb::blocked_range<Index>(0, num_elements),
      [&](const tbb::blocked_range<Index> &r) {
        for (Index idx = r.begin(); idx != r.end(); ++idx) {
          const Index i = aligned_begin + idx * stride;
          if (i < end) {
            f(i);
          }
        }
      },
      tbb::static_partitioner());
#else
  for (Index i = aligned_begin; i < end; i += stride) {
    f(i);
  }
#endif
}

} // namespace utils
