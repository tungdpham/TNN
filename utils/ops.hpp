#include <stdlib.h>

template <typename T>
inline void dot(const T* a, const T* b, T* result,
                   size_t m, size_t n, size_t k) {
    // Matrix multiplication: result = a * b
    // a: m x k, b: k x n, result: m x n
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1)
#endif
    for (size_t i = 0; i < m; ++i) {    
        for (size_t j = 0; j < n; ++j) {
            T sum = 0;
            for (size_t l = 0; l < k; ++l) {
                sum += a[i * k + l] * b[l * n + j];
            }
            result[i * n + j] = sum;
        }
    }
}