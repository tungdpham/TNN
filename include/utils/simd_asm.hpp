/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

float simd_dot_product_asm(const float *weights, const float *col_data,
                           size_t kernel_size);

#ifdef __cplusplus
}
#endif
