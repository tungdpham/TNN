; Function: float simd_dot_product_asm(const float *weights, const float *col_data, size_t kernel_size)
; Windows x64 calling convention:
;   RCX: weights (const float*)
;   RDX: col_data (const float*)
;   R8:  kernel_size (size_t)
; Return value in XMM0

section .text
global simd_dot_product_asm

simd_dot_product_asm:
    ; Save non-volatile registers if needed
    ; sub rsp, 28h        ; Shadow space + alignment
    
    vpxor ymm0, ymm0, ymm0      ; sum_vec = 0

    ; simd_end = kernel_size - (kernel_size % 8)
    mov rax, r8                 ; rax = kernel_size
    and r8, 7                   ; r8 = kernel_size % 8 (remainder)
    sub rax, r8                 ; rax = simd_end
    mov r9, rax                 ; r9 = loop counter

    ; jump to remainder loop if simd_end is 0
    test r9, r9
    jz remainder_loop

avx2_loop:
    ; load 8 floats each from weights, col_data, multiply and accumulate
    vmovups ymm1, [rcx]         ; load 8 floats from weights
    vmovups ymm2, [rdx]         ; load 8 floats from col_data
    vfmadd231ps ymm0, ymm1, ymm2 ; accumulate: ymm0 += ymm1 * ymm2

    ; advance pointers
    add rcx, 32                 ; advance weights pointer (8 * 4 bytes)
    add rdx, 32                 ; advance col_data pointer (8 * 4 bytes)

    ; decrement loop counter and continue if not zero
    sub r9, 8
    jnz avx2_loop

    ; Horizontal sum of ymm0
    vextractf128 xmm1, ymm0, 1  ; extract upper 128 bits
    vaddps xmm0, xmm0, xmm1     ; add upper and lower 128 bits
    vhaddps xmm0, xmm0, xmm0    ; horizontal add twice
    vhaddps xmm0, xmm0, xmm0

remainder_loop:
    test r8, r8                 ; check if remainder is 0
    jz finish

scalar_loop:
    ; Process remaining elements one by one
    dec r8
    vmovss xmm1, [rcx + r8*4]   ; load weights[i]
    vmulss xmm1, xmm1, [rdx + r8*4] ; multiply with col_data[i]
    vaddss xmm0, xmm0, xmm1     ; add to sum

    test r8, r8
    jnz scalar_loop

finish:
    vzeroupper                  ; clear upper bits of YMM registers
    ; add rsp, 28h              ; restore stack
    ret
