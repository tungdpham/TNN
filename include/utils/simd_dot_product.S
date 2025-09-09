# Function: float simd_dot_product_asm(const float *weights, const float *col_data, size_t kernel_size)
# Arguments (System V ABI):
#   %rdi: weights (const float*)
#   %rsi: col_data (const float*)
#   %rdx: kernel_size (size_t)

.section .text
.globl simd_dot_product_asm
.type simd_dot_product_asm, @function

simd_dot_product_asm:
    # save registers if needed
    # pushq %rbp
    # movq %rsp, %rbp
    
    vpxor %ymm0, %ymm0, %ymm0      # sum_vec = 0

    # simd_end = kernel_size - (kernel_size % 8)
    movq %rdx, %rax
    andq $0x7, %rax
    subq %rax, %rdx 
    movq %rdx, %rcx

    # jump to remainder loop if simd_end is 0
    testq %rcx, %rcx
    jz .remainder_loop

.avx2_loop:
    # load 4 floats each from weights, col_data. mult them and accumulate with ymm0
    vmovups (%rdi), %ymm1
    vmovups (%rsi), %ymm2
    vfmadd231ps %ymm2, %ymm1, %ymm0

    # advance pointer of weights, col_data
    addq $32, %rdi 
    addq $32, %rsi

    # subtract 8 from loop counter, go to loop start if not 0
    subq $8, %rcx
    jnz .avx2_loop

    # extract upper 128 bits from ymm0, add it with the lower 128 bits
    vextractf128 $1, %ymm0, %xmm1 
    vaddps %xmm1, %xmm0, %xmm0

    # horizontal add twice
    vhaddps %xmm0, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0

.remainder_loop:
    testq %rax, %rax
    jz .finish

.scalar_loop:
    # the result of the multiplication is in XMM1
    decq %rax
    vmovss (%rdi,%rax,4), %xmm1
    vmulss (%rsi,%rax,4), %xmm1, %xmm1

    vaddss %xmm1, %xmm0, %xmm0     # add to the sum in XMM0

    testq %rax, %rax
    jnz .scalar_loop

.finish:
    vzeroupper
    ret

.size simd_dot_product_asm, .-simd_dot_product_asm

.section .note.GNU-stack,"",@progbits # mark stack as non-executable