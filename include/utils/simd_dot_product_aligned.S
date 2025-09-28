.section .text
.globl simd_dot_product_asm_aligned
#ifdef __ELF__
.type simd_dot_product_asm_aligned, @function
#endif

simd_dot_product_asm_aligned:
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
    jz .remainder_loop_aligned

.avx2_loop_aligned:
    # load 8 floats each from weights, col_data using aligned loads. mult them and accumulate with ymm0
    vmovaps (%rdi), %ymm1          # aligned load for weights
    vmovaps (%rsi), %ymm2          # aligned load for col_data
    vfmadd231ps %ymm2, %ymm1, %ymm0

    # advance pointer of weights, col_data
    addq $32, %rdi 
    addq $32, %rsi

    # subtract 8 from loop counter, go to loop start if not 0
    subq $8, %rcx
    jnz .avx2_loop_aligned

    # extract upper 128 bits from ymm0, add it with the lower 128 bits
    vextractf128 $1, %ymm0, %xmm1 
    vaddps %xmm1, %xmm0, %xmm0

    # horizontal add twice
    vhaddps %xmm0, %xmm0, %xmm0
    vhaddps %xmm0, %xmm0, %xmm0

.remainder_loop_aligned:
    testq %rax, %rax
    jz .finish_aligned

.scalar_loop_aligned:
    # the result of the multiplication is in xmm1
    decq %rax
    vmovss (%rdi,%rax,4), %xmm1
    vmulss (%rsi,%rax,4), %xmm1, %xmm1

    vaddss %xmm1, %xmm0, %xmm0     # add to the sum in xmm0

    testq %rax, %rax
    jnz .scalar_loop_aligned

.finish_aligned:
    vzeroupper
    ret

#ifdef __ELF__
.size simd_dot_product_asm_aligned, .-simd_dot_product_asm_aligned

.section .note.GNU-stack,"",@progbits # mark stack as non-executable
#endif
