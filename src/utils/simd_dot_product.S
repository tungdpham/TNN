.section .text
.globl simd_dot_product_asm
#ifdef __ELF__
.type simd_dot_product_asm, @function
#endif

simd_dot_product_asm:
    # set all accumulators to zero, we will do 4 accumulators to unroll the loop
    vpxor %ymm0, %ymm0, %ymm0      
    vpxor %ymm3, %ymm3, %ymm3      
    vpxor %ymm6, %ymm6, %ymm6      
    vpxor %ymm9, %ymm9, %ymm9      

    # calculate bounds, r8 = kernel_size % 32, r9 = kernel_size - r8
    movq %rdx, %r8          
    andq $0x1F, %r8
    movq %rdx, %r9                 
    subq %r8, %r9                  

    testq %r9, %r9
    jz .check_regular_loop

.avx2_extended_loop:
    # unrolled 4 times, each iteration now do 32 elements (128 bytes)
    vmovups (%rdi), %ymm1          
    vmovups (%rsi), %ymm2          
    vfmadd231ps %ymm2, %ymm1, %ymm0
    
    vmovups 32(%rdi), %ymm4
    vmovups 32(%rsi), %ymm5
    vfmadd231ps %ymm5, %ymm4, %ymm3
    
    vmovups 64(%rdi), %ymm7
    vmovups 64(%rsi), %ymm8
    vfmadd231ps %ymm8, %ymm7, %ymm6

    vmovups 96(%rdi), %ymm10
    vmovups 96(%rsi), %ymm11
    vfmadd231ps %ymm11, %ymm10, %ymm9
    
    addq $128, %rdi
    addq $128, %rsi

    subq $32, %r9
    jnz .avx2_extended_loop

    # combine the four accumulators
    vaddps %ymm3, %ymm0, %ymm0
    vaddps %ymm6, %ymm0, %ymm0
    vaddps %ymm9, %ymm0, %ymm0

.check_regular_loop:
    # calculate bounds for regular loop, r8 = remaining elements after 32-element chunks
    movq %r8, %rax                 
    andq $0x7, %rax                
    subq %rax, %r8                 
    movq %r8, %rcx
    
    testq %rcx, %rcx
    jz .remainder_loop

.avx2_loop:
    # regular loop, 8 elements (32 bytes) per iteration
    vmovups (%rdi), %ymm1
    vmovups (%rsi), %ymm2
    vfmadd231ps %ymm2, %ymm1, %ymm0
    
    addq $32, %rdi
    addq $32, %rsi
    
    subq $8, %rcx
    jnz .avx2_loop

.remainder_loop:
    testq %rax, %rax
    jz .finish

.scalar_loop:
    # scalar loop for remaining elements
    decq %rax
    vmovss (%rdi,%rax,4), %xmm1
    vmulss (%rsi,%rax,4), %xmm1, %xmm1
    vaddss %xmm1, %xmm0, %xmm0     

    testq %rax, %rax
    jnz .scalar_loop

.finish:
    vextractf128 $1, %ymm0, %xmm1  
    vaddps %xmm1, %xmm0, %xmm0     
    vhaddps %xmm0, %xmm0, %xmm0    
    vhaddps %xmm0, %xmm0, %xmm0    

    vzeroupper                     
    ret

#ifdef __ELF__
.size simd_dot_product_asm, .-simd_dot_product_asm

.section .note.GNU-stack,"",@progbits
#endif