function(create_math_benchmark target_name source_file)
    add_executable(${target_name} ${source_file})

    # Link core DCNN libraries
    target_link_libraries(${target_name} PRIVATE
        dcnn_lib 
        dcnn_math 
        dcnn_tensor 
        dcnn_matrix 
        dcnn_utils
    )

    link_tbb(${target_name})
    link_cuda(${target_name})
    link_mkl(${target_name})
    link_windows_libs(${target_name})
endfunction()

create_executable(serialization_benchmark serialization_benchmark.cpp)
create_executable(compression_benchmark compression_benchmark.cpp)
create_math_benchmark(resize_benchmark resize_benchmark.cpp)
create_math_benchmark(reserve_benchmark reserve_benchmark.cpp)
create_math_benchmark(gemm_benchmark gemm_benchmark.cpp)
create_math_benchmark(im2col_benchmark im2col_benchmark.cpp)