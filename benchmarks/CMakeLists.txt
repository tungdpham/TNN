function(create_math_benchmark target_name source_file)
    add_executable(${target_name} ${source_file})

    # Link core TNN libraries
    target_link_libraries(${target_name} PRIVATE
        tnn_math 
        tnn_tensor 
        tnn_matrix 
        tnn_utils
    )
endfunction()

create_default(serialization_benchmark serialization_benchmark.cpp)
create_default(compression_benchmark compression_benchmark.cpp)
create_math_benchmark(resize_benchmark resize_benchmark.cpp)
create_math_benchmark(reserve_benchmark reserve_benchmark.cpp)
create_math_benchmark(gemm_benchmark gemm_benchmark.cpp)
create_math_benchmark(im2col_benchmark im2col_benchmark.cpp)