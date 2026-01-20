function(create_math_benchmark target_name source_file)
    add_executable(${target_name} ${source_file})

    # Link core TNN libraries
    target_link_libraries(${target_name} PRIVATE
        tnn_nn
        tnn_math 
        tnn_tensor 
        tnn_matrix 
        tnn_utils
        tnn_logging
    )

    if(ENABLE_CUDA)
        target_link_libraries(${target_name} PRIVATE tnn_cuda)
    endif()
endfunction()

create_default(serialization_benchmark serialization_benchmark.cpp)
create_default(tcp_communicator_benchmark tcp_communicator_benchmark.cpp)
create_math_benchmark(gemm_benchmark gemm_benchmark.cpp)
create_math_benchmark(conv2d_benchmark conv2d_benchmark.cpp)
create_math_benchmark(copy_benchmark copy_benchmark.cpp)
create_math_benchmark(pipeline_benchmark pipeline_benchmark.cpp)
