# Unit Tests CMakeLists.txt

include(GoogleTest)

function(create_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    
    target_link_libraries(${test_name} PRIVATE
        gtest_main
        gtest
        gmock
        tnn_lib
    )
    
    gtest_discover_tests(${test_name} DISCOVERY_MODE PRE_TEST)
endfunction()

set(TEST_FILES 
    tensor_test.cpp
    slice_ops_test.cpp
    device_manager_test.cpp
    cuda_kernels_test.cpp
    gpu_tensor_test.cpp
    gpu_tensor_ops_test.cpp
    cuda_conv2d_nchw_ops_test.cpp
    cuda_dense_ops_test.cpp
    cuda_maxpool_ops_test.cpp
    cuda_loss_funcs_test.cpp
    cuda_batchnorm_ops_test.cpp
    layer_device_agnosticity_test.cpp
    layer_buffer_reuse_test.cpp
    residual_block_test.cpp
    sequential_residual_block_test.cpp
    tiny_imagenet_loader_test.cpp
    dense_layer_test.cpp
    legacy_maxpool2d_layer_test.cpp
    legacy_avgpool_layer_test.cpp
    legacy_batchnorm_layer_test.cpp
    legacy_conv2d_layer_test.cpp
    groupnorm_layer_test.cpp
    attention_block_test.cpp
)

foreach(test_file IN LISTS TEST_FILES)
    get_filename_component(test_name ${test_file} NAME_WE)
    create_test(${test_name})
endforeach()

# CUDA test (only if CUDA is enabled)
if(ENABLE_CUDA)
    function(create_cuda_test test_name)
        add_executable(${test_name} ${test_name}.cpp)
        
        set_target_properties(${test_name} PROPERTIES 
            CUDA_ARCHITECTURES ${DETECTED_CUDA_ARCH}
            CUDA_STANDARD 17
            CUDA_SEPARABLE_COMPILATION ON
        )
        
        target_link_libraries(${test_name} PRIVATE
            gtest_main
            gtest
            gmock
            tnn_lib
        )
        
        gtest_discover_tests(${test_name} DISCOVERY_MODE PRE_TEST)
    endfunction()

    set(CUDA_TEST_TARGET cuda_kernels_test)
else()
    set(CUDA_TEST_TARGET "")
endif()

add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS ${TEST_FILES}
    ${CUDA_TEST_TARGET}
    COMMENT "Running all unit tests"
)