# Unit Tests CMakeLists.txt

# Include Google Test
include(GoogleTest)

# Function to create a test executable
function(create_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    
    # Link with Google Test and your project libraries - ADD PRIVATE HERE
    target_link_libraries(${test_name} PRIVATE
        gtest_main
        gtest
        gmock
        tnn_lib
    )
    
    # Discover tests automatically
    gtest_discover_tests(${test_name})
endfunction()

set(TEST_FILES 
    tensor_test.cpp
    device_manager_test.cpp
    cuda_kernels_test.cpp
    gpu_tensor_test.cpp
    gpu_tensor_ops_test.cpp
    cuda_conv2d_ops_test.cpp
    cuda_dense_ops_test.cpp
    cuda_maxpool_ops_test.cpp
    cuda_loss_funcs_test.cpp
    cuda_batchnorm_ops_test.cpp
    layer_device_agnosticity_test.cpp
    avgpool_layer_test.cpp
    layer_buffer_reuse_test.cpp
    residual_block_test.cpp
    sequential_residual_block_test.cpp
    tiny_imagenet_loader_test.cpp
    conv2d_layer_test.cpp
    dense_layer_test.cpp
    maxpool2d_layer_test.cpp
    batchnorm_layer_test.cpp
    groupnorm_layer_test.cpp
)

foreach(test_file IN LISTS TEST_FILES)
    get_filename_component(test_name ${test_file} NAME_WE)
    create_test(${test_name})
endforeach()

# CUDA test (only if CUDA is enabled)
if(ENABLE_CUDA)
    function(create_cuda_test test_name)
        add_executable(${test_name} ${test_name}.cpp)
        
        # Set CUDA properties
        set_target_properties(${test_name} PROPERTIES 
            CUDA_ARCHITECTURES ${DETECTED_CUDA_ARCH}
            CUDA_STANDARD 17
            CUDA_SEPARABLE_COMPILATION ON
        )
        
        # Link with Google Test and your project libraries
        target_link_libraries(${test_name} PRIVATE
            gtest_main
            gtest
            gmock
            tnn_lib
        )
        
        # Discover tests automatically
        gtest_discover_tests(${test_name})
    endfunction()

    set(CUDA_TEST_TARGET cuda_kernels_test)
else()
    set(CUDA_TEST_TARGET "")
endif()

# Optional: Create a target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS tensor_test 
    device_manager_test
    gpu_tensor_test
    gpu_tensor_ops_test
    cuda_conv2d_ops_test
    cuda_dense_ops_test
    cuda_maxpool_ops_test
    cuda_loss_funcs_test
    cuda_batchnorm_ops_test
    layer_device_agnosticity_test
    avgpool_layer_test
    layer_buffer_reuse_test
    residual_block_test
    sequential_residual_block_test
    ${CUDA_TEST_TARGET}
    COMMENT "Running all unit tests"
)