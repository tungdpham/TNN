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
    ${CUDA_TEST_TARGET}
    COMMENT "Running all unit tests"
)