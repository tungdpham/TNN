# Unit Tests CMakeLists.txt

# Include Google Test
include(GoogleTest)

# Function to create a test executable
function(create_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    
    # Link with Google Test and your project libraries
    target_link_libraries(${test_name} 
        gtest_main
        gtest
        gmock
    )
    
    # Add any additional libraries your tests might need
    if(ENABLE_TBB)
        target_link_libraries(${test_name} TBB::tbb)
    endif()
    
    if(ENABLE_OPENMP)
        target_link_libraries(${test_name} OpenMP::OpenMP_CXX)
    endif()
    
    if(ENABLE_CUDA)
        set_target_properties(${test_name} PROPERTIES 
            CUDA_ARCHITECTURES "89"
            CUDA_STANDARD 17
        )
        target_link_libraries(${test_name} CUDA::cudart)
    endif()
    
    # Include directories
    target_include_directories(${test_name} PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
    
    # Discover tests automatically
    gtest_discover_tests(${test_name})
endfunction()

# Add your test files here
create_test(tensor_tests)

# Optional: Create a target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS tensor_tests
    COMMENT "Running all unit tests"
)