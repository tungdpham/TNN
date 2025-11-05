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

    link_tbb(${test_name})
    link_cuda(${test_name})
    link_mkl(${test_name})
    link_windows_libs(${test_name})
    
    # Discover tests automatically
    gtest_discover_tests(${test_name})
endfunction()

set(TEST_FILES 
    tensor_test.cpp
    device_manager_test.cpp
)

foreach(test_file IN LISTS TEST_FILES)
    get_filename_component(test_name ${test_file} NAME_WE)
    create_test(${test_name})
endforeach()

# Optional: Create a target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS tensor_test 
    device_manager_test
    COMMENT "Running all unit tests"
)