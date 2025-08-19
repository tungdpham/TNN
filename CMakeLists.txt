cmake_minimum_required(VERSION 3.18)
project(DCNN LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(ENABLE_OPENMP "Enable OpenMP support" ON)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_TBB "Enable Intel TBB support" OFF)
option(ENABLE_DEBUG "Enable debug build with sanitizers" OFF)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=haswell -mavx -mavx2 -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address -march=haswell -mavx -mavx2 -flto")

# Add pedantic warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")

# Debug override
if(ENABLE_DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Find packages
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

if(ENABLE_TBB)
    find_package(TBB REQUIRED)
    add_compile_definitions(USE_TBB)
endif()

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(CUDA_ENABLED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "-arch=sm_89 --compiler-options -fPIC")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/asio/asio/include)

# Collect header files (this is a header-only template library)
file(GLOB_RECURSE HEADERS
    "matrix/*.hpp"
    "nn/*.hpp"
    "nn/layers_impl/*.hpp"
    "tensor/*.hpp"
    "pipeline/*.hpp"
    "utils/*.hpp"
)

if(ENABLE_CUDA)
    file(GLOB_RECURSE CU_SOURCES
        "matrix/*.cu"
        "nn/*.cu"
        "utils/*.cu"
    )
endif()

# Create a header-only interface library
add_library(dcnn_lib INTERFACE)

# Link libraries to common library (INTERFACE for header-only)
target_link_libraries(dcnn_lib INTERFACE m)

if(ENABLE_OPENMP)
    target_link_libraries(dcnn_lib INTERFACE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_TBB)
    target_link_libraries(dcnn_lib INTERFACE TBB::tbb)
endif()

if(ENABLE_CUDA)
    target_link_libraries(dcnn_lib INTERFACE CUDA::cudart CUDA::cublas CUDA::curand)
endif()

# Function to create executables
function(create_executable target_name source_file)
    if(ENABLE_CUDA)
        add_executable(${target_name} ${source_file})
        set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES 89)
    else()
        add_executable(${target_name} ${source_file})
    endif()
    
    target_link_libraries(${target_name} PRIVATE dcnn_lib)
    
    # Add pthread for network-related executables
    if(${target_name} MATCHES "network|distributed|pipeline")
        target_link_libraries(${target_name} PRIVATE pthread)
    endif()
endfunction()

# Main executable
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    create_executable(main main.cpp)
endif()

# Training executables
create_executable(mnist_cnn_trainer mnist_cnn_trainer.cpp)
create_executable(cifar10_cnn_trainer cifar10_cnn_trainer.cpp)
create_executable(cifar100_cnn_trainer cifar100_cnn_trainer.cpp)
create_executable(uji_ips_trainer uji_ips_trainer.cpp)

# Test executables
create_executable(mnist_cnn_test mnist_cnn_test.cpp)
create_executable(pipeline_test pipeline_test.cpp)

# Network executables
create_executable(network_worker network_worker_main.cpp)
create_executable(distributed_pipeline_docker distributed_pipeline_docker.cpp)

# Unit tests
file(GLOB TEST_SOURCES "unit_tests/*.cpp")
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    create_executable(${test_name} ${test_source})
endforeach()

# Custom targets for convenience
add_custom_target(tests)
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_dependencies(tests ${test_name})
endforeach()

# Run tests target
add_custom_target(run_tests
    COMMENT "Running all tests"
    DEPENDS tests
)

foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_custom_command(TARGET run_tests POST_BUILD
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
        COMMENT "Running ${test_name}"
    )
endforeach()
